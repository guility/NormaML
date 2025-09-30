"""
Потоковая обработка данных без промежуточной сериализации.

Обеспечивает chunked reading, прямое создание polars DataFrames 
из numpy chunks и memory-efficient processing.
"""

import io
import os
import mmap
from pathlib import Path
from typing import Iterator, Optional, Union, Dict, Any, List, Callable, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
import threading
from contextlib import contextmanager
import gc

try:
    import numpy as np
except ImportError:
    raise ImportError("numpy is required for streaming data processing")

try:
    import polars as pl
except ImportError:
    raise ImportError("polars is required for streaming data processing")

from .zero_copy import get_zero_copy_manager, DataStatus
from .shared_memory import get_shared_memory_manager

logger = logging.getLogger(__name__)


class ChunkFormat(Enum):
    """Форматы chunks для потоковой обработки."""
    NUMPY = "numpy"
    POLARS = "polars"
    RAW_BYTES = "raw_bytes"


@dataclass
class ChunkMetadata:
    """Метаданные для chunk данных."""
    chunk_id: int
    size_bytes: int
    row_count: int
    column_count: int
    dtypes: Dict[str, np.dtype]
    memory_offset: int = 0
    is_final: bool = False
    processing_time: Optional[float] = None


class DataSource:
    """Базовый класс для источников данных."""
    
    def __init__(self, chunk_size: int = 10000):
        self.chunk_size = chunk_size
        self._closed = False
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    def read_chunk(self) -> Optional[np.ndarray]:
        """Прочитать следующий chunk как numpy array."""
        raise NotImplementedError
    
    def get_metadata(self) -> Dict[str, Any]:
        """Получить метаданные источника."""
        raise NotImplementedError
    
    def close(self):
        """Закрыть источник данных."""
        self._closed = True


class MemoryMappedFileSource(DataSource):
    """
    Источник данных на основе memory-mapped file.
    
    Обеспечивает zero-copy чтение файлов через mmap.
    """
    
    def __init__(self, 
                 file_path: Union[str, Path], 
                 dtype: np.dtype,
                 chunk_size: int = 10000,
                 shape: Optional[Tuple[int, ...]] = None):
        super().__init__(chunk_size)
        self.file_path = Path(file_path)
        self.dtype = dtype
        self.shape = shape
        
        # Открыть файл и создать memory mapping
        self._file = open(self.file_path, 'rb')
        self._mmap = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
        
        # Определить размеры
        file_size = len(self._mmap)
        element_size = dtype.itemsize
        
        if shape is None:
            # Предполагаем 1D массив
            self.total_elements = file_size // element_size
            self.array_shape = (self.total_elements,)
        else:
            self.array_shape = shape
            self.total_elements = np.prod(shape)
        
        self._current_offset = 0
        self._elements_per_chunk = chunk_size
        
        logger.debug(f"Инициализирован memory-mapped source: {file_path}, "
                    f"размер: {file_size} байт, элементов: {self.total_elements}")
    
    def read_chunk(self) -> Optional[np.ndarray]:
        """Прочитать следующий chunk через memory mapping (zero-copy)."""
        if self._closed or self._current_offset >= self.total_elements:
            return None
        
        # Вычислить размер chunk'а
        remaining_elements = self.total_elements - self._current_offset
        chunk_elements = min(self._elements_per_chunk, remaining_elements)
        
        # Вычислить байтовые offset'ы
        byte_offset = self._current_offset * self.dtype.itemsize
        chunk_bytes = chunk_elements * self.dtype.itemsize
        
        # Создать numpy array из memory view (zero-copy)
        memory_view = memoryview(self._mmap)[byte_offset:byte_offset + chunk_bytes]
        chunk_array = np.frombuffer(memory_view, dtype=self.dtype)
        
        # Обновить offset
        self._current_offset += chunk_elements
        
        logger.debug(f"Прочитан chunk: {chunk_elements} элементов, "
                    f"offset: {self._current_offset}/{self.total_elements}")
        
        return chunk_array
    
    def get_metadata(self) -> Dict[str, Any]:
        """Получить метаданные файла."""
        return {
            'file_path': str(self.file_path),
            'file_size': len(self._mmap),
            'dtype': str(self.dtype),
            'shape': self.array_shape,
            'total_elements': self.total_elements,
            'chunk_size': self._elements_per_chunk,
            'current_offset': self._current_offset
        }
    
    def close(self):
        """Закрыть memory mapping и файл."""
        if not self._closed:
            if hasattr(self, '_mmap'):
                self._mmap.close()
            if hasattr(self, '_file'):
                self._file.close()
            super().close()
            logger.debug(f"Закрыт memory-mapped source: {self.file_path}")


class CSVStreamSource(DataSource):
    """
    Источник данных для потокового чтения CSV файлов.
    
    Использует polars lazy reading с конвертацией в numpy chunks.
    """
    
    def __init__(self, 
                 file_path: Union[str, Path],
                 chunk_size: int = 10000,
                 separator: str = ',',
                 has_header: bool = True,
                 dtype_hints: Optional[Dict[str, np.dtype]] = None):
        super().__init__(chunk_size)
        self.file_path = Path(file_path)
        self.separator = separator
        self.has_header = has_header
        self.dtype_hints = dtype_hints or {}
        
        # Инициализировать lazy frame
        self._lazy_frame = pl.scan_csv(
            self.file_path,
            separator=separator,
            has_header=has_header
        )
        
        # Получить информацию о схеме
        try:
            # Прочитать небольшой sample для определения схемы
            sample = self._lazy_frame.head(100).collect()
            self.columns = sample.columns
            self.polars_dtypes = {col: sample[col].dtype for col in self.columns}
            self.numpy_dtypes = self._convert_polars_to_numpy_dtypes()
            self.total_rows = self._estimate_total_rows()
        except Exception as e:
            logger.error(f"Ошибка инициализации CSV source: {e}")
            raise
        
        self._current_offset = 0
        
        logger.debug(f"Инициализирован CSV source: {file_path}, "
                    f"колонок: {len(self.columns)}, строк: ~{self.total_rows}")
    
    def _convert_polars_to_numpy_dtypes(self) -> Dict[str, np.dtype]:
        """Конвертировать polars dtypes в numpy dtypes."""
        mapping = {}
        
        for col, pl_dtype in self.polars_dtypes.items():
            if col in self.dtype_hints:
                mapping[col] = self.dtype_hints[col]
            elif pl_dtype == pl.Int64:
                mapping[col] = np.dtype('int64')
            elif pl_dtype == pl.Int32:
                mapping[col] = np.dtype('int32')
            elif pl_dtype == pl.Float64:
                mapping[col] = np.dtype('float64')
            elif pl_dtype == pl.Float32:
                mapping[col] = np.dtype('float32')
            elif pl_dtype == pl.Boolean:
                mapping[col] = np.dtype('bool')
            elif pl_dtype == pl.Utf8:
                mapping[col] = np.dtype('object')
            else:
                # Fallback
                mapping[col] = np.dtype('object')
        
        return mapping
    
    def _estimate_total_rows(self) -> int:
        """Оценить общее количество строк в файле."""
        try:
            # Быстрая оценка на основе размера файла
            file_size = self.file_path.stat().st_size
            # Прочитать несколько строк для оценки среднего размера
            sample = self._lazy_frame.head(1000).collect()
            if len(sample) == 0:
                return 0
            
            # Оценить средний размер строки
            avg_row_size = file_size / (len(sample) * (file_size / self.file_path.stat().st_size))
            estimated_rows = int(file_size / avg_row_size) if avg_row_size > 0 else len(sample)
            
            return max(estimated_rows, len(sample))
        except Exception:
            return -1  # Неизвестно
    
    def read_chunk(self) -> Optional[Dict[str, np.ndarray]]:
        """Прочитать следующий chunk как словарь numpy arrays."""
        if self._closed or (self.total_rows > 0 and self._current_offset >= self.total_rows):
            return None
        
        try:
            # Прочитать chunk с использованием polars lazy evaluation
            chunk_df = (self._lazy_frame
                       .slice(self._current_offset, self.chunk_size)
                       .collect())
            
            if len(chunk_df) == 0:
                return None
            
            # Конвертировать в numpy arrays (zero-copy где возможно)
            chunk_arrays = {}
            for col in self.columns:
                series = chunk_df[col]
                # Прямое извлечение numpy array из polars Series
                numpy_array = series.to_numpy()
                
                # Попытаться привести к желаемому dtype
                target_dtype = self.numpy_dtypes.get(col)
                if target_dtype and numpy_array.dtype != target_dtype:
                    try:
                        numpy_array = numpy_array.astype(target_dtype)
                    except (ValueError, TypeError):
                        # Оставить оригинальный dtype если конверсия не удалась
                        pass
                
                chunk_arrays[col] = numpy_array
            
            self._current_offset += len(chunk_df)
            
            logger.debug(f"Прочитан CSV chunk: {len(chunk_df)} строк, "
                        f"offset: {self._current_offset}")
            
            return chunk_arrays
            
        except Exception as e:
            logger.error(f"Ошибка чтения CSV chunk: {e}")
            return None
    
    def get_metadata(self) -> Dict[str, Any]:
        """Получить метаданные CSV источника."""
        return {
            'file_path': str(self.file_path),
            'file_size': self.file_path.stat().st_size,
            'columns': self.columns,
            'polars_dtypes': {k: str(v) for k, v in self.polars_dtypes.items()},
            'numpy_dtypes': {k: str(v) for k, v in self.numpy_dtypes.items()},
            'estimated_rows': self.total_rows,
            'chunk_size': self.chunk_size,
            'current_offset': self._current_offset
        }


class StreamingDataProcessor:
    """
    Основной процессор для потоковой обработки данных.
    
    Обеспечивает chunked reading, memory-efficient processing
    и прямое создание polars DataFrames из numpy chunks.
    """
    
    def __init__(self, 
                 max_memory_mb: int = 1024,
                 enable_zero_copy: bool = True,
                 auto_cleanup: bool = True):
        self.max_memory_mb = max_memory_mb
        self.enable_zero_copy = enable_zero_copy
        self.auto_cleanup = auto_cleanup
        
        self._zero_copy_manager = get_zero_copy_manager() if enable_zero_copy else None
        self._processing_stats = {
            'chunks_processed': 0,
            'total_rows': 0,
            'total_bytes': 0,
            'processing_time': 0.0
        }
        self._lock = threading.Lock()
        
        logger.info(f"Инициализирован StreamingDataProcessor: "
                   f"max_memory={max_memory_mb}MB, zero_copy={enable_zero_copy}")
    
    def process_file(self, 
                    file_path: Union[str, Path],
                    processor_func: Callable[[Union[np.ndarray, Dict[str, np.ndarray]]], Any],
                    file_type: str = 'auto',
                    chunk_size: int = 10000,
                    **kwargs) -> Iterator[Tuple[ChunkMetadata, Any]]:
        """
        Обработать файл по chunks с применением функции обработки.
        
        Args:
            file_path: путь к файлу
            processor_func: функция для обработки каждого chunk'а
            file_type: тип файла ('csv', 'binary', 'auto')
            chunk_size: размер chunk'а
            **kwargs: дополнительные параметры для источника данных
            
        Yields:
            Tuple[ChunkMetadata, Any]: метаданные chunk'а и результат обработки
        """
        file_path = Path(file_path)
        
        # Определить тип файла
        if file_type == 'auto':
            if file_path.suffix.lower() == '.csv':
                file_type = 'csv'
            else:
                file_type = 'binary'
        
        # Создать источник данных
        if file_type == 'csv':
            data_source = CSVStreamSource(file_path, chunk_size=chunk_size, **kwargs)
        elif file_type == 'binary':
            dtype = kwargs.get('dtype', np.float64)
            shape = kwargs.get('shape', None)
            data_source = MemoryMappedFileSource(file_path, dtype, chunk_size, shape)
        else:
            raise ValueError(f"Неподдерживаемый тип файла: {file_type}")
        
        try:
            with data_source:
                yield from self._process_source(data_source, processor_func)
        finally:
            if self.auto_cleanup:
                self._cleanup_memory()
    
    def process_numpy_chunks(self,
                           chunks: Iterator[np.ndarray],
                           processor_func: Callable[[np.ndarray], Any],
                           chunk_metadata: Optional[List[ChunkMetadata]] = None) -> Iterator[Tuple[ChunkMetadata, Any]]:
        """
        Обработать итератор numpy chunks.
        
        Args:
            chunks: итератор numpy arrays
            processor_func: функция обработки
            chunk_metadata: опциональные метаданные для chunks
            
        Yields:
            Tuple[ChunkMetadata, Any]: метаданные и результат обработки
        """
        chunk_id = 0
        
        for i, chunk in enumerate(chunks):
            # Создать метаданные если не предоставлены
            if chunk_metadata and i < len(chunk_metadata):
                metadata = chunk_metadata[i]
            else:
                metadata = ChunkMetadata(
                    chunk_id=chunk_id,
                    size_bytes=chunk.nbytes,
                    row_count=len(chunk) if chunk.ndim >= 1 else 1,
                    column_count=1,
                    dtypes={'data': chunk.dtype}
                )
            
            # Обработать chunk
            start_time = time.time()
            
            try:
                # Опционально сохранить в zero-copy storage
                if self.enable_zero_copy and self._zero_copy_manager:
                    chunk_name = f"streaming_chunk_{chunk_id}"
                    self._zero_copy_manager.store_array(chunk, chunk_name, tags={'streaming', 'temporary'})
                
                # Применить функцию обработки
                result = processor_func(chunk)
                
                # Обновить метаданные
                metadata.processing_time = time.time() - start_time
                
                # Обновить статистику
                with self._lock:
                    self._processing_stats['chunks_processed'] += 1
                    self._processing_stats['total_rows'] += metadata.row_count
                    self._processing_stats['total_bytes'] += metadata.size_bytes
                    self._processing_stats['processing_time'] += metadata.processing_time
                
                yield metadata, result
                
            except Exception as e:
                logger.error(f"Ошибка обработки chunk {chunk_id}: {e}")
                metadata.processing_time = time.time() - start_time
                yield metadata, None
            
            finally:
                chunk_id += 1
                
                # Периодическая очистка памяти
                if self.auto_cleanup and chunk_id % 10 == 0:
                    self._cleanup_memory()
    
    def create_dataframe_from_chunks(self,
                                   column_chunks: Dict[str, Iterator[np.ndarray]],
                                   batch_size: int = 100000) -> Iterator[pl.DataFrame]:
        """
        Создать polars DataFrames из column chunks.
        
        Args:
            column_chunks: словарь колонка -> итератор numpy arrays
            batch_size: размер batch'а для создания DataFrame
            
        Yields:
            polars DataFrame для каждого batch'а
        """
        # Подготовить итераторы
        column_names = list(column_chunks.keys())
        iterators = {col: iter(chunks) for col, chunks in column_chunks.items()}
        
        batch_data = {col: [] for col in column_names}
        current_batch_size = 0
        
        try:
            while True:
                # Попытаться получить следующий chunk для каждой колонки
                chunk_available = False
                
                for col in column_names:
                    try:
                        chunk = next(iterators[col])
                        batch_data[col].append(chunk)
                        chunk_available = True
                        
                        if current_batch_size == 0:
                            current_batch_size = len(chunk)
                        
                    except StopIteration:
                        # Этот итератор исчерпан
                        continue
                
                if not chunk_available:
                    # Все итераторы исчерпаны
                    break
                
                # Проверить достиг ли batch нужного размера
                if current_batch_size >= batch_size:
                    # Создать DataFrame
                    df = self._create_dataframe_from_batch(batch_data, column_names)
                    if df is not None:
                        yield df
                    
                    # Сброс batch'а
                    batch_data = {col: [] for col in column_names}
                    current_batch_size = 0
        
        finally:
            # Обработать оставшиеся данные
            if current_batch_size > 0:
                df = self._create_dataframe_from_batch(batch_data, column_names)
                if df is not None:
                    yield df
    
    def _process_source(self, 
                       data_source: DataSource,
                       processor_func: Callable) -> Iterator[Tuple[ChunkMetadata, Any]]:
        """Обработать источник данных."""
        chunk_id = 0
        
        while True:
            chunk = data_source.read_chunk()
            if chunk is None:
                break
            
            # Создать метаданные
            if isinstance(chunk, dict):
                # CSV chunk (словарь numpy arrays)
                total_bytes = sum(arr.nbytes for arr in chunk.values())
                row_count = len(next(iter(chunk.values()))) if chunk else 0
                dtypes = {col: arr.dtype for col, arr in chunk.items()}
            else:
                # Binary chunk (numpy array)
                total_bytes = chunk.nbytes
                row_count = len(chunk) if chunk.ndim >= 1 else 1
                dtypes = {'data': chunk.dtype}
            
            metadata = ChunkMetadata(
                chunk_id=chunk_id,
                size_bytes=total_bytes,
                row_count=row_count,
                column_count=len(dtypes),
                dtypes=dtypes
            )
            
            # Обработать chunk
            start_time = time.time()
            
            try:
                result = processor_func(chunk)
                metadata.processing_time = time.time() - start_time
                
                # Обновить статистику
                with self._lock:
                    self._processing_stats['chunks_processed'] += 1
                    self._processing_stats['total_rows'] += row_count
                    self._processing_stats['total_bytes'] += total_bytes
                    self._processing_stats['processing_time'] += metadata.processing_time
                
                yield metadata, result
                
            except Exception as e:
                logger.error(f"Ошибка обработки chunk {chunk_id}: {e}")
                metadata.processing_time = time.time() - start_time
                yield metadata, None
            
            chunk_id += 1
    
    def _create_dataframe_from_batch(self, 
                                   batch_data: Dict[str, List[np.ndarray]],
                                   column_names: List[str]) -> Optional[pl.DataFrame]:
        """Создать polars DataFrame из batch данных."""
        try:
            # Объединить chunks для каждой колонки
            combined_data = {}
            
            for col in column_names:
                if col in batch_data and batch_data[col]:
                    # Объединить numpy arrays для колонки (zero-copy где возможно)
                    combined_array = np.concatenate(batch_data[col])
                    combined_data[col] = combined_array
            
            if not combined_data:
                return None
            
            # Создать polars DataFrame
            # Polars может использовать zero-copy для некоторых numpy dtypes
            df = pl.DataFrame(combined_data)
            
            logger.debug(f"Создан DataFrame: {len(df)} строк, {len(df.columns)} колонок")
            return df
            
        except Exception as e:
            logger.error(f"Ошибка создания DataFrame: {e}")
            return None
    
    def _cleanup_memory(self):
        """Очистить память от временных данных."""
        try:
            # Очистить temporary данные из zero-copy manager
            if self.enable_zero_copy and self._zero_copy_manager:
                # Найти временные данные по тегу
                temp_data = self._zero_copy_manager.list_data(tags={'temporary'})
                for item in temp_data:
                    self._zero_copy_manager.remove(item['id'])
            
            # Принудительная сборка мусора
            gc.collect()
            
        except Exception as e:
            logger.warning(f"Ошибка при очистке памяти: {e}")
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Получить статистику обработки."""
        with self._lock:
            stats = self._processing_stats.copy()
            
            # Добавить дополнительные метрики
            if stats['chunks_processed'] > 0:
                stats['avg_chunk_size_bytes'] = stats['total_bytes'] / stats['chunks_processed']
                stats['avg_processing_time_per_chunk'] = stats['processing_time'] / stats['chunks_processed']
                stats['throughput_mb_per_sec'] = (stats['total_bytes'] / (1024 * 1024)) / max(stats['processing_time'], 0.001)
            
            return stats
    
    @contextmanager
    def processing_session(self):
        """Контекстный менеджер для сессии обработки."""
        logger.info("Начало сессии потоковой обработки")
        start_time = time.time()
        
        try:
            yield self
        finally:
            # Финальная очистка
            if self.auto_cleanup:
                self._cleanup_memory()
            
            # Логирование статистики
            session_time = time.time() - start_time
            stats = self.get_processing_stats()
            logger.info(f"Завершена сессия обработки: {session_time:.2f}s, "
                       f"chunks: {stats.get('chunks_processed', 0)}, "
                       f"rows: {stats.get('total_rows', 0)}, "
                       f"throughput: {stats.get('throughput_mb_per_sec', 0):.2f} MB/s")


# Convenience функции

def process_csv_streaming(file_path: Union[str, Path],
                         processor_func: Callable,
                         chunk_size: int = 10000,
                         **kwargs) -> Iterator[Tuple[ChunkMetadata, Any]]:
    """Потоковая обработка CSV файла."""
    processor = StreamingDataProcessor()
    
    with processor.processing_session():
        yield from processor.process_file(
            file_path, processor_func, 
            file_type='csv', chunk_size=chunk_size, **kwargs
        )


def process_binary_streaming(file_path: Union[str, Path],
                           processor_func: Callable,
                           dtype: np.dtype,
                           chunk_size: int = 10000,
                           shape: Optional[Tuple[int, ...]] = None) -> Iterator[Tuple[ChunkMetadata, Any]]:
    """Потоковая обработка бинарного файла."""
    processor = StreamingDataProcessor()
    
    with processor.processing_session():
        yield from processor.process_file(
            file_path, processor_func,
            file_type='binary', chunk_size=chunk_size,
            dtype=dtype, shape=shape
        )


def create_streaming_dataframe(column_chunks: Dict[str, Iterator[np.ndarray]],
                             batch_size: int = 100000) -> Iterator[pl.DataFrame]:
    """Создать потоковые polars DataFrames из column chunks."""
    processor = StreamingDataProcessor()
    
    with processor.processing_session():
        yield from processor.create_dataframe_from_chunks(column_chunks, batch_size)


# Импорт времени выполнения для ChunkMetadata.processing_time
import time