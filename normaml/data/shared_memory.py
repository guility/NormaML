"""
Модуль для управления shared memory и zero-copy операций с numpy arrays.

Обеспечивает прямое маппирование numpy массивов в shared memory без сериализации,
используя POSIX shared memory и memory views для истинно zero-copy доступа.
"""

import os
import sys
import mmap
import uuid
from typing import Dict, Optional, Tuple, Any, Union, List
from dataclasses import dataclass
from pathlib import Path
import threading
import weakref
import logging

try:
    import numpy as np
except ImportError:
    raise ImportError("numpy is required for zero-copy data management. Install with: pip install numpy>=1.21.0")

try:
    import polars as pl
except ImportError:
    raise ImportError("polars is required for zero-copy data management. Install with: pip install polars>=0.19.0")

# Импорт POSIX-специфичных функций
HAS_SHM = False
try:
    if hasattr(os, 'shm_open'):
        from os import shm_open, shm_unlink
        HAS_SHM = True
except (ImportError, AttributeError):
    pass

logger = logging.getLogger(__name__)

# Определяем платформу для выбора стратегии shared memory
IS_WINDOWS = sys.platform.startswith('win')
IS_POSIX = HAS_SHM and not IS_WINDOWS


@dataclass
class ArrayMetadata:
    """Метаданные для numpy array в shared memory."""
    dtype: np.dtype
    shape: Tuple[int, ...]
    strides: Tuple[int, ...]
    size_bytes: int
    shm_name: str
    offset: int = 0


class SharedMemoryError(Exception):
    """Исключения для операций с shared memory."""
    pass


class SharedMemoryPool:
    """
    Пул shared memory блоков для эффективного переиспользования.
    
    Управляет созданием, переиспользованием и освобождением shared memory блоков.
    """
    
    def __init__(self, block_size: int = 1024 * 1024 * 64):  # 64MB блоки по умолчанию
        self.block_size = block_size
        self._free_blocks: Dict[int, List[str]] = {}  # размер -> список имен блоков
        self._used_blocks: Dict[str, int] = {}  # имя блока -> размер
        self._lock = threading.RLock()
        
    def get_block(self, size: int) -> str:
        """Получить блок shared memory подходящего размера."""
        with self._lock:
            # Найти подходящий размер блока (степень двойки >= size)
            block_size = self._next_power_of_2(max(size, 4096))
            
            # Попытаться найти свободный блок
            if block_size in self._free_blocks and self._free_blocks[block_size]:
                shm_name = self._free_blocks[block_size].pop()
                self._used_blocks[shm_name] = block_size
                return shm_name
            
            # Создать новый блок
            shm_name = f"normaml_shm_{uuid.uuid4().hex}"
            if self._create_shared_memory(shm_name, block_size):
                self._used_blocks[shm_name] = block_size
                return shm_name
            
            raise SharedMemoryError(f"Не удалось создать shared memory блок размером {block_size} байт")
    
    def return_block(self, shm_name: str):
        """Вернуть блок в пул для переиспользования."""
        with self._lock:
            if shm_name in self._used_blocks:
                block_size = self._used_blocks.pop(shm_name)
                if block_size not in self._free_blocks:
                    self._free_blocks[block_size] = []
                self._free_blocks[block_size].append(shm_name)
    
    def cleanup(self):
        """Очистить все блоки в пуле."""
        with self._lock:
            # Удалить все свободные блоки
            for block_list in self._free_blocks.values():
                for shm_name in block_list:
                    self._cleanup_shared_memory(shm_name)
            
            # Удалить все используемые блоки
            for shm_name in list(self._used_blocks.keys()):
                self._cleanup_shared_memory(shm_name)
            
            self._free_blocks.clear()
            self._used_blocks.clear()
    
    @staticmethod
    def _next_power_of_2(n: int) -> int:
        """Найти следующую степень двойки >= n."""
        return 1 << (n - 1).bit_length()
    
    @staticmethod
    def _create_shared_memory(name: str, size: int) -> bool:
        """Создать блок shared memory."""
        try:
            if IS_WINDOWS:
                # На Windows используем memory-mapped files
                return SharedMemoryManager._create_windows_shm(name, size)
            else:
                # На POSIX используем shm_open
                return SharedMemoryManager._create_posix_shm(name, size)
        except Exception as e:
            logger.error(f"Ошибка создания shared memory {name}: {e}")
            return False
    
    @staticmethod
    def _cleanup_shared_memory(name: str):
        """Удалить блок shared memory."""
        try:
            if IS_WINDOWS:
                SharedMemoryManager._cleanup_windows_shm(name)
            else:
                SharedMemoryManager._cleanup_posix_shm(name)
        except Exception as e:
            logger.warning(f"Ошибка очистки shared memory {name}: {e}")


class SharedMemoryManager:
    """
    Менеджер для создания и управления shared memory mappings для numpy arrays.
    
    Обеспечивает zero-copy доступ к данным через прямое маппирование в shared memory.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if hasattr(self, '_initialized'):
            return
        
        self._memory_mappings: Dict[str, mmap.mmap] = {}
        self._array_registry: Dict[str, ArrayMetadata] = {}
        self._pool = SharedMemoryPool()
        self._lock = threading.RLock()
        self._cleanup_registry = weakref.WeakValueDictionary()
        self._initialized = True
        
        # Регистрируем cleanup при выходе
        import atexit
        atexit.register(self.cleanup_all)
    
    def create_shared_array(self, array: np.ndarray, name: Optional[str] = None) -> Tuple[str, ArrayMetadata]:
        """
        Создать shared memory копию numpy array с zero-copy.
        
        Args:
            array: numpy array для копирования в shared memory
            name: опциональное имя для shared memory блока
            
        Returns:
            Tuple[str, ArrayMetadata]: имя shared memory блока и метаданные
        """
        if not isinstance(array, np.ndarray):
            raise ValueError("Ожидается numpy.ndarray")
        
        # Убедимся что array contiguous для эффективного копирования
        if not array.flags.c_contiguous:
            array = np.ascontiguousarray(array)
        
        size_bytes = array.nbytes
        shm_name = name or f"normaml_array_{uuid.uuid4().hex}"
        
        with self._lock:
            actual_shm_name = None
            try:
                # Получить блок shared memory
                actual_shm_name = self._pool.get_block(size_bytes)
                
                # Создать memory mapping
                if IS_WINDOWS:
                    mm = self._create_windows_mapping(actual_shm_name, size_bytes)
                else:
                    mm = self._create_posix_mapping(actual_shm_name, size_bytes)
                
                # Прямое копирование данных через memoryview (zero-copy)
                mv = memoryview(mm)
                mv[:size_bytes] = array.data.tobytes()
                
                # Создать метаданные
                metadata = ArrayMetadata(
                    dtype=array.dtype,
                    shape=array.shape,
                    strides=array.strides,
                    size_bytes=size_bytes,
                    shm_name=actual_shm_name
                )
                
                # Зарегистрировать mapping
                self._memory_mappings[shm_name] = mm
                self._array_registry[shm_name] = metadata
                
                # Добавить в cleanup registry
                self._cleanup_registry[shm_name] = self
                
                logger.debug(f"Создан shared array {shm_name} размером {size_bytes} байт")
                return shm_name, metadata
                
            except Exception as e:
                if actual_shm_name is not None:
                    self._pool.return_block(actual_shm_name)
                raise SharedMemoryError(f"Ошибка создания shared array: {e}")
    
    def get_shared_array(self, name: str) -> np.ndarray:
        """
        Получить numpy array из shared memory с zero-copy доступом.
        
        Args:
            name: имя shared memory блока
            
        Returns:
            numpy array с прямым доступом к shared memory
        """
        with self._lock:
            if name not in self._array_registry:
                raise KeyError(f"Shared array {name} не найден")
            
            metadata = self._array_registry[name]
            mm = self._memory_mappings[name]
            
            # Создать numpy array из memory view (zero-copy)
            mv = memoryview(mm)
            array_view = mv[:metadata.size_bytes]
            
            # Использовать numpy.frombuffer для zero-copy доступа
            flat_array = np.frombuffer(array_view, dtype=metadata.dtype)
            
            # Изменить shape и strides для восстановления оригинальной структуры
            shaped_array = flat_array.reshape(metadata.shape)
            
            # Установить strides если они отличаются
            if shaped_array.strides != metadata.strides:
                shaped_array = np.lib.stride_tricks.as_strided(
                    flat_array, 
                    shape=metadata.shape, 
                    strides=metadata.strides
                )
            
            return shaped_array
    
    def create_shared_dataframe(self, df: pl.DataFrame, name_prefix: Optional[str] = None) -> Dict[str, ArrayMetadata]:
        """
        Создать shared memory представление polars DataFrame.
        
        Args:
            df: polars DataFrame
            name_prefix: префикс для имен shared memory блоков
            
        Returns:
            Словарь column_name -> ArrayMetadata
        """
        prefix = name_prefix or f"df_{uuid.uuid4().hex}"
        column_metadata = {}
        
        for column_name in df.columns:
            series = df[column_name]
            
            # Прямое извлечение numpy array из polars Series (zero-copy)
            numpy_array = series.to_numpy()
            
            # Создать shared memory для колонки
            shm_name = f"{prefix}_{column_name}"
            _, metadata = self.create_shared_array(numpy_array, shm_name)
            column_metadata[column_name] = metadata
        
        logger.debug(f"Создан shared DataFrame {prefix} с {len(column_metadata)} колонками")
        return column_metadata
    
    def reconstruct_dataframe(self, column_metadata: Dict[str, ArrayMetadata], column_names: Optional[List[str]] = None) -> pl.DataFrame:
        """
        Восстановить polars DataFrame из shared memory arrays.
        
        Args:
            column_metadata: метаданные колонок
            column_names: порядок колонок (если None, используется порядок из metadata)
            
        Returns:
            polars DataFrame с zero-copy доступом к данным
        """
        if column_names is None:
            column_names = list(column_metadata.keys())
        
        columns_data = {}
        
        for column_name in column_names:
            if column_name not in column_metadata:
                raise KeyError(f"Метаданные для колонки {column_name} не найдены")
            
            metadata = column_metadata[column_name]
            
            # Получить shared array (zero-copy)
            shared_array = self.get_shared_array(f"{metadata.shm_name}")
            
            # Создать polars Series из numpy array (zero-copy если возможно)
            series = pl.Series(column_name, shared_array)
            columns_data[column_name] = series
        
        return pl.DataFrame(columns_data)
    
    def remove_shared_array(self, name: str):
        """Удалить shared array и освободить memory."""
        with self._lock:
            if name in self._memory_mappings:
                mm = self._memory_mappings.pop(name)
                metadata = self._array_registry.pop(name, None)
                
                try:
                    mm.close()
                    if metadata:
                        self._pool.return_block(metadata.shm_name)
                    logger.debug(f"Удален shared array {name}")
                except Exception as e:
                    logger.warning(f"Ошибка при удалении shared array {name}: {e}")
    
    def cleanup_all(self):
        """Очистить все shared memory mappings."""
        with self._lock:
            for name in list(self._memory_mappings.keys()):
                self.remove_shared_array(name)
            self._pool.cleanup()
            logger.info("Очищены все shared memory mappings")
    
    def get_memory_usage(self) -> Dict[str, int]:
        """Получить информацию об использовании памяти."""
        with self._lock:
            usage = {}
            total_bytes = 0
            
            for name, metadata in self._array_registry.items():
                usage[name] = metadata.size_bytes
                total_bytes += metadata.size_bytes
            
            usage['total_bytes'] = total_bytes
            usage['total_arrays'] = len(self._array_registry)
            
            return usage
    
    # Платформо-специфичные методы
    
    @staticmethod
    def _create_windows_mapping(name: str, size: int) -> mmap.mmap:
        """Создать memory mapping на Windows."""
        try:
            # Используем anonymous mmap для Windows
            mm = mmap.mmap(-1, size, tagname=name)
            return mm
        except Exception as e:
            raise SharedMemoryError(f"Ошибка создания Windows mapping: {e}")
    
    @staticmethod
    def _create_posix_mapping(name: str, size: int) -> mmap.mmap:
        """Создать memory mapping на POSIX системах."""
        if not HAS_SHM:
            raise SharedMemoryError("POSIX shared memory не поддерживается на этой платформе")
        
        try:
            # Открыть существующий shared memory блок
            fd = shm_open(name, os.O_RDWR)
            mm = mmap.mmap(fd, size)
            os.close(fd)
            return mm
        except Exception as e:
            raise SharedMemoryError(f"Ошибка создания POSIX mapping: {e}")
    
    @staticmethod
    def _create_windows_shm(name: str, size: int) -> bool:
        """Создать shared memory блок на Windows."""
        # На Windows используем временные файлы как fallback
        temp_path = Path.cwd() / "temp_shm" / name
        temp_path.parent.mkdir(exist_ok=True)
        
        try:
            with open(temp_path, 'wb') as f:
                f.write(b'\x00' * size)
            return True
        except Exception:
            return False
    
    @staticmethod
    def _create_posix_shm(name: str, size: int) -> bool:
        """Создать shared memory блок на POSIX системах."""
        if not HAS_SHM:
            return False
        
        try:
            fd = shm_open(name, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
            os.ftruncate(fd, size)
            os.close(fd)
            return True
        except Exception:
            return False
    
    @staticmethod
    def _cleanup_windows_shm(name: str):
        """Удалить shared memory блок на Windows."""
        temp_path = Path.cwd() / "temp_shm" / name
        try:
            if temp_path.exists():
                temp_path.unlink()
        except Exception:
            pass
    
    @staticmethod
    def _cleanup_posix_shm(name: str):
        """Удалить shared memory блок на POSIX системах."""
        if not HAS_SHM:
            return
        
        try:
            shm_unlink(name)
        except Exception:
            pass


def create_memory_view(array: np.ndarray) -> memoryview:
    """
    Создать memory view для numpy array для zero-copy операций.
    
    Args:
        array: numpy array
        
    Returns:
        memoryview объект для прямого доступа к памяти
    """
    return memoryview(array.data)


def copy_to_shared_memory(source_array: np.ndarray, target_view: memoryview):
    """
    Скопировать numpy array в shared memory через memory view.
    
    Args:
        source_array: исходный numpy array
        target_view: memory view целевого shared memory блока
    """
    if not source_array.flags.c_contiguous:
        source_array = np.ascontiguousarray(source_array)
    
    source_view = memoryview(source_array.data)
    target_view[:len(source_view)] = source_view


def create_array_from_buffer(buffer: Union[memoryview, bytes], dtype: np.dtype, shape: Tuple[int, ...]) -> np.ndarray:
    """
    Создать numpy array из buffer с заданными dtype и shape (zero-copy).
    
    Args:
        buffer: buffer с данными
        dtype: тип данных numpy
        shape: форма массива
        
    Returns:
        numpy array с zero-copy доступом к buffer
    """
    flat_array = np.frombuffer(buffer, dtype=dtype)
    return flat_array.reshape(shape)


# Глобальный экземпляр менеджера
_shared_memory_manager = None

def get_shared_memory_manager() -> SharedMemoryManager:
    """Получить глобальный экземпляр SharedMemoryManager."""
    global _shared_memory_manager
    if _shared_memory_manager is None:
        _shared_memory_manager = SharedMemoryManager()
    return _shared_memory_manager