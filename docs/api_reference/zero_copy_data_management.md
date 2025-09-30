# Zero-Copy Data Management Architecture для NormaML

## Обзор системы управления данными

Система управления данными NormaML построена на принципах zero-copy, обеспечивая минимальное копирование данных при передаче между компонентами и процессами. Основа системы - Polars для lazy evaluation и Apache Arrow для columnar memory format.

```
┌─────────────────────────────────────────────────────────────┐
│                    Data Management Layer                   │
├─────────────────┬───────────────┬─────────────────────────────┤
│   Data Sources  │  Memory Pool  │    Zero-Copy Engine        │
│   (Multiple)    │  Management   │    (Arrow + Polars)        │
└─────────────────┴───────────────┴─────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                 Memory Management Strategies               │
├─────────────────┬───────────────┬─────────────────────────────┤
│ Shared Memory   │ Memory Maps   │    Copy-on-Write           │
│   (POSIX SHM)   │   (mmap)      │    (COW Buffers)           │
├─────────────────┼───────────────┼─────────────────────────────┤
│   Arrow IPC     │ Lazy Frames   │    Streaming Buffers       │
│   (Zero-copy)   │  (Polars)     │    (Chunked Processing)    │
└─────────────────┴───────────────┴─────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                   Data Flow Engine                         │
├─────────────────┬───────────────┬─────────────────────────────┤
│  Data Pipeline  │  Cache Layer  │    Resource Monitor        │
│  Orchestration  │  Management   │    & GC Integration        │
└─────────────────┴───────────────┴─────────────────────────────┘
```

## Core Data Management Components

### 1. ZeroCopyDataManager

```python
import polars as pl
import pyarrow as pa
import numpy as np
from typing import Dict, Any, List, Optional, Union, Iterator
from abc import ABC, abstractmethod
from enum import Enum
import weakref
import threading
from contextlib import contextmanager
import mmap
import os

class DataFormat(Enum):
    ARROW_IPC = "arrow_ipc"
    ARROW_FEATHER = "arrow_feather"
    ARROW_PARQUET = "arrow_parquet"
    POLARS_LAZY = "polars_lazy"
    MEMORY_MAPPED = "memory_mapped"
    SHARED_MEMORY = "shared_memory"

class DataLocation(Enum):
    MEMORY = "memory"
    SHARED_MEMORY = "shared_memory"
    MEMORY_MAPPED = "memory_mapped"
    DISK = "disk"
    REMOTE = "remote"

@dataclass
class DataDescriptor:
    """Дескриптор данных для zero-copy операций"""
    id: str
    format: DataFormat
    location: DataLocation
    size_bytes: int
    shape: tuple
    schema: pa.Schema
    dtype_info: Dict[str, str]
    memory_address: Optional[int] = None
    file_path: Optional[str] = None
    shared_memory_name: Optional[str] = None
    reference_count: int = 0
    is_mutable: bool = False
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)

class ZeroCopyDataManager:
    """Центральный менеджер данных с zero-copy операциями"""
    
    def __init__(self, max_memory_mb: int = 8192, 
                 enable_shared_memory: bool = True,
                 enable_memory_mapping: bool = True):
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.enable_shared_memory = enable_shared_memory
        self.enable_memory_mapping = enable_memory_mapping
        
        # Реестр данных
        self.data_registry: Dict[str, DataDescriptor] = {}
        self.arrow_buffers: Dict[str, pa.Buffer] = {}
        self.shared_memory_blocks: Dict[str, 'SharedMemory'] = {}
        self.memory_maps: Dict[str, mmap.mmap] = {}
        
        # Memory pool для Arrow
        self.arrow_memory_pool = pa.default_memory_pool()
        
        # Tracking и статистика
        self.current_memory_usage = 0
        self.memory_usage_lock = threading.RLock()
        self.access_stats: Dict[str, int] = {}
        
        # Weak references для автоматической очистки
        self.weak_refs: Dict[str, weakref.ref] = {}
        
        # Cache для lazy frames
        self.lazy_frame_cache: Dict[str, pl.LazyFrame] = {}
        
        # Initialization
        self._setup_memory_pool()
        self._start_background_cleaner()
    
    def _setup_memory_pool(self):
        """Настройка Arrow memory pool"""
        # Используем system memory pool с tracking
        self.arrow_memory_pool = pa.system_memory_pool()
    
    def _start_background_cleaner(self):
        """Запуск фонового процесса для очистки неиспользуемых данных"""
        import atexit
        atexit.register(self.cleanup_all)
        
        def cleanup_worker():
            while True:
                try:
                    self._cleanup_unused_data()
                    time.sleep(30)  # Очистка каждые 30 секунд
                except Exception as e:
                    print(f"Background cleaner error: {e}")
        
        import threading
        cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        cleanup_thread.start()
    
    def register_data(self, data: Union[pl.DataFrame, pl.LazyFrame, pa.Table], 
                     data_id: str = None, 
                     strategy: str = "auto") -> str:
        """Регистрация данных в системе с выбором оптимальной стратегии"""
        if data_id is None:
            data_id = f"data_{len(self.data_registry)}"
        
        # Определяем оптимальную стратегию хранения
        if strategy == "auto":
            strategy = self._choose_storage_strategy(data)
        
        descriptor = self._create_data_descriptor(data, data_id, strategy)
        
        # Сохраняем данные согласно стратегии
        if strategy == "shared_memory":
            self._store_in_shared_memory(data, descriptor)
        elif strategy == "memory_mapped":
            self._store_as_memory_mapped(data, descriptor)
        elif strategy == "arrow_buffer":
            self._store_as_arrow_buffer(data, descriptor)
        elif strategy == "lazy_frame":
            self._store_as_lazy_frame(data, descriptor)
        else:
            raise ValueError(f"Unknown storage strategy: {strategy}")
        
        # Регистрируем в реестре
        self.data_registry[data_id] = descriptor
        
        # Устанавливаем weak reference для автоочистки
        self._setup_weak_reference(data_id, data)
        
        return data_id
    
    def _choose_storage_strategy(self, data: Union[pl.DataFrame, pl.LazyFrame, pa.Table]) -> str:
        """Автоматический выбор стратегии хранения"""
        if isinstance(data, pl.LazyFrame):
            return "lazy_frame"
        
        # Оцениваем размер данных
        if isinstance(data, pl.DataFrame):
            size_estimate = data.estimated_size()
        elif isinstance(data, pa.Table):
            size_estimate = data.nbytes
        else:
            size_estimate = 0
        
        # Стратегия на основе размера
        if size_estimate > 1024 * 1024 * 1024:  # > 1GB
            return "memory_mapped"
        elif size_estimate > 100 * 1024 * 1024:  # > 100MB
            return "shared_memory" if self.enable_shared_memory else "arrow_buffer"
        else:
            return "arrow_buffer"
    
    def _create_data_descriptor(self, data: Any, data_id: str, strategy: str) -> DataDescriptor:
        """Создание дескриптора данных"""
        if isinstance(data, pl.DataFrame):
            arrow_table = data.to_arrow()
            schema = arrow_table.schema
            shape = arrow_table.shape
            size_bytes = arrow_table.nbytes
        elif isinstance(data, pl.LazyFrame):
            # Для lazy frame собираем только schema
            schema = data.collect_schema().to_arrow()
            shape = (0, len(schema))  # Неизвестно до вычисления
            size_bytes = 0
        elif isinstance(data, pa.Table):
            schema = data.schema
            shape = data.shape
            size_bytes = data.nbytes
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")
        
        return DataDescriptor(
            id=data_id,
            format=DataFormat.ARROW_IPC if strategy != "lazy_frame" else DataFormat.POLARS_LAZY,
            location=self._strategy_to_location(strategy),
            size_bytes=size_bytes,
            shape=shape,
            schema=schema,
            dtype_info={field.name: str(field.type) for field in schema},
            is_mutable=False
        )
    
    def _strategy_to_location(self, strategy: str) -> DataLocation:
        """Преобразование стратегии в location"""
        mapping = {
            "shared_memory": DataLocation.SHARED_MEMORY,
            "memory_mapped": DataLocation.MEMORY_MAPPED,
            "arrow_buffer": DataLocation.MEMORY,
            "lazy_frame": DataLocation.MEMORY
        }
        return mapping.get(strategy, DataLocation.MEMORY)
    
    def _store_in_shared_memory(self, data: Any, descriptor: DataDescriptor):
        """Сохранение данных в shared memory"""
        if not self.enable_shared_memory:
            raise ValueError("Shared memory is disabled")
        
        # Конвертируем в Arrow table
        if isinstance(data, pl.DataFrame):
            arrow_table = data.to_arrow()
        elif isinstance(data, pa.Table):
            arrow_table = data
        else:
            raise ValueError("Cannot store lazy frame in shared memory")
        
        # Сериализация в Arrow IPC format
        sink = pa.BufferOutputStream()
        with pa.ipc.new_stream(sink, arrow_table.schema) as writer:
            writer.write_table(arrow_table)
        
        buffer_bytes = sink.getvalue().to_pybytes()
        
        # Создание shared memory блока
        from multiprocessing import shared_memory
        shm = shared_memory.SharedMemory(
            create=True,
            size=len(buffer_bytes),
            name=f"normaml_{descriptor.id}"
        )
        
        # Копирование данных
        shm.buf[:len(buffer_bytes)] = buffer_bytes
        
        # Обновление дескриптора
        descriptor.shared_memory_name = shm.name
        descriptor.memory_address = id(shm.buf)
        
        # Сохранение ссылки на shared memory
        self.shared_memory_blocks[descriptor.id] = shm
        
        # Обновление счетчика памяти
        with self.memory_usage_lock:
            self.current_memory_usage += len(buffer_bytes)
    
    def _store_as_memory_mapped(self, data: Any, descriptor: DataDescriptor):
        """Сохранение данных как memory-mapped file"""
        if not self.enable_memory_mapping:
            raise ValueError("Memory mapping is disabled")
        
        # Конвертируем в Arrow table
        if isinstance(data, pl.DataFrame):
            arrow_table = data.to_arrow()
        elif isinstance(data, pa.Table):
            arrow_table = data
        else:
            raise ValueError("Cannot store lazy frame as memory-mapped")
        
        # Создаем временный файл
        import tempfile
        fd, temp_path = tempfile.mkstemp(suffix=".arrow", prefix="normaml_")
        
        try:
            # Записываем в Arrow IPC format
            with pa.OSFile(temp_path, 'wb') as sink:
                with pa.ipc.new_file(sink, arrow_table.schema) as writer:
                    writer.write_table(arrow_table)
            
            # Создаем memory map
            with open(temp_path, 'rb') as f:
                mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            
            # Обновление дескриптора
            descriptor.file_path = temp_path
            descriptor.memory_address = id(mm)
            
            # Сохранение ссылки на memory map
            self.memory_maps[descriptor.id] = mm
            
        finally:
            os.close(fd)
    
    def _store_as_arrow_buffer(self, data: Any, descriptor: DataDescriptor):
        """Сохранение данных как Arrow buffer в памяти"""
        # Конвертируем в Arrow table
        if isinstance(data, pl.DataFrame):
            arrow_table = data.to_arrow()
        elif isinstance(data, pa.Table):
            arrow_table = data
        else:
            raise ValueError("Cannot store lazy frame as arrow buffer")
        
        # Сериализация в Arrow IPC
        sink = pa.BufferOutputStream()
        with pa.ipc.new_stream(sink, arrow_table.schema) as writer:
            writer.write_table(arrow_table)
        
        buffer = sink.getvalue()
        
        # Сохранение buffer
        self.arrow_buffers[descriptor.id] = buffer
        descriptor.memory_address = buffer.address
        
        # Обновление счетчика памяти
        with self.memory_usage_lock:
            self.current_memory_usage += buffer.size
    
    def _store_as_lazy_frame(self, data: pl.LazyFrame, descriptor: DataDescriptor):
        """Сохранение lazy frame (без материализации)"""
        if not isinstance(data, pl.LazyFrame):
            if isinstance(data, pl.DataFrame):
                data = data.lazy()
            else:
                raise ValueError("Can only store LazyFrame with lazy strategy")
        
        # Сохраняем lazy frame в cache
        self.lazy_frame_cache[descriptor.id] = data
    
    def get_data(self, data_id: str, format: str = "polars") -> Union[pl.DataFrame, pl.LazyFrame, pa.Table]:
        """Получение данных по ID с zero-copy when possible"""
        if data_id not in self.data_registry:
            raise KeyError(f"Data {data_id} not found")
        
        descriptor = self.data_registry[data_id]
        
        # Обновляем статистику доступа
        self.access_stats[data_id] = self.access_stats.get(data_id, 0) + 1
        descriptor.last_accessed = time.time()
        
        # Получаем данные согласно текущему location
        if descriptor.location == DataLocation.SHARED_MEMORY:
            return self._load_from_shared_memory(descriptor, format)
        elif descriptor.location == DataLocation.MEMORY_MAPPED:
            return self._load_from_memory_mapped(descriptor, format)
        elif descriptor.location == DataLocation.MEMORY:
            return self._load_from_arrow_buffer(descriptor, format)
        else:  # LAZY_FRAME
            return self._load_lazy_frame(descriptor, format)
    
    def _load_from_shared_memory(self, descriptor: DataDescriptor, format: str):
        """Загрузка из shared memory (zero-copy)"""
        if descriptor.id not in self.shared_memory_blocks:
            raise RuntimeError(f"Shared memory block for {descriptor.id} not found")
        
        shm = self.shared_memory_blocks[descriptor.id]
        
        # Создаем Arrow buffer из shared memory (zero-copy)
        buffer = pa.py_buffer(shm.buf[:descriptor.size_bytes])
        
        # Читаем Arrow table
        reader = pa.ipc.open_stream(buffer)
        arrow_table = reader.read_all()
        
        return self._convert_to_format(arrow_table, format)
    
    def _load_from_memory_mapped(self, descriptor: DataDescriptor, format: str):
        """Загрузка из memory-mapped file (zero-copy)"""
        if descriptor.id not in self.memory_maps:
            raise RuntimeError(f"Memory map for {descriptor.id} not found")
        
        mm = self.memory_maps[descriptor.id]
        
        # Создаем Arrow buffer из memory map (zero-copy)
        buffer = pa.py_buffer(mm)
        
        # Читаем Arrow table
        reader = pa.ipc.open_file(buffer)
        arrow_table = reader.read_all()
        
        return self._convert_to_format(arrow_table, format)
    
    def _load_from_arrow_buffer(self, descriptor: DataDescriptor, format: str):
        """Загрузка из Arrow buffer (zero-copy)"""
        if descriptor.id not in self.arrow_buffers:
            raise RuntimeError(f"Arrow buffer for {descriptor.id} not found")
        
        buffer = self.arrow_buffers[descriptor.id]
        
        # Читаем Arrow table
        reader = pa.ipc.open_stream(buffer)
        arrow_table = reader.read_all()
        
        return self._convert_to_format(arrow_table, format)
    
    def _load_lazy_frame(self, descriptor: DataDescriptor, format: str):
        """Загрузка lazy frame"""
        if descriptor.id not in self.lazy_frame_cache:
            raise RuntimeError(f"Lazy frame for {descriptor.id} not found")
        
        lazy_frame = self.lazy_frame_cache[descriptor.id]
        
        if format == "polars_lazy":
            return lazy_frame
        elif format == "polars":
            return lazy_frame.collect()
        elif format == "arrow":
            return lazy_frame.collect().to_arrow()
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _convert_to_format(self, arrow_table: pa.Table, format: str):
        """Конвертация Arrow table в запрошенный формат"""
        if format == "arrow":
            return arrow_table
        elif format == "polars":
            return pl.from_arrow(arrow_table)
        elif format == "polars_lazy":
            return pl.from_arrow(arrow_table).lazy()
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def create_view(self, data_id: str, view_name: str, 
                   selection: Dict[str, Any] = None) -> str:
        """Создание представления данных (zero-copy)"""
        if data_id not in self.data_registry:
            raise KeyError(f"Data {data_id} not found")
        
        base_descriptor = self.data_registry[data_id]
        
        # Для lazy frames создаем новый lazy frame с операциями
        if base_descriptor.location == DataLocation.MEMORY:
            lazy_frame = self.get_data(data_id, "polars_lazy")
            
            # Применяем селекцию
            if selection:
                if 'columns' in selection:
                    lazy_frame = lazy_frame.select(selection['columns'])
                if 'filter' in selection:
                    lazy_frame = lazy_frame.filter(selection['filter'])
                if 'limit' in selection:
                    lazy_frame = lazy_frame.limit(selection['limit'])
            
            # Регистрируем view как lazy frame
            return self.register_data(lazy_frame, view_name, "lazy_frame")
        
        else:
            # Для других типов создаем новый дескриптор с ссылкой на исходные данные
            view_descriptor = DataDescriptor(
                id=view_name,
                format=base_descriptor.format,
                location=base_descriptor.location,
                size_bytes=base_descriptor.size_bytes,  # Может быть меньше
                shape=base_descriptor.shape,  # Может измениться
                schema=base_descriptor.schema,  # Может измениться
                dtype_info=base_descriptor.dtype_info,
                memory_address=base_descriptor.memory_address,
                file_path=base_descriptor.file_path,
                shared_memory_name=base_descriptor.shared_memory_name,
                is_mutable=False
            )
            
            self.data_registry[view_name] = view_descriptor
            
            # Увеличиваем reference count для исходных данных
            base_descriptor.reference_count += 1
            
            return view_name
    
    @contextmanager
    def get_zero_copy_buffer(self, data_id: str):
        """Context manager для получения zero-copy buffer"""
        descriptor = self.data_registry[data_id]
        
        try:
            if descriptor.location == DataLocation.SHARED_MEMORY:
                shm = self.shared_memory_blocks[descriptor.id]
                yield memoryview(shm.buf[:descriptor.size_bytes])
            elif descriptor.location == DataLocation.MEMORY_MAPPED:
                mm = self.memory_maps[descriptor.id]
                yield memoryview(mm)
            elif descriptor.location == DataLocation.MEMORY:
                buffer = self.arrow_buffers[descriptor.id]
                yield memoryview(buffer.to_pybytes())
            else:
                raise ValueError("Zero-copy buffer not available for lazy frames")
        finally:
            # Обновляем статистику доступа
            descriptor.last_accessed = time.time()
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Получение статистики использования памяти"""
        total_arrow = sum(buf.size for buf in self.arrow_buffers.values())
        total_shared = sum(
            desc.size_bytes for desc in self.data_registry.values()
            if desc.location == DataLocation.SHARED_MEMORY
        )
        total_mapped = len(self.memory_maps)
        
        return {
            'total_memory_usage': self.current_memory_usage,
            'max_memory_limit': self.max_memory_bytes,
            'memory_utilization': self.current_memory_usage / self.max_memory_bytes,
            'arrow_buffers_size': total_arrow,
            'shared_memory_size': total_shared,
            'memory_mapped_files': total_mapped,
            'lazy_frames_count': len(self.lazy_frame_cache),
            'registered_datasets': len(self.data_registry),
            'arrow_pool_bytes_allocated': self.arrow_memory_pool.bytes_allocated(),
            'access_stats': self.access_stats.copy()
        }
    
    def _setup_weak_reference(self, data_id: str, data: Any):
        """Настройка weak reference для автоматической очистки"""
        def cleanup_callback(ref):
            self._cleanup_data(data_id)
        
        self.weak_refs[data_id] = weakref.ref(data, cleanup_callback)
    
    def _cleanup_data(self, data_id: str):
        """Очистка данных"""
        if data_id not in self.data_registry:
            return
        
        descriptor = self.data_registry[data_id]
        
        # Проверяем reference count
        if descriptor.reference_count > 0:
            return
        
        try:
            # Очистка по типу хранения
            if descriptor.location == DataLocation.SHARED_MEMORY:
                if data_id in self.shared_memory_blocks:
                    shm = self.shared_memory_blocks[data_id]
                    shm.close()
                    shm.unlink()
                    del self.shared_memory_blocks[data_id]
            
            elif descriptor.location == DataLocation.MEMORY_MAPPED:
                if data_id in self.memory_maps:
                    mm = self.memory_maps[data_id]
                    mm.close()
                    del self.memory_maps[data_id]
                
                if descriptor.file_path and os.path.exists(descriptor.file_path):
                    os.unlink(descriptor.file_path)
            
            elif descriptor.location == DataLocation.MEMORY:
                if data_id in self.arrow_buffers:
                    buffer = self.arrow_buffers[data_id]
                    del self.arrow_buffers[data_id]
                    
                    with self.memory_usage_lock:
                        self.current_memory_usage -= buffer.size
            
            # Очистка lazy frame cache
            if data_id in self.lazy_frame_cache:
                del self.lazy_frame_cache[data_id]
            
            # Удаление из реестра
            del self.data_registry[data_id]
            
            # Очистка статистики
            if data_id in self.access_stats:
                del self.access_stats[data_id]
            
            if data_id in self.weak_refs:
                del self.weak_refs[data_id]
                
        except Exception as e:
            print(f"Error cleaning up data {data_id}: {e}")
    
    def _cleanup_unused_data(self):
        """Очистка неиспользуемых данных"""
        current_time = time.time()
        timeout = 3600  # 1 час
        
        to_cleanup = []
        for data_id, descriptor in self.data_registry.items():
            if (current_time - descriptor.last_accessed > timeout and 
                descriptor.reference_count == 0):
                to_cleanup.append(data_id)
        
        for data_id in to_cleanup:
            self._cleanup_data(data_id)
    
    def cleanup_all(self):
        """Полная очистка всех данных"""
        for data_id in list(self.data_registry.keys()):
            self._cleanup_data(data_id)
```

## Streaming Data Processing

```python
from typing import Iterator, Callable
import polars as pl

class StreamingDataProcessor:
    """Обработка данных потоками для больших датасетов"""
    
    def __init__(self, data_manager: ZeroCopyDataManager, 
                 chunk_size: int = 100000):
        self.data_manager = data_manager
        self.chunk_size = chunk_size
    
    def stream_chunks(self, source: str, **kwargs) -> Iterator[pl.DataFrame]:
        """Потоковое чтение данных по частям"""
        if source.endswith('.csv'):
            yield from self._stream_csv(source, **kwargs)
        elif source.endswith('.parquet'):
            yield from self._stream_parquet(source, **kwargs)
        else:
            raise ValueError(f"Unsupported source format: {source}")
    
    def _stream_csv(self, source: str, **kwargs) -> Iterator[pl.DataFrame]:
        """Потоковое чтение CSV"""
        lazy_frame = pl.scan_csv(source, **kwargs)
        
        # Читаем по частям
        offset = 0
        while True:
            chunk = lazy_frame.slice(offset, self.chunk_size).collect()
            if chunk.height == 0:
                break
            
            yield chunk
            offset += self.chunk_size
    
    def _stream_parquet(self, source: str, **kwargs) -> Iterator[pl.DataFrame]:
        """Потоковое чтение Parquet"""
        lazy_frame = pl.scan_parquet(source, **kwargs)
        
        # Читаем по частям
        offset = 0
        while True:
            chunk = lazy_frame.slice(offset, self.chunk_size).collect()
            if chunk.height == 0:
                break
            
            yield chunk
            offset += self.chunk_size
    
    def process_streaming(self, source: str, 
                         processor: Callable[[pl.DataFrame], pl.DataFrame],
                         output_path: str = None) -> str:
        """Обработка данных потоком с применением функции"""
        results = []
        
        for chunk in self.stream_chunks(source):
            processed_chunk = processor(chunk)
            results.append(processed_chunk)
        
        # Объединяем результаты
        final_result = pl.concat(results)
        
        if output_path:
            final_result.write_parquet(output_path)
            return self.data_manager.register_data(
                pl.scan_parquet(output_path), 
                strategy="memory_mapped"
            )
        else:
            return self.data_manager.register_data(final_result)
```

## Copy-on-Write Implementation

```python
class COWDataFrame:
    """Copy-on-Write wrapper для DataFrame"""
    
    def __init__(self, data_manager: ZeroCopyDataManager, data_id: str):
        self.data_manager = data_manager
        self.data_id = data_id
        self._is_modified = False
        self._modifications = []
    
    def __getattr__(self, name):
        """Проксирование атрибутов к исходному DataFrame"""
        data = self.data_manager.get_data(self.data_id, "polars")
        return getattr(data, name)
    
    def select(self, *args, **kwargs):
        """Copy-on-write select"""
        if not self._is_modified:
            # Первая модификация - создаем view
            view_id = self.data_manager.create_view(
                self.data_id, 
                f"{self.data_id}_view_{len(self._modifications)}"
            )
            return COWDataFrame(self.data_manager, view_id)
        else:
            # Уже модифицирован - применяем операции
            data = self.data_manager.get_data(self.data_id, "polars")
            result = data.select(*args, **kwargs)
            new_id = self.data_manager.register_data(result)
            return COWDataFrame(self.data_manager, new_id)
    
    def filter(self, *args, **kwargs):
        """Copy-on-write filter"""
        if not self._is_modified:
            lazy_data = self.data_manager.get_data(self.data_id, "polars_lazy")
            filtered = lazy_data.filter(*args, **kwargs)
            new_id = self.data_manager.register_data(filtered, strategy="lazy_frame")
            return COWDataFrame(self.data_manager, new_id)
        else:
            data = self.data_manager.get_data(self.data_id, "polars")
            result = data.filter(*args, **kwargs)
            new_id = self.data_manager.register_data(result)
            return COWDataFrame(self.data_manager, new_id)
    
    def collect(self):
        """Материализация lazy operations"""
        return self.data_manager.get_data(self.data_id, "polars")
```

## Performance Optimizations

```python
class DataOptimizer:
    """Оптимизации для работы с данными"""
    
    def __init__(self, data_manager: ZeroCopyDataManager):
        self.data_manager = data_manager
    
    def optimize_data_layout(self, data_id: str) -> str:
        """Оптимизация layout данных для лучшей производительности"""
        data = self.data_manager.get_data(data_id, "polars")
        
        # Анализируем типы данных
        schema = data.schema
        
        # Оптимизируем типы данных
        optimized_data = data
        for column, dtype in schema.items():
            if dtype == pl.Int64:
                # Проверяем, можно ли использовать меньший тип
                min_val = data.select(pl.col(column).min()).item()
                max_val = data.select(pl.col(column).max()).item()
                
                if min_val >= 0 and max_val <= 255:
                    optimized_data = optimized_data.with_columns(
                        pl.col(column).cast(pl.UInt8)
                    )
                elif min_val >= -128 and max_val <= 127:
                    optimized_data = optimized_data.with_columns(
                        pl.col(column).cast(pl.Int8)
                    )
                elif min_val >= -32768 and max_val <= 32767:
                    optimized_data = optimized_data.with_columns(
                        pl.col(column).cast(pl.Int16)
                    )
        
        # Регистрируем оптимизированную версию
        return self.data_manager.register_data(
            optimized_data, 
            f"{data_id}_optimized"
        )
    
    def create_column_index(self, data_id: str, column: str) -> Dict[str, Any]:
        """Создание индекса по колонке для быстрого поиска"""
        data = self.data_manager.get_data(data_id, "polars")
        
        # Создаем индекс как dictionary
        indexed_data = data.select([column, pl.arange(0, data.height).alias("_row_id")])
        index_dict = {
            row[column]: row["_row_id"] 
            for row in indexed_data.to_dicts()
        }
        
        return {
            'type': 'hash_index',
            'column': column,
            'data_id': data_id,
            'index': index_dict,
            'size': len(index_dict)
        }
```

## Integration Examples

```python
# Примеры использования zero-copy data management

# Инициализация
data_manager = ZeroCopyDataManager(
    max_memory_mb=8192,
    enable_shared_memory=True,
    enable_memory_mapping=True
)

# Загрузка больших данных
large_data_id = data_manager.register_data(
    pl.scan_parquet("large_dataset.parquet"),
    strategy="memory_mapped"
)

# Создание view без копирования
subset_id = data_manager.create_view(
    large_data_id,
    "subset_view",
    selection={
        'columns': ['feature1', 'feature2', 'target'],
        'filter': pl.col('target').is_not_null()
    }
)

# Zero-copy передача в plugin
with data_manager.get_zero_copy_buffer(subset_id) as buffer:
    # Передача buffer в plugin через shared memory
    plugin_result = some_plugin.process_buffer(buffer)

# Streaming обработка
streaming_processor = StreamingDataProcessor(data_manager)
processed_id = streaming_processor.process_streaming(
    "huge_dataset.csv",
    lambda chunk: chunk.select(['important_features']).drop_nulls()
)

# Copy-on-write operations
cow_data = COWDataFrame(data_manager, large_data_id)
filtered_data = cow_data.filter(pl.col('value') > 100)  # No copy yet
result = filtered_data.collect()  # Copy only when needed

# Memory monitoring
stats = data_manager.get_memory_stats()
print(f"Memory utilization: {stats['memory_utilization']:.2%}")
print(f"Active datasets: {stats['registered_datasets']}")
```

Эта система обеспечивает:

1. **True Zero-Copy** - минимальное копирование данных между компонентами
2. **Memory Efficiency** - оптимальное использование памяти с автоматической очисткой
3. **Scalability** - поддержка больших датасетов через streaming и memory mapping
4. **Flexibility** - multiple storage strategies для разных сценариев
5. **Performance** - lazy evaluation и columnar formats для максимальной производительности
6. **Resource Management** - автоматический garbage collection и monitoring