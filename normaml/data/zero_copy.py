"""
Система управления данными с zero-copy поддержкой БЕЗ сериализации.

Обеспечивает прямое маппирование polars.Series через numpy arrays в shared memory,
registry данных с метаданными без сериализации и thread-safe операции.
"""

import uuid
import time
import threading
import weakref
from typing import Dict, List, Optional, Union, Any, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import logging
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

try:
    import numpy as np
except ImportError:
    raise ImportError("numpy is required for zero-copy data management")

try:
    import polars as pl
except ImportError:
    raise ImportError("polars is required for zero-copy data management")

from .shared_memory import (
    SharedMemoryManager, 
    ArrayMetadata, 
    get_shared_memory_manager,
    SharedMemoryError
)

logger = logging.getLogger(__name__)


class DataStatus(Enum):
    """Статус данных в registry."""
    PENDING = "pending"
    READY = "ready"
    PROCESSING = "processing"
    ERROR = "error"
    EXPIRED = "expired"


@dataclass
class DataDescriptor:
    """
    Дескриптор данных с numpy array metadata без сериализации.
    
    Содержит всю необходимую информацию для восстановления данных
    из shared memory без промежуточной сериализации.
    """
    id: str
    name: str
    status: DataStatus
    dtype: np.dtype
    shape: Tuple[int, ...]
    strides: Tuple[int, ...]
    size_bytes: int
    shm_name: str
    created_at: float
    accessed_at: float
    access_count: int = 0
    is_view: bool = False
    parent_id: Optional[str] = None
    column_names: List[str] = field(default_factory=list)
    column_metadata: Dict[str, ArrayMetadata] = field(default_factory=dict)
    tags: Set[str] = field(default_factory=set)
    
    def mark_accessed(self):
        """Отметить доступ к данным."""
        self.accessed_at = time.time()
        self.access_count += 1
    
    def is_expired(self, ttl_seconds: float) -> bool:
        """Проверить истек ли TTL данных."""
        return time.time() - self.accessed_at > ttl_seconds
    
    def get_memory_info(self) -> Dict[str, Any]:
        """Получить информацию о использовании памяти."""
        return {
            'size_bytes': self.size_bytes,
            'dtype': str(self.dtype),
            'shape': self.shape,
            'is_view': self.is_view,
            'parent_id': self.parent_id,
            'access_count': self.access_count,
            'age_seconds': time.time() - self.created_at
        }


class DataRegistry:
    """
    Registry данных с метаданными без сериализации.
    
    Thread-safe реестр для управления DataDescriptor'ами и
    отслеживания жизненного цикла данных.
    """
    
    def __init__(self, default_ttl: float = 3600.0):  # 1 час по умолчанию
        self._descriptors: Dict[str, DataDescriptor] = {}
        self._name_to_id: Dict[str, str] = {}
        self._tags_index: Dict[str, Set[str]] = {}
        self._parent_children: Dict[str, Set[str]] = {}
        self._lock = threading.RLock()
        self._default_ttl = default_ttl
        
        # Запуск cleanup thread
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_worker, daemon=True
        )
        self._cleanup_running = True
        self._cleanup_thread.start()
    
    def register(self, descriptor: DataDescriptor, overwrite: bool = False) -> bool:
        """Зарегистрировать дескриптор данных."""
        with self._lock:
            # Проверить существующую запись
            if descriptor.name in self._name_to_id:
                existing_id = self._name_to_id[descriptor.name]
                if not overwrite and existing_id != descriptor.id:
                    return False
                
                # Удалить старую запись
                self._unregister_internal(existing_id)
            
            # Добавить новую запись
            self._descriptors[descriptor.id] = descriptor
            self._name_to_id[descriptor.name] = descriptor.id
            
            # Индексировать теги
            for tag in descriptor.tags:
                if tag not in self._tags_index:
                    self._tags_index[tag] = set()
                self._tags_index[tag].add(descriptor.id)
            
            # Добавить связь родитель-ребенок
            if descriptor.parent_id:
                if descriptor.parent_id not in self._parent_children:
                    self._parent_children[descriptor.parent_id] = set()
                self._parent_children[descriptor.parent_id].add(descriptor.id)
            
            logger.debug(f"Зарегистрирован дескриптор {descriptor.name} ({descriptor.id})")
            return True
    
    def get_by_id(self, data_id: str) -> Optional[DataDescriptor]:
        """Получить дескриптор по ID."""
        with self._lock:
            descriptor = self._descriptors.get(data_id)
            if descriptor:
                descriptor.mark_accessed()
            return descriptor
    
    def get_by_name(self, name: str) -> Optional[DataDescriptor]:
        """Получить дескриптор по имени."""
        with self._lock:
            data_id = self._name_to_id.get(name)
            if data_id:
                return self.get_by_id(data_id)
            return None
    
    def find_by_tags(self, tags: Set[str], match_all: bool = True) -> List[DataDescriptor]:
        """Найти дескрипторы по тегам."""
        with self._lock:
            if match_all:
                # Пересечение всех тегов
                result_ids = None
                for tag in tags:
                    tag_ids = self._tags_index.get(tag, set())
                    result_ids = tag_ids if result_ids is None else result_ids & tag_ids
                    if not result_ids:
                        break
                result_ids = result_ids or set()
            else:
                # Объединение всех тегов
                result_ids = set()
                for tag in tags:
                    result_ids.update(self._tags_index.get(tag, set()))
            
            return [self._descriptors[data_id] for data_id in result_ids if data_id in self._descriptors]
    
    def get_children(self, parent_id: str) -> List[DataDescriptor]:
        """Получить дочерние дескрипторы."""
        with self._lock:
            child_ids = self._parent_children.get(parent_id, set())
            return [self._descriptors[child_id] for child_id in child_ids if child_id in self._descriptors]
    
    def unregister(self, data_id: str) -> bool:
        """Удалить дескриптор из registry."""
        with self._lock:
            return self._unregister_internal(data_id)
    
    def _unregister_internal(self, data_id: str) -> bool:
        """Внутренний метод удаления дескриптора."""
        if data_id not in self._descriptors:
            return False
        
        descriptor = self._descriptors.pop(data_id)
        
        # Удалить из индекса имен
        if descriptor.name in self._name_to_id:
            del self._name_to_id[descriptor.name]
        
        # Удалить из индекса тегов
        for tag in descriptor.tags:
            if tag in self._tags_index:
                self._tags_index[tag].discard(data_id)
                if not self._tags_index[tag]:
                    del self._tags_index[tag]
        
        # Удалить связи родитель-ребенок
        if descriptor.parent_id and descriptor.parent_id in self._parent_children:
            self._parent_children[descriptor.parent_id].discard(data_id)
            if not self._parent_children[descriptor.parent_id]:
                del self._parent_children[descriptor.parent_id]
        
        # Удалить детей
        if data_id in self._parent_children:
            for child_id in list(self._parent_children[data_id]):
                self._unregister_internal(child_id)
            del self._parent_children[data_id]
        
        return True
    
    def cleanup_expired(self, ttl_seconds: Optional[float] = None) -> int:
        """Очистить истекшие дескрипторы."""
        ttl = ttl_seconds or self._default_ttl
        expired_ids = []
        
        with self._lock:
            for data_id, descriptor in self._descriptors.items():
                if descriptor.is_expired(ttl):
                    expired_ids.append(data_id)
            
            for data_id in expired_ids:
                self._unregister_internal(data_id)
        
        if expired_ids:
            logger.info(f"Очищено {len(expired_ids)} истекших дескрипторов")
        
        return len(expired_ids)
    
    def get_stats(self) -> Dict[str, Any]:
        """Получить статистику registry."""
        with self._lock:
            total_size = sum(desc.size_bytes for desc in self._descriptors.values())
            status_counts = {}
            for desc in self._descriptors.values():
                status_counts[desc.status.value] = status_counts.get(desc.status.value, 0) + 1
            
            return {
                'total_descriptors': len(self._descriptors),
                'total_size_bytes': total_size,
                'status_distribution': status_counts,
                'unique_tags': len(self._tags_index),
                'parent_child_relationships': len(self._parent_children)
            }
    
    def _cleanup_worker(self):
        """Worker thread для автоматической очистки."""
        while self._cleanup_running:
            try:
                time.sleep(60)  # Проверка каждую минуту
                self.cleanup_expired()
            except Exception as e:
                logger.error(f"Ошибка в cleanup worker: {e}")
    
    def shutdown(self):
        """Остановить registry."""
        self._cleanup_running = False
        if self._cleanup_thread.is_alive():
            self._cleanup_thread.join(timeout=5)


class ZeroCopyDataManager:
    """
    Основной менеджер данных с zero-copy поддержкой.
    
    Обеспечивает прямое маппирование polars.Series через numpy arrays,
    управление shared memory и thread-safe операции без сериализации.
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
        
        self._registry = DataRegistry()
        self._shm_manager = get_shared_memory_manager()
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="zero_copy")
        self._lock = threading.RLock()
        self._initialized = True
        
        # Регистрировать cleanup при выходе
        import atexit
        atexit.register(self.cleanup)
    
    def store_array(self, 
                   array: np.ndarray, 
                   name: str, 
                   tags: Optional[Set[str]] = None,
                   ttl_seconds: Optional[float] = None) -> str:
        """
        Сохранить numpy array в shared memory с zero-copy.
        
        Args:
            array: numpy array для сохранения
            name: имя для данных
            tags: опциональные теги для индексирования
            ttl_seconds: время жизни (если None, используется по умолчанию)
            
        Returns:
            ID сохраненных данных
        """
        data_id = f"array_{uuid.uuid4().hex}"
        
        try:
            # Создать shared memory array
            shm_name, array_metadata = self._shm_manager.create_shared_array(array, data_id)
            
            # Создать дескриптор
            descriptor = DataDescriptor(
                id=data_id,
                name=name,
                status=DataStatus.READY,
                dtype=array_metadata.dtype,
                shape=array_metadata.shape,
                strides=array_metadata.strides,
                size_bytes=array_metadata.size_bytes,
                shm_name=array_metadata.shm_name,
                created_at=time.time(),
                accessed_at=time.time(),
                tags=tags or set()
            )
            
            # Зарегистрировать
            if not self._registry.register(descriptor):
                # Очистить shared memory если регистрация не удалась
                self._shm_manager.remove_shared_array(data_id)
                raise ValueError(f"Не удалось зарегистрировать данные с именем {name}")
            
            logger.info(f"Сохранен array {name} ({data_id}) размером {array.nbytes} байт")
            return data_id
            
        except Exception as e:
            logger.error(f"Ошибка сохранения array {name}: {e}")
            raise
    
    def store_dataframe(self, 
                       df: pl.DataFrame, 
                       name: str,
                       tags: Optional[Set[str]] = None) -> str:
        """
        Сохранить polars DataFrame в shared memory с zero-copy.
        
        Args:
            df: polars DataFrame для сохранения
            name: имя для данных
            tags: опциональные теги
            
        Returns:
            ID сохраненных данных
        """
        data_id = f"dataframe_{uuid.uuid4().hex}"
        
        try:
            # Создать shared memory для каждой колонки
            column_metadata = self._shm_manager.create_shared_dataframe(df, data_id)
            
            # Подсчитать общий размер
            total_size = sum(meta.size_bytes for meta in column_metadata.values())
            
            # Создать дескриптор
            descriptor = DataDescriptor(
                id=data_id,
                name=name,
                status=DataStatus.READY,
                dtype=np.dtype('O'),  # Object dtype для DataFrame
                shape=(len(df), len(df.columns)),
                strides=(),  # Не применимо для DataFrame
                size_bytes=total_size,
                shm_name=data_id,
                created_at=time.time(),
                accessed_at=time.time(),
                column_names=df.columns,
                column_metadata=column_metadata,
                tags=tags or set()
            )
            
            # Зарегистрировать
            if not self._registry.register(descriptor):
                # Очистить shared memory если регистрация не удалась
                for col_name in df.columns:
                    self._shm_manager.remove_shared_array(f"{data_id}_{col_name}")
                raise ValueError(f"Не удалось зарегистрировать DataFrame с именем {name}")
            
            logger.info(f"Сохранен DataFrame {name} ({data_id}): {len(df)} строк, {len(df.columns)} колонок, {total_size} байт")
            return data_id
            
        except Exception as e:
            logger.error(f"Ошибка сохранения DataFrame {name}: {e}")
            raise
    
    def get_array(self, name_or_id: str) -> Optional[np.ndarray]:
        """
        Получить numpy array по имени или ID с zero-copy доступом.
        
        Args:
            name_or_id: имя или ID данных
            
        Returns:
            numpy array или None если не найден
        """
        # Попытаться найти по имени или ID
        descriptor = self._registry.get_by_name(name_or_id) or self._registry.get_by_id(name_or_id)
        
        if not descriptor:
            return None
        
        if descriptor.status != DataStatus.READY:
            logger.warning(f"Данные {name_or_id} не готовы (статус: {descriptor.status})")
            return None
        
        try:
            # Получить shared array (zero-copy)
            array = self._shm_manager.get_shared_array(descriptor.id)
            descriptor.mark_accessed()
            return array
        except Exception as e:
            logger.error(f"Ошибка получения array {name_or_id}: {e}")
            # Обновить статус на ошибку
            descriptor.status = DataStatus.ERROR
            return None
    
    def get_dataframe(self, name_or_id: str) -> Optional[pl.DataFrame]:
        """
        Получить polars DataFrame по имени или ID с zero-copy доступом.
        
        Args:
            name_or_id: имя или ID данных
            
        Returns:
            polars DataFrame или None если не найден
        """
        # Попытаться найти по имени или ID
        descriptor = self._registry.get_by_name(name_or_id) or self._registry.get_by_id(name_or_id)
        
        if not descriptor:
            return None
        
        if descriptor.status != DataStatus.READY:
            logger.warning(f"Данные {name_or_id} не готовы (статус: {descriptor.status})")
            return None
        
        if not descriptor.column_metadata:
            logger.error(f"Данные {name_or_id} не являются DataFrame")
            return None
        
        try:
            # Восстановить DataFrame из shared memory (zero-copy)
            df = self._shm_manager.reconstruct_dataframe(
                descriptor.column_metadata, 
                descriptor.column_names
            )
            descriptor.mark_accessed()
            return df
        except Exception as e:
            logger.error(f"Ошибка получения DataFrame {name_or_id}: {e}")
            # Обновить статус на ошибку
            descriptor.status = DataStatus.ERROR
            return None
    
    def create_view(self, 
                   source_name_or_id: str, 
                   view_name: str, 
                   slice_obj: Optional[Union[slice, Tuple[slice, ...]]] = None) -> Optional[str]:
        """
        Создать view (представление) существующих данных без копирования.
        
        Args:
            source_name_or_id: исходные данные
            view_name: имя для view
            slice_obj: срез для создания view
            
        Returns:
            ID созданного view или None если ошибка
        """
        # Найти исходный дескриптор
        source_descriptor = (self._registry.get_by_name(source_name_or_id) or 
                           self._registry.get_by_id(source_name_or_id))
        
        if not source_descriptor or source_descriptor.status != DataStatus.READY:
            return None
        
        try:
            # Получить исходный array
            source_array = self.get_array(source_name_or_id)
            if source_array is None:
                return None
            
            # Создать view
            if slice_obj is not None:
                view_array = source_array[slice_obj]
            else:
                view_array = source_array.view()
            
            # Создать дескриптор для view
            view_id = f"view_{uuid.uuid4().hex}"
            view_descriptor = DataDescriptor(
                id=view_id,
                name=view_name,
                status=DataStatus.READY,
                dtype=view_array.dtype,
                shape=view_array.shape,
                strides=view_array.strides,
                size_bytes=view_array.nbytes,
                shm_name=source_descriptor.shm_name,  # Тот же shared memory
                created_at=time.time(),
                accessed_at=time.time(),
                is_view=True,
                parent_id=source_descriptor.id,
                tags=source_descriptor.tags.copy()
            )
            
            # Зарегистрировать view
            if self._registry.register(view_descriptor):
                logger.info(f"Создан view {view_name} ({view_id}) от {source_name_or_id}")
                return view_id
            else:
                return None
                
        except Exception as e:
            logger.error(f"Ошибка создания view {view_name}: {e}")
            return None
    
    def remove(self, name_or_id: str) -> bool:
        """
        Удалить данные по имени или ID.
        
        Args:
            name_or_id: имя или ID данных для удаления
            
        Returns:
            True если удалено, False если не найдено
        """
        # Найти дескриптор
        descriptor = self._registry.get_by_name(name_or_id) or self._registry.get_by_id(name_or_id)
        
        if not descriptor:
            return False
        
        try:
            # Удалить из shared memory если это не view
            if not descriptor.is_view:
                if descriptor.column_metadata:
                    # DataFrame - удалить все колонки
                    for col_name in descriptor.column_names:
                        self._shm_manager.remove_shared_array(f"{descriptor.id}_{col_name}")
                else:
                    # Array
                    self._shm_manager.remove_shared_array(descriptor.id)
            
            # Удалить из registry
            success = self._registry.unregister(descriptor.id)
            
            if success:
                logger.info(f"Удалены данные {name_or_id} ({descriptor.id})")
            
            return success
            
        except Exception as e:
            logger.error(f"Ошибка удаления данных {name_or_id}: {e}")
            return False
    
    def list_data(self, tags: Optional[Set[str]] = None) -> List[Dict[str, Any]]:
        """
        Получить список всех данных с метаинформацией.
        
        Args:
            tags: фильтр по тегам (если None, возвращает все)
            
        Returns:
            Список метаданных о данных
        """
        if tags:
            descriptors = self._registry.find_by_tags(tags, match_all=False)
        else:
            with self._registry._lock:
                descriptors = list(self._registry._descriptors.values())
        
        result = []
        for desc in descriptors:
            info = {
                'id': desc.id,
                'name': desc.name,
                'status': desc.status.value,
                'dtype': str(desc.dtype),
                'shape': desc.shape,
                'size_bytes': desc.size_bytes,
                'is_view': desc.is_view,
                'parent_id': desc.parent_id,
                'tags': list(desc.tags),
                'created_at': desc.created_at,
                'accessed_at': desc.accessed_at,
                'access_count': desc.access_count
            }
            
            if desc.column_metadata:
                info['type'] = 'dataframe'
                info['columns'] = desc.column_names
                info['column_count'] = len(desc.column_names)
            else:
                info['type'] = 'array'
            
            result.append(info)
        
        return result
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Получить информацию об использовании памяти."""
        shm_usage = self._shm_manager.get_memory_usage()
        registry_stats = self._registry.get_stats()
        
        return {
            'shared_memory': shm_usage,
            'registry': registry_stats,
            'total_data_items': registry_stats['total_descriptors']
        }
    
    @contextmanager
    def batch_operation(self):
        """
        Контекстный менеджер для пакетных операций.
        
        Позволяет выполнять несколько операций более эффективно.
        """
        with self._lock:
            yield self
    
    def cleanup_expired(self, ttl_seconds: Optional[float] = None) -> int:
        """Очистить истекшие данные."""
        return self._registry.cleanup_expired(ttl_seconds)
    
    def cleanup(self):
        """Полная очистка менеджера."""
        logger.info("Начало очистки ZeroCopyDataManager")
        
        # Остановить executor
        self._executor.shutdown(wait=True)
        
        # Очистить registry
        self._registry.shutdown()
        
        # Очистить shared memory
        self._shm_manager.cleanup_all()
        
        logger.info("Очистка ZeroCopyDataManager завершена")


# Глобальный экземпляр менеджера
_zero_copy_manager = None

def get_zero_copy_manager() -> ZeroCopyDataManager:
    """Получить глобальный экземпляр ZeroCopyDataManager."""
    global _zero_copy_manager
    if _zero_copy_manager is None:
        _zero_copy_manager = ZeroCopyDataManager()
    return _zero_copy_manager


# Convenience функции для простого использования

def store_array(array: np.ndarray, name: str, **kwargs) -> str:
    """Сохранить numpy array."""
    return get_zero_copy_manager().store_array(array, name, **kwargs)

def store_dataframe(df: pl.DataFrame, name: str, **kwargs) -> str:
    """Сохранить polars DataFrame."""
    return get_zero_copy_manager().store_dataframe(df, name, **kwargs)

def get_array(name_or_id: str) -> Optional[np.ndarray]:
    """Получить numpy array."""
    return get_zero_copy_manager().get_array(name_or_id)

def get_dataframe(name_or_id: str) -> Optional[pl.DataFrame]:
    """Получить polars DataFrame."""
    return get_zero_copy_manager().get_dataframe(name_or_id)

def remove_data(name_or_id: str) -> bool:
    """Удалить данные."""
    return get_zero_copy_manager().remove(name_or_id)

def list_all_data(**kwargs) -> List[Dict[str, Any]]:
    """Получить список всех данных."""
    return get_zero_copy_manager().list_data(**kwargs)