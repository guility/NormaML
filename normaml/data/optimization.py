"""
Оптимизации для эффективной работы с данными.

Включает COWDataFrame с прямым memory sharing, dtype optimization 
для numpy arrays и column indexing для быстрого доступа.
"""

import weakref
import threading
from typing import Dict, List, Optional, Union, Tuple, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import defaultdict
import hashlib
import gc

try:
    import numpy as np
except ImportError:
    raise ImportError("numpy is required for data optimization")

try:
    import polars as pl
except ImportError:
    raise ImportError("polars is required for data optimization")

from .zero_copy import get_zero_copy_manager, DataDescriptor
from .shared_memory import get_shared_memory_manager

logger = logging.getLogger(__name__)


class OptimizationLevel(Enum):
    """Уровни оптимизации данных."""
    NONE = "none"
    BASIC = "basic"
    AGGRESSIVE = "aggressive"
    EXTREME = "extreme"


@dataclass
class DTypeOptimizationResult:
    """Результат оптимизации типов данных."""
    original_dtype: np.dtype
    optimized_dtype: np.dtype
    memory_saved_bytes: int
    compression_ratio: float
    is_lossy: bool = False
    min_value: Optional[float] = None
    max_value: Optional[float] = None


@dataclass
class ColumnIndex:
    """Индекс для быстрого доступа к колонкам."""
    name: str
    dtype: np.dtype
    unique_values: int
    null_count: int
    min_value: Optional[Any] = None
    max_value: Optional[Any] = None
    memory_usage: int = 0
    is_sorted: bool = False
    has_duplicates: bool = True
    hash_signature: Optional[str] = None


class DTypeOptimizer:
    """
    Оптимизатор типов данных для numpy arrays.
    
    Автоматически подбирает наиболее эффективные типы данных
    для экономии памяти без потери точности.
    """
    
    def __init__(self, optimization_level: OptimizationLevel = OptimizationLevel.BASIC):
        self.optimization_level = optimization_level
        
        # Правила оптимизации для различных типов
        self._int_optimizations = [
            (np.int8, -128, 127),
            (np.int16, -32768, 32767),
            (np.int32, -2147483648, 2147483647),
            (np.int64, -9223372036854775808, 9223372036854775807)
        ]
        
        self._uint_optimizations = [
            (np.uint8, 0, 255),
            (np.uint16, 0, 65535),
            (np.uint32, 0, 4294967295),
            (np.uint64, 0, 18446744073709551615)
        ]
        
        self._float_optimizations = [
            (np.float16, -65504, 65504),  # Приближенные границы
            (np.float32, -3.4e38, 3.4e38),
            (np.float64, -1.8e308, 1.8e308)
        ]
    
    def optimize_array(self, array: np.ndarray, allow_lossy: bool = False) -> DTypeOptimizationResult:
        """
        Оптимизировать тип данных numpy array.
        
        Args:
            array: исходный numpy array
            allow_lossy: разрешить потерю точности для лучшего сжатия
            
        Returns:
            Результат оптимизации с новым dtype и статистикой
        """
        original_dtype = array.dtype
        original_size = array.nbytes
        
        # Определить категорию типа данных
        if np.issubdtype(original_dtype, np.integer):
            result = self._optimize_integer_array(array, allow_lossy)
        elif np.issubdtype(original_dtype, np.floating):
            result = self._optimize_float_array(array, allow_lossy)
        elif np.issubdtype(original_dtype, np.bool_):
            # bool уже оптимален
            result = DTypeOptimizationResult(
                original_dtype=original_dtype,
                optimized_dtype=original_dtype,
                memory_saved_bytes=0,
                compression_ratio=1.0
            )
        else:
            # Строки и другие типы - пока не оптимизируем
            result = DTypeOptimizationResult(
                original_dtype=original_dtype,
                optimized_dtype=original_dtype,
                memory_saved_bytes=0,
                compression_ratio=1.0
            )
        
        # Вычислить сэкономленную память
        if result.optimized_dtype != original_dtype:
            new_size = array.size * result.optimized_dtype.itemsize
            result.memory_saved_bytes = original_size - new_size
            result.compression_ratio = original_size / new_size if new_size > 0 else 1.0
        
        return result
    
    def _optimize_integer_array(self, array: np.ndarray, allow_lossy: bool) -> DTypeOptimizationResult:
        """Оптимизировать integer array."""
        min_val = np.min(array)
        max_val = np.max(array)
        
        # Выбрать оптимальный тип
        optimized_dtype = array.dtype
        
        # Сначала проверить unsigned типы если все значения >= 0
        if min_val >= 0:
            for dtype, dtype_min, dtype_max in self._uint_optimizations:
                if min_val >= dtype_min and max_val <= dtype_max:
                    optimized_dtype = dtype
                    break
        
        # Проверить signed типы
        for dtype, dtype_min, dtype_max in self._int_optimizations:
            if min_val >= dtype_min and max_val <= dtype_max:
                if optimized_dtype == array.dtype or dtype.itemsize < optimized_dtype.itemsize:
                    optimized_dtype = dtype
                break
        
        return DTypeOptimizationResult(
            original_dtype=array.dtype,
            optimized_dtype=optimized_dtype,
            memory_saved_bytes=0,  # Будет вычислено в optimize_array
            compression_ratio=1.0,
            min_value=float(min_val),
            max_value=float(max_val)
        )
    
    def _optimize_float_array(self, array: np.ndarray, allow_lossy: bool) -> DTypeOptimizationResult:
        """Оптимизировать float array."""
        min_val = np.min(array)
        max_val = np.max(array)
        
        optimized_dtype = array.dtype
        is_lossy = False
        
        # Проверить можно ли использовать меньший float тип
        if self.optimization_level in [OptimizationLevel.AGGRESSIVE, OptimizationLevel.EXTREME]:
            # Проверить float32 если исходный float64
            if array.dtype == np.float64:
                # Проверить помещаются ли значения в float32
                if (min_val >= -3.4e38 and max_val <= 3.4e38 and 
                    (allow_lossy or self._check_float32_precision_loss(array))):
                    optimized_dtype = np.float32
                    is_lossy = not self._check_float32_precision_loss(array)
            
            # Проверить float16 для экстремальной оптимизации
            if (self.optimization_level == OptimizationLevel.EXTREME and 
                allow_lossy and 
                min_val >= -65504 and max_val <= 65504):
                optimized_dtype = np.float16
                is_lossy = True
        
        return DTypeOptimizationResult(
            original_dtype=array.dtype,
            optimized_dtype=optimized_dtype,
            memory_saved_bytes=0,
            compression_ratio=1.0,
            is_lossy=is_lossy,
            min_value=float(min_val),
            max_value=float(max_val)
        )
    
    def _check_float32_precision_loss(self, array: np.ndarray, sample_size: int = 1000) -> bool:
        """Проверить будет ли потеря точности при конверсии в float32."""
        if len(array) > sample_size:
            # Проверить на sample для больших массивов
            sample_indices = np.random.choice(len(array), sample_size, replace=False)
            sample = array[sample_indices]
        else:
            sample = array
        
        # Конвертировать в float32 и обратно
        converted = sample.astype(np.float32).astype(array.dtype)
        
        # Проверить различия
        diff = np.abs(sample - converted)
        max_diff = np.max(diff)
        
        # Считаем приемлемым если максимальная разница < 1e-6
        return max_diff < 1e-6


class COWDataFrame:
    """
    Copy-on-Write DataFrame с прямым memory sharing.
    
    Обеспечивает эффективное копирование DataFrame'ов через
    shared memory до первого изменения данных.
    """
    
    def __init__(self, 
                 data: Optional[Union[pl.DataFrame, Dict[str, np.ndarray]]] = None,
                 name: Optional[str] = None,
                 parent: Optional['COWDataFrame'] = None):
        self._name = name or f"cow_df_{id(self)}"
        self._parent = parent
        self._children: Set[weakref.ReferenceType] = set()
        self._is_modified = False
        self._lock = threading.RLock()
        
        # Менеджеры
        self._zero_copy_manager = get_zero_copy_manager()
        self._optimizer = DTypeOptimizer()
        
        # Данные
        self._data_id: Optional[str] = None
        self._column_index: Dict[str, ColumnIndex] = {}
        
        if data is not None:
            self._initialize_data(data)
        
        # Регистрировать у родителя
        if parent:
            parent._add_child(self)
    
    def _initialize_data(self, data: Union[pl.DataFrame, Dict[str, np.ndarray]]):
        """Инициализировать данные."""
        if isinstance(data, pl.DataFrame):
            # Сохранить DataFrame в zero-copy storage
            self._data_id = self._zero_copy_manager.store_dataframe(
                data, self._name, tags={'cow', 'dataframe'}
            )
        elif isinstance(data, dict):
            # Создать DataFrame из словаря numpy arrays
            df = pl.DataFrame(data)
            self._data_id = self._zero_copy_manager.store_dataframe(
                df, self._name, tags={'cow', 'dataframe'}
            )
        else:
            raise ValueError("Данные должны быть polars.DataFrame или Dict[str, np.ndarray]")
        
        # Создать индекс колонок
        self._build_column_index()
    
    def _build_column_index(self):
        """Построить индекс колонок для быстрого доступа."""
        if not self._data_id:
            return
        
        df = self._zero_copy_manager.get_dataframe(self._data_id)
        if df is None:
            return
        
        self._column_index = {}
        
        for col_name in df.columns:
            series = df[col_name]
            
            # Получить numpy array для анализа
            array = series.to_numpy()
            
            # Создать индекс
            col_index = ColumnIndex(
                name=col_name,
                dtype=array.dtype,
                unique_values=len(np.unique(array)) if array.size > 0 else 0,
                null_count=int(series.null_count()),
                memory_usage=array.nbytes,
                is_sorted=self._check_sorted(array),
                has_duplicates=len(np.unique(array)) < len(array) if array.size > 0 else False
            )
            
            # Добавить min/max для числовых типов
            if np.issubdtype(array.dtype, np.number) and array.size > 0:
                col_index.min_value = float(np.min(array))
                col_index.max_value = float(np.max(array))
            
            # Создать hash signature для детекции дублированных колонок
            col_index.hash_signature = self._compute_column_hash(array)
            
            self._column_index[col_name] = col_index
    
    def _check_sorted(self, array: np.ndarray) -> bool:
        """Проверить отсортирован ли массив."""
        if array.size <= 1:
            return True
        
        # Для больших массивов проверяем sample
        if array.size > 10000:
            sample_size = 1000
            indices = np.linspace(0, len(array) - 1, sample_size, dtype=int)
            sample = array[indices]
            return np.all(sample[:-1] <= sample[1:])
        else:
            return np.all(array[:-1] <= array[1:])
    
    def _compute_column_hash(self, array: np.ndarray) -> str:
        """Вычислить hash signature для колонки."""
        # Для больших массивов используем sample
        if array.size > 10000:
            sample_size = 1000
            indices = np.linspace(0, len(array) - 1, sample_size, dtype=int)
            sample_data = array[indices].tobytes()
        else:
            sample_data = array.tobytes()
        
        return hashlib.md5(sample_data).hexdigest()
    
    def _add_child(self, child: 'COWDataFrame'):
        """Добавить дочерний DataFrame."""
        with self._lock:
            self._children.add(weakref.ref(child))
    
    def _copy_on_write(self) -> str:
        """Выполнить копирование при записи."""
        if self._is_modified or not self._parent:
            return self._data_id
        
        with self._lock:
            if self._is_modified:  # Double-check
                return self._data_id
            
            # Скопировать данные от родителя
            parent_df = self._zero_copy_manager.get_dataframe(self._parent._data_id)
            if parent_df is None:
                raise RuntimeError("Родительские данные недоступны")
            
            # Создать копию с новым именем
            new_name = f"{self._name}_cow_copy"
            self._data_id = self._zero_copy_manager.store_dataframe(
                parent_df, new_name, tags={'cow', 'dataframe', 'copy'}
            )
            
            self._is_modified = True
            self._build_column_index()
            
            logger.debug(f"COW копирование выполнено для {self._name}")
            return self._data_id
    
    @property
    def data(self) -> Optional[pl.DataFrame]:
        """Получить данные DataFrame."""
        if not self._data_id:
            return None
        return self._zero_copy_manager.get_dataframe(self._data_id)
    
    @property
    def columns(self) -> List[str]:
        """Получить список колонок."""
        return list(self._column_index.keys())
    
    @property
    def shape(self) -> Tuple[int, int]:
        """Получить размеры DataFrame."""
        df = self.data
        return (len(df), len(df.columns)) if df is not None else (0, 0)
    
    def get_column(self, name: str) -> Optional[np.ndarray]:
        """Получить колонку как numpy array."""
        df = self.data
        if df is None or name not in df.columns:
            return None
        
        return df[name].to_numpy()
    
    def get_column_index(self, name: str) -> Optional[ColumnIndex]:
        """Получить индекс колонки."""
        return self._column_index.get(name)
    
    def optimize_dtypes(self, 
                       allow_lossy: bool = False,
                       columns: Optional[List[str]] = None) -> Dict[str, DTypeOptimizationResult]:
        """
        Оптимизировать типы данных колонок.
        
        Args:
            allow_lossy: разрешить потерю точности
            columns: список колонок для оптимизации (если None, все колонки)
            
        Returns:
            Словарь результатов оптимизации по колонкам
        """
        # Выполнить COW если нужно
        self._copy_on_write()
        
        df = self.data
        if df is None:
            return {}
        
        target_columns = columns or df.columns
        optimization_results = {}
        optimized_data = {}
        
        for col_name in target_columns:
            if col_name not in df.columns:
                continue
            
            array = df[col_name].to_numpy()
            result = self._optimizer.optimize_array(array, allow_lossy)
            
            optimization_results[col_name] = result
            
            # Применить оптимизацию если есть улучшение
            if result.optimized_dtype != result.original_dtype:
                optimized_array = array.astype(result.optimized_dtype)
                optimized_data[col_name] = optimized_array
                
                logger.info(f"Оптимизирована колонка {col_name}: "
                           f"{result.original_dtype} -> {result.optimized_dtype}, "
                           f"сэкономлено {result.memory_saved_bytes} байт")
        
        # Создать новый DataFrame с оптимизированными типами если есть изменения
        if optimized_data:
            # Скопировать неизмененные колонки
            final_data = {}
            for col in df.columns:
                if col in optimized_data:
                    final_data[col] = optimized_data[col]
                else:
                    final_data[col] = df[col].to_numpy()
            
            # Создать новый DataFrame
            optimized_df = pl.DataFrame(final_data)
            
            # Обновить данные
            new_name = f"{self._name}_optimized"
            self._data_id = self._zero_copy_manager.store_dataframe(
                optimized_df, new_name, tags={'cow', 'dataframe', 'optimized'}
            )
            
            # Обновить индекс
            self._build_column_index()
        
        return optimization_results
    
    def find_duplicate_columns(self) -> Dict[str, List[str]]:
        """
        Найти дублированные колонки на основе hash signatures.
        
        Returns:
            Словарь hash -> список имен колонок с таким же hash
        """
        hash_groups = defaultdict(list)
        
        for col_name, col_index in self._column_index.items():
            if col_index.hash_signature:
                hash_groups[col_index.hash_signature].append(col_name)
        
        # Вернуть только группы с дубликатами
        return {h: cols for h, cols in hash_groups.items() if len(cols) > 1}
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Получить информацию об использовании памяти."""
        total_memory = sum(idx.memory_usage for idx in self._column_index.values())
        
        return {
            'total_bytes': total_memory,
            'column_count': len(self._column_index),
            'row_count': self.shape[0],
            'is_cow_copy': self._is_modified,
            'has_parent': self._parent is not None,
            'children_count': len([ref for ref in self._children if ref() is not None]),
            'column_details': {
                name: {
                    'dtype': str(idx.dtype),
                    'memory_bytes': idx.memory_usage,
                    'unique_values': idx.unique_values,
                    'null_count': idx.null_count,
                    'is_sorted': idx.is_sorted
                }
                for name, idx in self._column_index.items()
            }
        }
    
    def copy(self, deep: bool = False) -> 'COWDataFrame':
        """
        Создать копию DataFrame.
        
        Args:
            deep: если False, создается COW копия; если True, полная копия
            
        Returns:
            Новый COWDataFrame
        """
        if deep:
            # Полная копия данных
            df = self.data
            if df is None:
                return COWDataFrame()
            
            return COWDataFrame(df.clone(), name=f"{self._name}_deep_copy")
        else:
            # COW копия
            return COWDataFrame(parent=self, name=f"{self._name}_cow")
    
    def select_columns(self, columns: List[str]) -> 'COWDataFrame':
        """
        Выбрать подмножество колонок (создает COW view).
        
        Args:
            columns: список имен колонок
            
        Returns:
            Новый COWDataFrame с выбранными колонками
        """
        df = self.data
        if df is None:
            return COWDataFrame()
        
        # Проверить существование колонок
        existing_columns = [col for col in columns if col in df.columns]
        if not existing_columns:
            return COWDataFrame()
        
        # Создать новый DataFrame с выбранными колонками
        selected_df = df.select(existing_columns)
        
        return COWDataFrame(
            selected_df, 
            name=f"{self._name}_select_{len(existing_columns)}cols"
        )
    
    def __len__(self) -> int:
        """Количество строк."""
        return self.shape[0]
    
    def __repr__(self) -> str:
        """Строковое представление."""
        shape = self.shape
        cow_status = "COW" if not self._is_modified and self._parent else "OWNED"
        return f"COWDataFrame(name='{self._name}', shape={shape}, status={cow_status})"


class ColumnIndexManager:
    """
    Менеджер индексов колонок для быстрого поиска и фильтрации.
    
    Поддерживает индексы по значениям, статистике и metadata.
    """
    
    def __init__(self):
        self._indexes: Dict[str, Dict[str, Any]] = {}
        self._global_stats: Dict[str, Any] = {}
        self._lock = threading.RLock()
    
    def create_index(self, 
                    dataframe_id: str, 
                    column_name: str, 
                    array: np.ndarray,
                    index_type: str = 'basic') -> ColumnIndex:
        """
        Создать индекс для колонки.
        
        Args:
            dataframe_id: идентификатор DataFrame
            column_name: имя колонки
            array: данные колонки
            index_type: тип индекса ('basic', 'sorted', 'hash')
            
        Returns:
            Созданный индекс колонки
        """
        with self._lock:
            index_key = f"{dataframe_id}:{column_name}"
            
            # Создать базовый индекс
            col_index = ColumnIndex(
                name=column_name,
                dtype=array.dtype,
                unique_values=len(np.unique(array)) if array.size > 0 else 0,
                null_count=int(np.sum(np.isnan(array))) if np.issubdtype(array.dtype, np.floating) else 0,
                memory_usage=array.nbytes,
                is_sorted=np.all(array[:-1] <= array[1:]) if array.size > 1 else True,
                has_duplicates=len(np.unique(array)) < len(array) if array.size > 0 else False
            )
            
            # Добавить min/max для числовых типов
            if np.issubdtype(array.dtype, np.number) and array.size > 0:
                col_index.min_value = float(np.min(array))
                col_index.max_value = float(np.max(array))
            
            # Создать hash signature
            col_index.hash_signature = hashlib.md5(array.tobytes()).hexdigest()
            
            # Сохранить индекс
            if dataframe_id not in self._indexes:
                self._indexes[dataframe_id] = {}
            
            self._indexes[dataframe_id][column_name] = {
                'index': col_index,
                'type': index_type,
                'created_at': time.time()
            }
            
            # Обновить глобальную статистику
            self._update_global_stats(col_index)
            
            logger.debug(f"Создан индекс {index_type} для {index_key}")
            return col_index
    
    def find_similar_columns(self, 
                           target_index: ColumnIndex,
                           similarity_threshold: float = 0.9) -> List[Tuple[str, str, float]]:
        """
        Найти похожие колонки на основе статистики.
        
        Args:
            target_index: целевой индекс для поиска похожих
            similarity_threshold: порог схожести (0.0 - 1.0)
            
        Returns:
            Список (dataframe_id, column_name, similarity_score)
        """
        similar_columns = []
        
        with self._lock:
            for df_id, df_indexes in self._indexes.items():
                for col_name, col_data in df_indexes.items():
                    col_index = col_data['index']
                    
                    # Пропустить если тот же индекс
                    if col_index is target_index:
                        continue
                    
                    # Вычислить схожесть
                    similarity = self._compute_similarity(target_index, col_index)
                    
                    if similarity >= similarity_threshold:
                        similar_columns.append((df_id, col_name, similarity))
        
        # Отсортировать по схожести
        similar_columns.sort(key=lambda x: x[2], reverse=True)
        return similar_columns
    
    def _compute_similarity(self, index1: ColumnIndex, index2: ColumnIndex) -> float:
        """Вычислить схожесть между двумя индексами колонок."""
        score = 0.0
        factors = 0
        
        # Проверить dtype
        if index1.dtype == index2.dtype:
            score += 0.3
        factors += 1
        
        # Проверить hash signature (точное совпадение данных)
        if index1.hash_signature == index2.hash_signature:
            return 1.0  # Точное совпадение
        
        # Проверить количество уникальных значений
        if index1.unique_values > 0 and index2.unique_values > 0:
            unique_ratio = min(index1.unique_values, index2.unique_values) / max(index1.unique_values, index2.unique_values)
            score += unique_ratio * 0.2
        factors += 1
        
        # Проверить null count
        total_nulls = max(index1.null_count + index2.null_count, 1)
        null_similarity = 1.0 - abs(index1.null_count - index2.null_count) / total_nulls
        score += null_similarity * 0.1
        factors += 1
        
        # Проверить отсортированность
        if index1.is_sorted == index2.is_sorted:
            score += 0.1
        factors += 1
        
        # Проверить наличие дубликатов
        if index1.has_duplicates == index2.has_duplicates:
            score += 0.1
        factors += 1
        
        # Проверить range для числовых типов
        if (index1.min_value is not None and index1.max_value is not None and
            index2.min_value is not None and index2.max_value is not None):
            
            range1 = index1.max_value - index1.min_value
            range2 = index2.max_value - index2.min_value
            
            if range1 > 0 and range2 > 0:
                range_ratio = min(range1, range2) / max(range1, range2)
                score += range_ratio * 0.2
                factors += 1
        
        return score / factors if factors > 0 else 0.0
    
    def _update_global_stats(self, col_index: ColumnIndex):
        """Обновить глобальную статистику."""
        dtype_str = str(col_index.dtype)
        
        if 'dtype_distribution' not in self._global_stats:
            self._global_stats['dtype_distribution'] = defaultdict(int)
        
        self._global_stats['dtype_distribution'][dtype_str] += 1
        
        # Обновить общую статистику памяти
        if 'total_memory' not in self._global_stats:
            self._global_stats['total_memory'] = 0
        self._global_stats['total_memory'] += col_index.memory_usage
    
    def get_global_stats(self) -> Dict[str, Any]:
        """Получить глобальную статистику индексов."""
        with self._lock:
            return dict(self._global_stats)
    
    def cleanup_indexes(self, dataframe_id: str):
        """Очистить индексы для DataFrame."""
        with self._lock:
            if dataframe_id in self._indexes:
                del self._indexes[dataframe_id]
                logger.debug(f"Очищены индексы для DataFrame {dataframe_id}")


# Глобальные экземпляры
_dtype_optimizer = None
_column_index_manager = None

def get_dtype_optimizer() -> DTypeOptimizer:
    """Получить глобальный экземпляр DTypeOptimizer."""
    global _dtype_optimizer
    if _dtype_optimizer is None:
        _dtype_optimizer = DTypeOptimizer()
    return _dtype_optimizer

def get_column_index_manager() -> ColumnIndexManager:
    """Получить глобальный экземпляр ColumnIndexManager."""
    global _column_index_manager
    if _column_index_manager is None:
        _column_index_manager = ColumnIndexManager()
    return _column_index_manager


# Convenience функции

def optimize_dataframe_dtypes(df: pl.DataFrame, 
                            allow_lossy: bool = False) -> Tuple[pl.DataFrame, Dict[str, DTypeOptimizationResult]]:
    """
    Оптимизировать типы данных polars DataFrame.
    
    Args:
        df: исходный DataFrame
        allow_lossy: разрешить потерю точности
        
    Returns:
        Tuple[оптимизированный DataFrame, результаты оптимизации]
    """
    optimizer = get_dtype_optimizer()
    optimization_results = {}
    optimized_data = {}
    
    for col_name in df.columns:
        array = df[col_name].to_numpy()
        result = optimizer.optimize_array(array, allow_lossy)
        optimization_results[col_name] = result
        
        if result.optimized_dtype != result.original_dtype:
            optimized_data[col_name] = array.astype(result.optimized_dtype)
        else:
            optimized_data[col_name] = array
    
    optimized_df = pl.DataFrame(optimized_data)
    return optimized_df, optimization_results


def create_cow_dataframe(data: Union[pl.DataFrame, Dict[str, np.ndarray]], 
                        name: Optional[str] = None) -> COWDataFrame:
    """Создать COWDataFrame."""
    return COWDataFrame(data, name)


# Импорт времени для временных меток
import time