"""Экспорт компонентов подмодуля normaml.data для zero-copy операций."""
from .shared_memory import (
    get_shared_memory_manager,
    SharedMemoryManager,
    ArrayMetadata,
    create_memory_view,
    create_array_from_buffer,
    copy_to_shared_memory,
)

from .zero_copy import (
    get_zero_copy_manager,
    ZeroCopyDataManager,
    DataRegistry,
    DataDescriptor,
    DataStatus,
    store_array,
    store_dataframe,
    get_array,
    get_dataframe,
    remove_data,
    list_all_data,
)

from .streaming import (
    StreamingDataProcessor,
    process_csv_streaming,
    process_binary_streaming,
    create_streaming_dataframe,
)

from .optimization import (
    DTypeOptimizer,
    COWDataFrame,
    optimize_dataframe_dtypes,
    create_cow_dataframe,
    get_dtype_optimizer,
    get_column_index_manager,
)

__all__ = [
    "get_shared_memory_manager",
    "SharedMemoryManager",
    "ArrayMetadata",
    "create_memory_view",
    "create_array_from_buffer",
    "copy_to_shared_memory",
    "get_zero_copy_manager",
    "ZeroCopyDataManager",
    "DataRegistry",
    "DataDescriptor",
    "DataStatus",
    "store_array",
    "store_dataframe",
    "get_array",
    "get_dataframe",
    "remove_data",
    "list_all_data",
    "StreamingDataProcessor",
    "process_csv_streaming",
    "process_binary_streaming",
    "create_streaming_dataframe",
    "DTypeOptimizer",
    "COWDataFrame",
    "optimize_dataframe_dtypes",
    "create_cow_dataframe",
    "get_dtype_optimizer",
    "get_column_index_manager",
]