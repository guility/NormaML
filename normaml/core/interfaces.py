"""
Core interfaces for NormaML framework.

This module defines the fundamental interfaces and abstract classes that form
the backbone of the NormaML architecture, including shared memory management,
plugin system, and data processing abstractions.
"""

from abc import ABC, abstractmethod
from typing import Union, Dict, Any, Optional, List, TypeVar
import polars as pl
import pyarrow as pa


# Type variable for model types
ModelType = TypeVar('ModelType')


class SharedMemoryHandle:
    """
    Descriptor for shared memory operations enabling zero-copy data transfer.
    
    This class encapsulates the metadata needed to access shared memory segments
    containing Arrow-serialized data across processes and containers.
    
    Attributes:
        name: Unique identifier for the shared memory segment
        size: Size of the shared memory segment in bytes
        dtype: Data type identifier (e.g., 'arrow_ipc')
        shape: Shape tuple indicating dimensions of the data
    """
    
    def __init__(self, name: str, size: int, dtype: str, shape: tuple) -> None:
        """
        Initialize shared memory handle.
        
        Args:
            name: Unique name for the shared memory segment
            size: Size in bytes of the shared memory segment
            dtype: Data type identifier
            shape: Tuple representing data dimensions
        """
        self.name = name
        self.size = size
        self.dtype = dtype
        self.shape = shape
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert handle to dictionary for serialization.
        
        Returns:
            Dictionary representation of the handle
        """
        return {
            'name': self.name,
            'size': self.size,
            'dtype': self.dtype,
            'shape': self.shape
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SharedMemoryHandle':
        """
        Create handle from dictionary.
        
        Args:
            data: Dictionary containing handle metadata
            
        Returns:
            New SharedMemoryHandle instance
        """
        return cls(data['name'], data['size'], data['dtype'], data['shape'])
    
    def __repr__(self) -> str:
        return f"SharedMemoryHandle(name='{self.name}', size={self.size}, dtype='{self.dtype}', shape={self.shape})"


class DataManager(ABC):
    """
    Abstract base class for data management with zero-copy operations through shared memory.
    
    This interface defines the contract for loading, managing, and sharing data across
    processes and containers using shared memory and memory mapping techniques.
    """
    
    @abstractmethod
    def load_data(self, source: str, **kwargs) -> pl.LazyFrame:
        """
        Lazy loading of data from various sources.
        
        Args:
            source: Path or identifier of the data source
            **kwargs: Additional parameters for data loading
            
        Returns:
            Lazy frame for deferred execution
        """
        pass
    
    @abstractmethod
    def create_shared_memory_view(self, data: pl.DataFrame, name: str) -> SharedMemoryHandle:
        """
        Create shared memory representation of data for zero-copy access.
        
        Args:
            data: DataFrame to be shared
            name: Unique identifier for the shared memory segment
            
        Returns:
            Handle for accessing the shared memory segment
        """
        pass
    
    @abstractmethod
    def load_from_shared_memory(self, handle: SharedMemoryHandle) -> pl.LazyFrame:
        """
        Load data from shared memory using handle.
        
        Args:
            handle: Shared memory handle containing metadata
            
        Returns:
            Lazy frame containing the shared data
        """
        pass
    
    @abstractmethod
    def get_memory_usage(self) -> Dict[str, int]:
        """
        Get information about current memory usage.
        
        Returns:
            Dictionary with memory usage statistics
        """
        pass
    
    @abstractmethod
    def create_arrow_buffer(self, data: pl.DataFrame) -> pa.Buffer:
        """
        Create Arrow buffer for inter-process data exchange.
        
        Args:
            data: DataFrame to convert to Arrow buffer
            
        Returns:
            Arrow buffer containing serialized data
        """
        pass
    
    @abstractmethod
    def share_across_containers(self, data_id: str, container_ids: List[str]) -> Dict[str, Any]:
        """
        Prepare data for sharing across Docker containers.
        
        Args:
            data_id: Identifier of the data to share
            container_ids: List of container IDs that need access
            
        Returns:
            Configuration for container data sharing
        """
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """Clean up all shared memory resources."""
        pass


class Plugin(ABC):
    """
    Base interface for all NormaML plugins.
    
    This abstract class defines the common contract for all plugin types,
    including lifecycle management, configuration validation, and shared memory support.
    
    Attributes:
        name: Plugin identifier
        version: Plugin version string
        dependencies: List of required dependencies
        supports_shared_memory: Whether plugin supports shared memory operations
    """
    
    name: str
    version: str
    dependencies: List[str] = []
    supports_shared_memory: bool = True
    
    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize plugin with configuration.
        
        Args:
            config: Plugin configuration parameters
            
        Raises:
            ValueError: If configuration is invalid
        """
        pass
    
    @abstractmethod
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate plugin configuration.
        
        Args:
            config: Configuration to validate
            
        Returns:
            True if configuration is valid, False otherwise
        """
        pass
    
    @abstractmethod
    def get_capabilities(self) -> Dict[str, Any]:
        """
        Get plugin capabilities and supported features.
        
        Returns:
            Dictionary describing plugin capabilities
        """
        pass
    
    def setup_shared_memory(self, shm_config: Dict[str, Any]) -> None:
        """
        Configure shared memory settings for plugin.
        
        Args:
            shm_config: Shared memory configuration parameters
        """
        pass
    
    def cleanup(self) -> None:
        """Clean up plugin resources including shared memory."""
        pass


class DataLoaderPlugin(Plugin):
    """
    Plugin interface for data loading with shared memory support.
    
    This interface defines the contract for plugins that load data from various
    sources and can optionally load directly into shared memory for zero-copy access.
    """
    
    @abstractmethod
    def supports_source(self, source: str) -> bool:
        """
        Check if loader supports the given data source.
        
        Args:
            source: Data source path or identifier
            
        Returns:
            True if source is supported, False otherwise
        """
        pass
    
    @abstractmethod
    def load(self, source: str, **kwargs) -> pl.LazyFrame:
        """
        Load data from source.
        
        Args:
            source: Data source path or identifier
            **kwargs: Additional loading parameters
            
        Returns:
            Lazy frame containing the loaded data
        """
        pass
    
    @abstractmethod
    def load_to_shared_memory(self, source: str, shm_name: str, **kwargs) -> SharedMemoryHandle:
        """
        Load data directly into shared memory.
        
        Args:
            source: Data source path or identifier
            shm_name: Name for the shared memory segment
            **kwargs: Additional loading parameters
            
        Returns:
            Handle for the shared memory segment
        """
        pass
    
    @abstractmethod
    def get_schema(self, source: str) -> Dict[str, str]:
        """
        Get data schema from source.
        
        Args:
            source: Data source path or identifier
            
        Returns:
            Dictionary mapping column names to data types
        """
        pass
    
    @abstractmethod
    def estimate_size(self, source: str) -> int:
        """
        Estimate data size in bytes.
        
        Args:
            source: Data source path or identifier
            
        Returns:
            Estimated size in bytes
        """
        pass
    
    def supports_streaming(self) -> bool:
        """
        Check if plugin supports streaming data loading.
        
        Returns:
            True if streaming is supported, False otherwise
        """
        return False


class FeatureEngineerPlugin(Plugin):
    """
    Plugin interface for feature engineering with shared memory support.
    
    This interface defines the contract for plugins that perform feature analysis,
    selection, and creation operations on datasets.
    """
    
    @abstractmethod
    def analyze_features(self, data_handle: Union[pl.LazyFrame, SharedMemoryHandle]) -> Dict[str, Any]:
        """
        Analyze features in the dataset.
        
        Args:
            data_handle: Data source (lazy frame or shared memory handle)
            
        Returns:
            Dictionary containing feature analysis results
        """
        pass
    
    @abstractmethod
    def select_features(self, data_handle: Union[pl.LazyFrame, SharedMemoryHandle], 
                       target: str, **kwargs) -> List[str]:
        """
        Select relevant features for modeling.
        
        Args:
            data_handle: Data source (lazy frame or shared memory handle)
            target: Name of the target column
            **kwargs: Additional selection parameters
            
        Returns:
            List of selected feature names
        """
        pass
    
    @abstractmethod
    def create_features(self, data_handle: Union[pl.LazyFrame, SharedMemoryHandle], 
                       config: Dict[str, Any]) -> Union[pl.LazyFrame, SharedMemoryHandle]:
        """
        Create new features based on configuration.
        
        Args:
            data_handle: Data source (lazy frame or shared memory handle)
            config: Feature creation configuration
            
        Returns:
            Data with new features (same type as input)
        """
        pass
    
    @abstractmethod
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        pass


class ModelTrainerPlugin(Plugin):
    """
    Plugin interface for model training with shared memory support.
    
    This interface defines the contract for plugins that train machine learning
    models using various algorithms and configurations.
    """
    
    @abstractmethod
    def train(self, data_handle: Union[pl.LazyFrame, SharedMemoryHandle], 
              target: str, config: Dict[str, Any]) -> ModelType:
        """
        Train a model on the provided data.
        
        Args:
            data_handle: Training data (lazy frame or shared memory handle)
            target: Name of the target column
            config: Training configuration parameters
            
        Returns:
            Trained model instance
        """
        pass
    
    @abstractmethod
    def grid_search(self, data_handle: Union[pl.LazyFrame, SharedMemoryHandle], 
                   target: str, param_grid: Dict[str, List[Any]]) -> Dict[str, Any]:
        """
        Perform grid search for hyperparameter tuning.
        
        Args:
            data_handle: Training data (lazy frame or shared memory handle)
            target: Name of the target column
            param_grid: Grid of parameters to search
            
        Returns:
            Dictionary containing best parameters and results
        """
        pass
    
    @abstractmethod
    def supports_task_type(self, task_type: str) -> bool:
        """
        Check if trainer supports the given task type.
        
        Args:
            task_type: Type of ML task (e.g., 'classification', 'regression')
            
        Returns:
            True if task type is supported, False otherwise
        """
        pass
    
    @abstractmethod
    def get_default_params(self, task_type: str) -> Dict[str, Any]:
        """
        Get default parameters for a task type.
        
        Args:
            task_type: Type of ML task
            
        Returns:
            Dictionary of default parameters
        """
        pass
    
    def supports_incremental_training(self) -> bool:
        """
        Check if trainer supports incremental training.
        
        Returns:
            True if incremental training is supported, False otherwise
        """
        return False


class EvaluatorPlugin(Plugin):
    """
    Plugin interface for model evaluation with shared memory support.
    
    This interface defines the contract for plugins that evaluate trained models
    using various metrics and validation strategies.
    """
    
    @abstractmethod
    def evaluate(self, model: ModelType, 
                test_data_handle: Union[pl.LazyFrame, SharedMemoryHandle], 
                target: str) -> Dict[str, float]:
        """
        Evaluate model performance on test data.
        
        Args:
            model: Trained model to evaluate
            test_data_handle: Test data (lazy frame or shared memory handle)
            target: Name of the target column
            
        Returns:
            Dictionary mapping metric names to values
        """
        pass
    
    @abstractmethod
    def cross_validate(self, trainer: ModelTrainerPlugin,
                      data_handle: Union[pl.LazyFrame, SharedMemoryHandle], 
                      target: str, cv_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform cross-validation.
        
        Args:
            trainer: Model trainer plugin to use
            data_handle: Data for cross-validation (lazy frame or shared memory handle)
            target: Name of the target column
            cv_config: Cross-validation configuration
            
        Returns:
            Dictionary containing cross-validation results
        """
        pass
    
    @abstractmethod
    def get_supported_metrics(self, task_type: str) -> List[str]:
        """
        Get list of supported metrics for task type.
        
        Args:
            task_type: Type of ML task
            
        Returns:
            List of supported metric names
        """
        pass


# Exception classes for plugin system
class PluginError(Exception):
    """Base exception for plugin-related errors."""
    pass


class PluginInitializationError(PluginError):
    """Raised when plugin initialization fails."""
    pass


class PluginConfigurationError(PluginError):
    """Raised when plugin configuration is invalid."""
    pass


class SharedMemoryError(Exception):
    """Base exception for shared memory operations."""
    pass


class DataManagerError(Exception):
    """Base exception for data manager operations."""
    pass