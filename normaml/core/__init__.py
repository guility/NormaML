"""
NormaML Core Module.

This module provides the core functionality of NormaML including:
- Data management with shared memory support
- Plugin system for extensibility
- Execution engine for parallel processing
- Experiment tracking integration
- Event bus for component communication
"""

from .interfaces import (
    SharedMemoryHandle,
    DataManager,
    Plugin,
    DataLoaderPlugin,
    FeatureEngineerPlugin,
    ModelTrainerPlugin,
    EvaluatorPlugin,
    PluginError,
    PluginInitializationError,
    PluginConfigurationError,
    SharedMemoryError,
    DataManagerError
)

from .data_manager import SharedMemoryDataManager
from .plugin_manager import PluginManager, PluginRegistry
from .execution_engine import ExecutionEngine, ExecutionResult, DockerExecutionConfig
from .experiment_tracker import ExperimentTracker
from .event_bus import (
    EventBus,
    Event,
    SystemEvent,
    DataEvent,
    PluginEvent,
    ExecutionEvent,
    ExperimentEvent,
    EventSubscription,
    get_global_event_bus,
    set_global_event_bus
)

__all__ = [
    # Interfaces
    "SharedMemoryHandle",
    "DataManager",
    "Plugin",
    "DataLoaderPlugin",
    "FeatureEngineerPlugin", 
    "ModelTrainerPlugin",
    "EvaluatorPlugin",
    
    # Exceptions
    "PluginError",
    "PluginInitializationError",
    "PluginConfigurationError",
    "SharedMemoryError",
    "DataManagerError",
    
    # Core components
    "SharedMemoryDataManager",
    "PluginManager",
    "PluginRegistry",
    "ExecutionEngine",
    "ExecutionResult",
    "DockerExecutionConfig",
    "ExperimentTracker",
    
    # Event system
    "EventBus",
    "Event",
    "SystemEvent",
    "DataEvent",
    "PluginEvent",
    "ExecutionEvent",
    "ExperimentEvent",
    "EventSubscription",
    "get_global_event_bus",
    "set_global_event_bus"
]