"""
NormaML - Modular Machine Learning Library with Zero-Copy Data Processing.

This library provides a plugin-based architecture for machine learning pipelines
with focus on zero-copy data processing and efficient resource utilization.
"""

from normaml.__version__ import __version__, check_python_version

# Check Python version compatibility on import
check_python_version()

# Import core components
from . import core
from . import config

# Core interface imports for convenient access
from .core import (
    # Main managers
    SharedMemoryDataManager,
    PluginManager,
    ExecutionEngine,
    ExperimentTracker,
    EventBus,
    
    # Interfaces
    DataManager,
    Plugin,
    DataLoaderPlugin,
    FeatureEngineerPlugin,
    ModelTrainerPlugin,
    EvaluatorPlugin,
    
    # Shared memory
    SharedMemoryHandle,
    
    # Events
    Event,
    SystemEvent,
    DataEvent,
    PluginEvent,
    ExecutionEvent,
    ExperimentEvent,
    
    # Event bus utilities
    get_global_event_bus,
    set_global_event_bus
)

# Configuration imports
from .config import (
    ConfigManager,
    get_config_for_environment,
    get_default_config,
    validate_environment_name
)

# Public API exports
__all__ = [
    "__version__",
    
    # Core modules
    "core",
    "config",
    
    # Main components
    "SharedMemoryDataManager",
    "PluginManager",
    "ExecutionEngine",
    "ExperimentTracker",
    "EventBus",
    "ConfigManager",
    
    # Interfaces
    "DataManager",
    "Plugin",
    "DataLoaderPlugin",
    "FeatureEngineerPlugin",
    "ModelTrainerPlugin",
    "EvaluatorPlugin",
    
    # Shared memory
    "SharedMemoryHandle",
    
    # Events
    "Event",
    "SystemEvent",
    "DataEvent",
    "PluginEvent",
    "ExecutionEvent",
    "ExperimentEvent",
    
    # Event bus utilities
    "get_global_event_bus",
    "set_global_event_bus",
    
    # Configuration utilities
    "get_config_for_environment",
    "get_default_config",
    "validate_environment_name"
]

# Library metadata
__title__ = "normaml"
__description__ = "Modular Machine Learning Library with Zero-Copy Data Processing"
__author__ = "NormaML Team"
__email__ = "team@normaml.org"
__license__ = "GPL-3.0"
__copyright__ = "Copyright 2024 NormaML Team"