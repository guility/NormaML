"""
Configuration management module for NormaML.

This module provides configuration management capabilities including:
- Configuration loading from multiple sources
- Validation using Pydantic schemas
- Environment-specific configurations
- Hot reloading support
"""

from .manager import ConfigManager, ConfigSource, ConfigurationError
from .schemas import (
    # Configuration schemas
    NormaMLConfig,
    SystemConfig,
    LoggingConfig,
    SecurityConfig,
    DataManagerConfig,
    SharedMemoryConfig,
    PluginManagerConfig,
    PluginConfig,
    ExecutionEngineConfig,
    DockerConfig,
    ExperimentTrackerConfig,
    MLflowConfig,
    EventBusConfig,
    APIConfig,
    
    # Validation functions
    validate_config,
    get_default_config as get_schema_defaults,
    create_config_from_env
)

from .defaults import (
    # Default configurations
    DEFAULT_CONFIG,
    DEVELOPMENT_CONFIG,
    TESTING_CONFIG,
    PRODUCTION_CONFIG,
    MINIMAL_CONFIG,
    HIGH_PERFORMANCE_CONFIG,
    
    # Utility functions
    get_config_for_environment,
    get_default_config,
    create_config_template,
    validate_environment_name,
    
    # Environment-specific getters
    get_development_config,
    get_testing_config,
    get_production_config,
    get_minimal_config,
    get_high_performance_config,
    
    # Component-specific getters
    get_docker_config,
    get_shared_memory_config,
    get_mlflow_config
)

__all__ = [
    # Core configuration management
    "ConfigManager",
    "ConfigSource", 
    "ConfigurationError",
    
    # Pydantic schemas
    "NormaMLConfig",
    "SystemConfig",
    "LoggingConfig",
    "SecurityConfig",
    "DataManagerConfig",
    "SharedMemoryConfig",
    "PluginManagerConfig",
    "PluginConfig",
    "ExecutionEngineConfig",
    "DockerConfig",
    "ExperimentTrackerConfig",
    "MLflowConfig",
    "EventBusConfig",
    "APIConfig",
    
    # Schema validation
    "validate_config",
    "get_schema_defaults",
    "create_config_from_env",
    
    # Default configurations
    "DEFAULT_CONFIG",
    "DEVELOPMENT_CONFIG",
    "TESTING_CONFIG",
    "PRODUCTION_CONFIG",
    "MINIMAL_CONFIG",
    "HIGH_PERFORMANCE_CONFIG",
    
    # Configuration utilities
    "get_config_for_environment",
    "get_default_config",
    "create_config_template",
    "validate_environment_name",
    
    # Environment-specific configurations
    "get_development_config",
    "get_testing_config",
    "get_production_config",
    "get_minimal_config",
    "get_high_performance_config",
    
    # Component-specific configurations
    "get_docker_config",
    "get_shared_memory_config",
    "get_mlflow_config"
]