"""
Pydantic schemas for NormaML configuration validation.

This module defines validation schemas for all configuration sections
using Pydantic models to ensure type safety and proper validation.
"""

from typing import Dict, List, Optional, Union, Any
from pathlib import Path

try:
    from pydantic import BaseModel, Field, validator, root_validator
    PYDANTIC_AVAILABLE = True
except ImportError:
    # Fallback base class when Pydantic is not available
    class BaseModel:
        pass
    
    def Field(*args, **kwargs):
        return None
    
    def validator(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    
    def root_validator(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    
    PYDANTIC_AVAILABLE = False


class SharedMemoryConfig(BaseModel):
    """Configuration schema for shared memory settings."""
    
    max_memory_mb: int = Field(default=1024, ge=1, le=32768, 
                              description="Maximum shared memory in MB")
    temp_dir: Optional[str] = Field(default=None, 
                                   description="Temporary directory for memory-mapped files")
    cleanup_on_exit: bool = Field(default=True, 
                                 description="Whether to cleanup shared memory on exit")
    
    if PYDANTIC_AVAILABLE:
        @validator('temp_dir')
        def validate_temp_dir(cls, v):
            if v is not None:
                path = Path(v)
                if not path.parent.exists():
                    raise ValueError(f"Parent directory does not exist: {path.parent}")
            return v


class DataManagerConfig(BaseModel):
    """Configuration schema for DataManager."""
    
    shared_memory: SharedMemoryConfig = Field(default_factory=SharedMemoryConfig)
    supported_formats: List[str] = Field(default=['csv', 'parquet', 'json', 'excel'], 
                                        description="Supported data formats")
    lazy_loading: bool = Field(default=True, 
                              description="Enable lazy loading for large datasets")
    arrow_optimization: bool = Field(default=True, 
                                   description="Use Apache Arrow for optimization")
    
    if PYDANTIC_AVAILABLE:
        @validator('supported_formats')
        def validate_formats(cls, v):
            valid_formats = {'csv', 'parquet', 'json', 'excel', 'yaml', 'feather', 'orc'}
            for fmt in v:
                if fmt not in valid_formats:
                    raise ValueError(f"Unsupported format: {fmt}")
            return v


class PluginConfig(BaseModel):
    """Configuration schema for individual plugins."""
    
    enabled: bool = Field(default=True, description="Whether plugin is enabled")
    priority: int = Field(default=0, description="Plugin loading priority")
    config: Dict[str, Any] = Field(default_factory=dict, 
                                  description="Plugin-specific configuration")
    dependencies: List[str] = Field(default_factory=list, 
                                   description="Plugin dependencies")
    supports_shared_memory: bool = Field(default=True, 
                                        description="Whether plugin supports shared memory")


class PluginManagerConfig(BaseModel):
    """Configuration schema for PluginManager."""
    
    plugin_dirs: List[str] = Field(default_factory=lambda: ["./normaml/plugins"], 
                                  description="Directories to search for plugins")
    auto_discover: bool = Field(default=True, 
                               description="Automatically discover plugins on startup")
    plugins: Dict[str, PluginConfig] = Field(default_factory=dict, 
                                            description="Individual plugin configurations")
    shared_memory: SharedMemoryConfig = Field(default_factory=SharedMemoryConfig)
    
    if PYDANTIC_AVAILABLE:
        @validator('plugin_dirs')
        def validate_plugin_dirs(cls, v):
            for directory in v:
                if not Path(directory).exists():
                    raise ValueError(f"Plugin directory does not exist: {directory}")
            return v


class DockerConfig(BaseModel):
    """Configuration schema for Docker execution."""
    
    enabled: bool = Field(default=True, description="Enable Docker support")
    default_image: str = Field(default="python:3.9-slim", 
                              description="Default Docker image")
    timeout: int = Field(default=3600, ge=1, 
                        description="Default timeout for containers in seconds")
    shm_size: str = Field(default="2g", 
                         description="Shared memory size for containers")
    remove_containers: bool = Field(default=True, 
                                   description="Remove containers after execution")
    registry_url: Optional[str] = Field(default=None, 
                                       description="Docker registry URL")
    
    if PYDANTIC_AVAILABLE:
        @validator('shm_size')
        def validate_shm_size(cls, v):
            if not v.endswith(('k', 'm', 'g', 'K', 'M', 'G')):
                raise ValueError("shm_size must end with k, m, or g (e.g., '2g', '512m')")
            return v


class ExecutionEngineConfig(BaseModel):
    """Configuration schema for ExecutionEngine."""
    
    max_workers: Optional[int] = Field(default=None, ge=1, 
                                      description="Maximum number of worker threads/processes")
    docker: DockerConfig = Field(default_factory=DockerConfig)
    enable_process_pool: bool = Field(default=True, 
                                     description="Enable process pool for CPU-bound tasks")
    enable_thread_pool: bool = Field(default=True, 
                                    description="Enable thread pool for I/O-bound tasks")
    task_timeout: int = Field(default=1800, ge=1, 
                             description="Default task timeout in seconds")
    
    if PYDANTIC_AVAILABLE:
        @validator('max_workers')
        def validate_max_workers(cls, v):
            if v is not None and v > 64:
                raise ValueError("max_workers should not exceed 64 for performance reasons")
            return v


class MLflowConfig(BaseModel):
    """Configuration schema for MLflow integration."""
    
    tracking_uri: Optional[str] = Field(default=None, 
                                       description="MLflow tracking server URI")
    experiment_name: str = Field(default="normaml_experiment", 
                                description="Default experiment name")
    artifact_location: Optional[str] = Field(default=None, 
                                            description="Artifact storage location")
    auto_create_experiment: bool = Field(default=True, 
                                        description="Auto-create experiment if not exists")
    log_system_info: bool = Field(default=True, 
                                 description="Log system information with runs")
    
    if PYDANTIC_AVAILABLE:
        @validator('tracking_uri')
        def validate_tracking_uri(cls, v):
            if v is not None:
                if not (v.startswith('http://') or v.startswith('https://') or 
                       v.startswith('file://') or v.startswith('sqlite://')):
                    raise ValueError("tracking_uri must be a valid URI")
            return v


class ExperimentTrackerConfig(BaseModel):
    """Configuration schema for ExperimentTracker."""
    
    mlflow: MLflowConfig = Field(default_factory=MLflowConfig)
    enable_shared_memory_tracking: bool = Field(default=True, 
                                               description="Track shared memory usage")
    auto_log_parameters: bool = Field(default=True, 
                                     description="Automatically log model parameters")
    auto_log_metrics: bool = Field(default=True, 
                                  description="Automatically log metrics")
    save_artifacts: bool = Field(default=True, 
                                description="Save model artifacts")


class EventBusConfig(BaseModel):
    """Configuration schema for EventBus."""
    
    max_history: int = Field(default=1000, ge=0, 
                            description="Maximum events to keep in history")
    enable_history: bool = Field(default=True, 
                                description="Enable event history")
    thread_pool_size: int = Field(default=4, ge=1, le=32, 
                                 description="Thread pool size for async processing")
    enable_global_bus: bool = Field(default=True, 
                                   description="Enable global event bus")
    
    if PYDANTIC_AVAILABLE:
        @validator('max_history')
        def validate_max_history(cls, v):
            if v > 10000:
                raise ValueError("max_history should not exceed 10000 for memory efficiency")
            return v


class LoggingConfig(BaseModel):
    """Configuration schema for logging."""
    
    level: str = Field(default="INFO", 
                      description="Logging level")
    format: str = Field(default="%(asctime)s - %(name)s - %(levelname)s - %(message)s", 
                       description="Log message format")
    file_path: Optional[str] = Field(default=None, 
                                    description="Log file path")
    max_file_size_mb: int = Field(default=10, ge=1, 
                                 description="Maximum log file size in MB")
    backup_count: int = Field(default=5, ge=0, 
                             description="Number of backup log files")
    
    if PYDANTIC_AVAILABLE:
        @validator('level')
        def validate_level(cls, v):
            valid_levels = {'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'}
            if v.upper() not in valid_levels:
                raise ValueError(f"level must be one of: {valid_levels}")
            return v.upper()


class SecurityConfig(BaseModel):
    """Configuration schema for security settings."""
    
    enable_authentication: bool = Field(default=False, 
                                       description="Enable authentication")
    api_key: Optional[str] = Field(default=None, 
                                  description="API key for authentication")
    allowed_hosts: List[str] = Field(default_factory=lambda: ["localhost", "127.0.0.1"], 
                                    description="Allowed hosts for API access")
    max_request_size_mb: int = Field(default=100, ge=1, 
                                    description="Maximum request size in MB")
    
    if PYDANTIC_AVAILABLE:
        @validator('api_key')
        def validate_api_key(cls, v):
            if v is not None and len(v) < 32:
                raise ValueError("api_key must be at least 32 characters long")
            return v


class SystemConfig(BaseModel):
    """Configuration schema for system-wide settings."""
    
    environment: str = Field(default="development", 
                            description="Environment name")
    debug: bool = Field(default=False, 
                       description="Enable debug mode")
    temp_dir: Optional[str] = Field(default=None, 
                                   description="System temporary directory")
    max_memory_usage_mb: int = Field(default=4096, ge=512, 
                                    description="Maximum system memory usage in MB")
    cleanup_on_shutdown: bool = Field(default=True, 
                                     description="Cleanup resources on shutdown")
    
    if PYDANTIC_AVAILABLE:
        @validator('environment')
        def validate_environment(cls, v):
            valid_envs = {'development', 'testing', 'staging', 'production'}
            if v not in valid_envs:
                raise ValueError(f"environment must be one of: {valid_envs}")
            return v


class NormaMLConfig(BaseModel):
    """Root configuration schema for NormaML."""
    
    system: SystemConfig = Field(default_factory=SystemConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    
    data_manager: DataManagerConfig = Field(default_factory=DataManagerConfig)
    plugin_manager: PluginManagerConfig = Field(default_factory=PluginManagerConfig)
    execution_engine: ExecutionEngineConfig = Field(default_factory=ExecutionEngineConfig)
    experiment_tracker: ExperimentTrackerConfig = Field(default_factory=ExperimentTrackerConfig)
    event_bus: EventBusConfig = Field(default_factory=EventBusConfig)
    
    if PYDANTIC_AVAILABLE:
        @root_validator(skip_on_failure=True)
        def validate_config_consistency(cls, values):
            """Validate configuration consistency across components."""
            
            # Ensure shared memory settings are consistent
            system_temp_dir = values.get('system', {}).temp_dir
            data_temp_dir = values.get('data_manager', {}).shared_memory.temp_dir
            
            if system_temp_dir and data_temp_dir and system_temp_dir != data_temp_dir:
                raise ValueError("System and data manager temp directories must match if both are specified")
            
            # Validate Docker settings if enabled
            execution_config = values.get('execution_engine', {})
            if hasattr(execution_config, 'docker') and execution_config.docker.enabled:
                if not execution_config.docker.default_image:
                    raise ValueError("default_image must be specified when Docker is enabled")
            
            return values


class APIConfig(BaseModel):
    """Configuration schema for API server (if implemented)."""
    
    host: str = Field(default="localhost", description="API server host")
    port: int = Field(default=8000, ge=1, le=65535, description="API server port")
    workers: int = Field(default=1, ge=1, description="Number of API workers")
    enable_docs: bool = Field(default=True, description="Enable API documentation")
    cors_origins: List[str] = Field(default_factory=lambda: ["*"], 
                                   description="CORS allowed origins")
    
    if PYDANTIC_AVAILABLE:
        @validator('host')
        def validate_host(cls, v):
            if v not in ['localhost', '127.0.0.1', '0.0.0.0'] and not v.startswith('http'):
                raise ValueError("host must be localhost, 127.0.0.1, 0.0.0.0, or a valid URL")
            return v


# Convenience functions for validation
def validate_config(config_dict: Dict[str, Any], schema_class: BaseModel = NormaMLConfig) -> BaseModel:
    """
    Validate configuration dictionary against schema.
    
    Args:
        config_dict: Configuration data to validate
        schema_class: Pydantic model class to use for validation
        
    Returns:
        Validated configuration object
        
    Raises:
        ValidationError: If configuration is invalid
    """
    if not PYDANTIC_AVAILABLE:
        raise ImportError("Pydantic is required for configuration validation")
    
    return schema_class(**config_dict)


def get_default_config() -> Dict[str, Any]:
    """
    Get default configuration as dictionary.
    
    Returns:
        Default configuration dictionary
    """
    if PYDANTIC_AVAILABLE:
        return NormaMLConfig().dict()
    else:
        # Return basic default configuration when Pydantic is not available
        return {
            'system': {
                'environment': 'development',
                'debug': False,
                'max_memory_usage_mb': 4096
            },
            'logging': {
                'level': 'INFO',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            },
            'data_manager': {
                'shared_memory': {
                    'max_memory_mb': 1024
                },
                'supported_formats': ['csv', 'parquet', 'json']
            },
            'plugin_manager': {
                'plugin_dirs': ['./normaml/plugins'],
                'auto_discover': True
            },
            'execution_engine': {
                'docker': {
                    'enabled': True,
                    'default_image': 'python:3.9-slim'
                }
            },
            'experiment_tracker': {
                'mlflow': {
                    'experiment_name': 'normaml_experiment'
                }
            },
            'event_bus': {
                'max_history': 1000,
                'thread_pool_size': 4
            }
        }


def create_config_from_env(prefix: str = "NORMAML") -> Dict[str, Any]:
    """
    Create configuration from environment variables.
    
    Args:
        prefix: Environment variable prefix
        
    Returns:
        Configuration dictionary created from environment variables
    """
    import os
    
    config = {}
    env_prefix = f"{prefix}_"
    
    for key, value in os.environ.items():
        if key.startswith(env_prefix):
            # Convert environment variable to nested dict path
            config_path = key[len(env_prefix):].lower().split('_')
            
            # Parse value
            parsed_value = _parse_env_value(value)
            
            # Set nested value
            current = config
            for part in config_path[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            current[config_path[-1]] = parsed_value
    
    return config


def _parse_env_value(value: str) -> Union[str, int, float, bool]:
    """Parse environment variable value to appropriate type."""
    # Try boolean
    if value.lower() in ('true', 'false'):
        return value.lower() == 'true'
    
    # Try integer
    try:
        return int(value)
    except ValueError:
        pass
    
    # Try float
    try:
        return float(value)
    except ValueError:
        pass
    
    # Return as string
    return value