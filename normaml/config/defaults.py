"""
Default configurations for NormaML.

This module provides default configuration values for all components
of the NormaML framework, organized by environment and use case.
"""

import os
from pathlib import Path
from typing import Dict, Any


# Base default configuration
DEFAULT_CONFIG = {
    "system": {
        "environment": "development",
        "debug": False,
        "temp_dir": None,  # Will use system temp dir if not specified
        "max_memory_usage_mb": 4096,
        "cleanup_on_shutdown": True
    },
    
    "logging": {
        "level": "INFO",
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "file_path": None,  # Log to console by default
        "max_file_size_mb": 10,
        "backup_count": 5
    },
    
    "security": {
        "enable_authentication": False,
        "api_key": None,
        "allowed_hosts": ["localhost", "127.0.0.1"],
        "max_request_size_mb": 100
    },
    
    "data_manager": {
        "shared_memory": {
            "max_memory_mb": 1024,
            "temp_dir": None,
            "cleanup_on_exit": True
        },
        "supported_formats": ["csv", "parquet", "json", "excel"],
        "lazy_loading": True,
        "arrow_optimization": True
    },
    
    "plugin_manager": {
        "plugin_dirs": ["./normaml/plugins"],
        "auto_discover": True,
        "plugins": {},
        "shared_memory": {
            "max_memory_mb": 512,
            "temp_dir": None,
            "cleanup_on_exit": True
        }
    },
    
    "execution_engine": {
        "max_workers": None,  # Will auto-detect based on CPU cores
        "docker": {
            "enabled": True,
            "default_image": "python:3.9-slim",
            "timeout": 3600,
            "shm_size": "2g",
            "remove_containers": True,
            "registry_url": None
        },
        "enable_process_pool": True,
        "enable_thread_pool": True,
        "task_timeout": 1800
    },
    
    "experiment_tracker": {
        "mlflow": {
            "tracking_uri": None,  # Will use local SQLite by default
            "experiment_name": "normaml_experiment",
            "artifact_location": None,
            "auto_create_experiment": True,
            "log_system_info": True
        },
        "enable_shared_memory_tracking": True,
        "auto_log_parameters": True,
        "auto_log_metrics": True,
        "save_artifacts": True
    },
    
    "event_bus": {
        "max_history": 1000,
        "enable_history": True,
        "thread_pool_size": 4,
        "enable_global_bus": True
    }
}


# Development environment overrides
DEVELOPMENT_CONFIG = {
    "system": {
        "environment": "development",
        "debug": True,
        "max_memory_usage_mb": 2048
    },
    
    "logging": {
        "level": "DEBUG",
        "file_path": "./logs/normaml_dev.log"
    },
    
    "data_manager": {
        "shared_memory": {
            "max_memory_mb": 512
        }
    },
    
    "execution_engine": {
        "max_workers": 2,
        "docker": {
            "timeout": 1800,  # Shorter timeout for development
            "remove_containers": True
        },
        "task_timeout": 900
    },
    
    "experiment_tracker": {
        "mlflow": {
            "experiment_name": "normaml_dev",
            "artifact_location": "./artifacts"
        }
    },
    
    "event_bus": {
        "max_history": 500
    }
}


# Testing environment overrides
TESTING_CONFIG = {
    "system": {
        "environment": "testing",
        "debug": True,
        "max_memory_usage_mb": 1024,
        "cleanup_on_shutdown": True
    },
    
    "logging": {
        "level": "WARNING",
        "file_path": "./logs/normaml_test.log"
    },
    
    "data_manager": {
        "shared_memory": {
            "max_memory_mb": 256,
            "cleanup_on_exit": True
        },
        "lazy_loading": False  # Eager loading for faster tests
    },
    
    "plugin_manager": {
        "auto_discover": False,  # Manual plugin loading in tests
        "shared_memory": {
            "max_memory_mb": 128
        }
    },
    
    "execution_engine": {
        "max_workers": 1,
        "docker": {
            "enabled": False,  # Disable Docker in tests by default
            "timeout": 300
        },
        "enable_process_pool": False,  # Simpler execution for tests
        "task_timeout": 300
    },
    
    "experiment_tracker": {
        "mlflow": {
            "experiment_name": "normaml_test",
            "artifact_location": "./test_artifacts",
            "log_system_info": False
        },
        "auto_log_parameters": False,
        "auto_log_metrics": False,
        "save_artifacts": False
    },
    
    "event_bus": {
        "max_history": 100,
        "enable_history": False,  # Disable history in tests
        "thread_pool_size": 1
    }
}


# Production environment overrides
PRODUCTION_CONFIG = {
    "system": {
        "environment": "production",
        "debug": False,
        "max_memory_usage_mb": 8192,
        "cleanup_on_shutdown": True
    },
    
    "logging": {
        "level": "INFO",
        "file_path": "/var/log/normaml/normaml.log",
        "max_file_size_mb": 50,
        "backup_count": 10
    },
    
    "security": {
        "enable_authentication": True,
        "allowed_hosts": ["0.0.0.0"],  # Will be overridden by environment
        "max_request_size_mb": 500
    },
    
    "data_manager": {
        "shared_memory": {
            "max_memory_mb": 4096,
            "cleanup_on_exit": True
        },
        "arrow_optimization": True
    },
    
    "plugin_manager": {
        "plugin_dirs": ["/opt/normaml/plugins", "./normaml/plugins"],
        "shared_memory": {
            "max_memory_mb": 2048
        }
    },
    
    "execution_engine": {
        "max_workers": None,  # Auto-detect based on CPU cores
        "docker": {
            "enabled": True,
            "timeout": 7200,  # Longer timeout for production workloads
            "shm_size": "4g",
            "remove_containers": True
        },
        "task_timeout": 3600
    },
    
    "experiment_tracker": {
        "mlflow": {
            "tracking_uri": "sqlite:///mlflow.db",  # Default to SQLite
            "experiment_name": "normaml_production",
            "artifact_location": "/var/lib/normaml/artifacts",
            "log_system_info": True
        },
        "enable_shared_memory_tracking": True,
        "save_artifacts": True
    },
    
    "event_bus": {
        "max_history": 5000,
        "thread_pool_size": 8
    }
}


# Minimal configuration for resource-constrained environments
MINIMAL_CONFIG = {
    "system": {
        "max_memory_usage_mb": 512
    },
    
    "data_manager": {
        "shared_memory": {
            "max_memory_mb": 128
        },
        "supported_formats": ["csv", "json"],  # Minimal format support
        "arrow_optimization": False
    },
    
    "plugin_manager": {
        "auto_discover": False,
        "shared_memory": {
            "max_memory_mb": 64
        }
    },
    
    "execution_engine": {
        "max_workers": 1,
        "docker": {
            "enabled": False
        },
        "enable_process_pool": False,
        "enable_thread_pool": True
    },
    
    "experiment_tracker": {
        "mlflow": {
            "tracking_uri": None,  # In-memory tracking
            "log_system_info": False
        },
        "auto_log_parameters": False,
        "auto_log_metrics": False,
        "save_artifacts": False
    },
    
    "event_bus": {
        "max_history": 50,
        "enable_history": False,
        "thread_pool_size": 1
    }
}


# High-performance configuration for powerful servers
HIGH_PERFORMANCE_CONFIG = {
    "system": {
        "max_memory_usage_mb": 16384
    },
    
    "data_manager": {
        "shared_memory": {
            "max_memory_mb": 8192
        },
        "arrow_optimization": True
    },
    
    "plugin_manager": {
        "shared_memory": {
            "max_memory_mb": 4096
        }
    },
    
    "execution_engine": {
        "max_workers": None,  # Use all available cores
        "docker": {
            "enabled": True,
            "shm_size": "8g",
            "timeout": 14400  # 4 hours for long-running tasks
        },
        "task_timeout": 7200
    },
    
    "experiment_tracker": {
        "mlflow": {
            "log_system_info": True
        },
        "enable_shared_memory_tracking": True
    },
    
    "event_bus": {
        "max_history": 10000,
        "thread_pool_size": 16
    }
}


def get_config_for_environment(environment: str = "development") -> Dict[str, Any]:
    """
    Get configuration for a specific environment.
    
    Args:
        environment: Environment name (development, testing, production, minimal, high_performance)
        
    Returns:
        Complete configuration dictionary for the environment
    """
    # Start with base config
    config = _deep_copy(DEFAULT_CONFIG)
    
    # Apply environment-specific overrides
    if environment == "development":
        config = _deep_merge(config, DEVELOPMENT_CONFIG)
    elif environment == "testing":
        config = _deep_merge(config, TESTING_CONFIG)
    elif environment == "production":
        config = _deep_merge(config, PRODUCTION_CONFIG)
    elif environment == "minimal":
        config = _deep_merge(config, MINIMAL_CONFIG)
    elif environment == "high_performance":
        config = _deep_merge(config, HIGH_PERFORMANCE_CONFIG)
    
    # Apply auto-detection for certain values
    config = _apply_auto_detection(config)
    
    return config


def get_default_config() -> Dict[str, Any]:
    """
    Get the base default configuration.
    
    Returns:
        Default configuration dictionary
    """
    return _deep_copy(DEFAULT_CONFIG)


def create_config_template(output_path: str = "./config/config.yaml") -> None:
    """
    Create a configuration template file.
    
    Args:
        output_path: Path where to save the template
    """
    import yaml
    
    config = get_config_for_environment("development")
    
    # Create directory if it doesn't exist
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Add comments to the configuration
    commented_config = _add_config_comments(config)
    
    with open(output_path, 'w') as f:
        yaml.dump(commented_config, f, default_flow_style=False, indent=2)
    
    print(f"Configuration template created at: {output_path}")


def validate_environment_name(environment: str) -> bool:
    """
    Validate if environment name is supported.
    
    Args:
        environment: Environment name to validate
        
    Returns:
        True if environment is supported
    """
    valid_environments = {
        "development", "testing", "production", 
        "minimal", "high_performance"
    }
    return environment in valid_environments


def _deep_copy(obj: Any) -> Any:
    """Deep copy a dictionary or other object."""
    if isinstance(obj, dict):
        return {key: _deep_copy(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [_deep_copy(item) for item in obj]
    else:
        return obj


def _deep_merge(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge two dictionaries."""
    result = _deep_copy(base)
    
    for key, value in update.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = _deep_copy(value)
    
    return result


def _apply_auto_detection(config: Dict[str, Any]) -> Dict[str, Any]:
    """Apply auto-detection for configuration values."""
    # Auto-detect number of workers if not specified
    if config["execution_engine"]["max_workers"] is None:
        config["execution_engine"]["max_workers"] = min(32, (os.cpu_count() or 1) + 4)
    
    # Set temp directory if not specified
    if config["system"]["temp_dir"] is None:
        config["system"]["temp_dir"] = str(Path.cwd() / "temp")
    
    if config["data_manager"]["shared_memory"]["temp_dir"] is None:
        config["data_manager"]["shared_memory"]["temp_dir"] = config["system"]["temp_dir"]
    
    # Set MLflow artifact location if not specified
    if config["experiment_tracker"]["mlflow"]["artifact_location"] is None:
        config["experiment_tracker"]["mlflow"]["artifact_location"] = str(Path.cwd() / "mlruns")
    
    return config


def _add_config_comments(config: Dict[str, Any]) -> Dict[str, Any]:
    """Add helpful comments to configuration for template generation."""
    # This is a simplified version - in a real implementation,
    # you might use a YAML library that preserves comments
    comments = {
        "system": "# System-wide configuration",
        "logging": "# Logging configuration", 
        "security": "# Security settings",
        "data_manager": "# Data management configuration",
        "plugin_manager": "# Plugin system configuration",
        "execution_engine": "# Execution engine configuration",
        "experiment_tracker": "# Experiment tracking configuration",
        "event_bus": "# Event bus configuration"
    }
    
    commented_config = {}
    for key, value in config.items():
        if key in comments:
            commented_config[f"_comment_{key}"] = comments[key]
        commented_config[key] = value
    
    return commented_config


# Environment-specific quick access functions
def get_development_config() -> Dict[str, Any]:
    """Get development environment configuration."""
    return get_config_for_environment("development")


def get_testing_config() -> Dict[str, Any]:
    """Get testing environment configuration."""
    return get_config_for_environment("testing")


def get_production_config() -> Dict[str, Any]:
    """Get production environment configuration."""
    return get_config_for_environment("production")


def get_minimal_config() -> Dict[str, Any]:
    """Get minimal resource configuration."""
    return get_config_for_environment("minimal")


def get_high_performance_config() -> Dict[str, Any]:
    """Get high-performance configuration."""
    return get_config_for_environment("high_performance")


# Utility functions for common configuration tasks
def get_docker_config(environment: str = "development") -> Dict[str, Any]:
    """Get Docker configuration for environment."""
    config = get_config_for_environment(environment)
    return config["execution_engine"]["docker"]


def get_shared_memory_config(environment: str = "development") -> Dict[str, Any]:
    """Get shared memory configuration for environment."""
    config = get_config_for_environment(environment)
    return config["data_manager"]["shared_memory"]


def get_mlflow_config(environment: str = "development") -> Dict[str, Any]:
    """Get MLflow configuration for environment."""
    config = get_config_for_environment(environment)
    return config["experiment_tracker"]["mlflow"]