"""
Configuration management for NormaML.

This module implements the ConfigManager that handles loading, validation,
and management of configuration from various sources with support for
environment-specific profiles and hot reloading.
"""

import os
import logging
import threading
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Type, Callable
from dataclasses import dataclass, field
import json

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    yaml = None

try:
    from pydantic import BaseModel, ValidationError
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    BaseModel = object
    ValidationError = Exception

from ..core.event_bus import EventBus, get_global_event_bus


logger = logging.getLogger(__name__)


@dataclass
class ConfigSource:
    """Represents a configuration source."""
    source_type: str  # 'file', 'env', 'dict'
    location: str    # file path, env prefix, or description
    priority: int = 0  # Higher numbers = higher priority
    data: Dict[str, Any] = field(default_factory=dict)
    last_modified: Optional[float] = None
    watch_for_changes: bool = False


class ConfigurationError(Exception):
    """Raised when configuration is invalid or cannot be loaded."""
    pass


class ConfigManager:
    """
    Manages application configuration from multiple sources.
    
    The ConfigManager provides:
    - Loading from YAML, JSON files and environment variables
    - Configuration validation using Pydantic schemas
    - Environment-specific profiles (dev, test, prod)
    - Hierarchical configuration inheritance
    - Hot reloading of configuration files
    - Event notifications for configuration changes
    """
    
    def __init__(self, config_dir: Optional[str] = None, 
                 environment: str = "development",
                 auto_reload: bool = True,
                 event_bus: Optional[EventBus] = None):
        """
        Initialize the ConfigManager.
        
        Args:
            config_dir: Directory containing configuration files
            environment: Environment name (development, test, production)
            auto_reload: Whether to automatically reload changed files
            event_bus: EventBus instance for notifications
        """
        self.config_dir = Path(config_dir) if config_dir else Path.cwd() / "config"
        self.environment = environment
        self.auto_reload = auto_reload
        self.event_bus = event_bus or get_global_event_bus()
        
        # Configuration storage
        self._config_sources: List[ConfigSource] = []
        self._merged_config: Dict[str, Any] = {}
        self._schemas: Dict[str, Any] = {}
        self._validated_configs: Dict[str, Any] = {}
        
        # File watching for hot reload
        self._file_watchers: Dict[str, float] = {}  # file_path -> last_modified
        self._watch_thread: Optional[threading.Thread] = None
        self._stop_watching = threading.Event()
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Configuration change callbacks
        self._change_callbacks: List[Callable[[str, Any], None]] = []
        
        logger.info(f"ConfigManager initialized for environment '{environment}' "
                   f"with config_dir='{self.config_dir}'")
        
        # Load default configurations
        self._load_default_configurations()
        
        # Start file watching if auto_reload is enabled
        if self.auto_reload:
            self._start_file_watching()
    
    def _load_default_configurations(self) -> None:
        """Load default configuration files."""
        try:
            # Load base configuration
            base_config_file = self.config_dir / "config.yaml"
            if base_config_file.exists():
                self.load_from_file(str(base_config_file), watch=self.auto_reload)
            
            # Load environment-specific configuration
            env_config_file = self.config_dir / f"config.{self.environment}.yaml"
            if env_config_file.exists():
                self.load_from_file(str(env_config_file), priority=10, watch=self.auto_reload)
            
            # Load from environment variables
            self.load_from_environment("NORMAML", priority=20)
            
            # Merge all configurations
            self._merge_configurations()
            
        except Exception as e:
            logger.error(f"Failed to load default configurations: {e}")
    
    def load_from_file(self, file_path: str, priority: int = 0, 
                      watch: bool = False) -> None:
        """
        Load configuration from a file.
        
        Args:
            file_path: Path to configuration file (YAML or JSON)
            priority: Priority of this source (higher = more important)
            watch: Whether to watch file for changes
        """
        file_path_obj = Path(file_path)
        
        if not file_path_obj.exists():
            raise ConfigurationError(f"Configuration file not found: {file_path}")
        
        try:
            with open(file_path_obj, 'r', encoding='utf-8') as f:
                if file_path_obj.suffix.lower() in ['.yaml', '.yml']:
                    if not YAML_AVAILABLE or yaml is None:
                        raise ConfigurationError("PyYAML is required for YAML configuration files")
                    data = yaml.safe_load(f) or {}
                elif file_path_obj.suffix.lower() == '.json':
                    data = json.load(f) or {}
                else:
                    raise ConfigurationError(f"Unsupported configuration file format: {file_path_obj.suffix}")
            
            # Get file modification time
            last_modified = file_path_obj.stat().st_mtime
            
            # Create configuration source
            source = ConfigSource(
                source_type='file',
                location=str(file_path),
                priority=priority,
                data=data,
                last_modified=last_modified,
                watch_for_changes=watch
            )
            
            with self._lock:
                # Remove existing source with same location
                self._config_sources = [s for s in self._config_sources 
                                      if s.location != str(file_path)]
                
                # Add new source
                self._config_sources.append(source)
                
                # Add to file watchers if needed
                if watch:
                    self._file_watchers[str(file_path)] = last_modified
                
                # Re-merge configurations
                self._merge_configurations()
            
            logger.info(f"Loaded configuration from {file_path}")
            
            # Notify about configuration change
            self._notify_config_change('file_loaded', {'file_path': str(file_path)})
            
        except Exception as e:
            raise ConfigurationError(f"Failed to load configuration from {file_path}: {e}") from e
    
    def load_from_environment(self, prefix: str = "NORMAML", priority: int = 0) -> None:
        """
        Load configuration from environment variables.
        
        Args:
            prefix: Environment variable prefix
            priority: Priority of this source
        """
        data = {}
        env_prefix = f"{prefix}_"
        
        for key, value in os.environ.items():
            if key.startswith(env_prefix):
                # Convert environment variable to nested dict
                config_key = key[len(env_prefix):].lower()
                config_path = config_key.split('_')
                
                # Parse value (try to convert to appropriate type)
                parsed_value = self._parse_env_value(value)
                
                # Set nested value
                current = data
                for part in config_path[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
                current[config_path[-1]] = parsed_value
        
        if data:
            source = ConfigSource(
                source_type='env',
                location=f"Environment variables with prefix {prefix}",
                priority=priority,
                data=data
            )
            
            with self._lock:
                # Remove existing env source with same prefix
                self._config_sources = [s for s in self._config_sources 
                                      if not (s.source_type == 'env' and prefix in s.location)]
                
                # Add new source
                self._config_sources.append(source)
                
                # Re-merge configurations
                self._merge_configurations()
            
            logger.info(f"Loaded configuration from environment variables (prefix: {prefix})")
            
            # Notify about configuration change
            self._notify_config_change('env_loaded', {'prefix': prefix})
    
    def load_from_dict(self, config_data: Dict[str, Any], 
                      source_name: str = "dict", priority: int = 0) -> None:
        """
        Load configuration from a dictionary.
        
        Args:
            config_data: Configuration data
            source_name: Name for this configuration source
            priority: Priority of this source
        """
        source = ConfigSource(
            source_type='dict',
            location=source_name,
            priority=priority,
            data=config_data.copy()
        )
        
        with self._lock:
            # Remove existing dict source with same name
            self._config_sources = [s for s in self._config_sources 
                                  if not (s.source_type == 'dict' and s.location == source_name)]
            
            # Add new source
            self._config_sources.append(source)
            
            # Re-merge configurations
            self._merge_configurations()
        
        logger.info(f"Loaded configuration from dictionary: {source_name}")
        
        # Notify about configuration change
        self._notify_config_change('dict_loaded', {'source_name': source_name})
    
    def _parse_env_value(self, value: str) -> Any:
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
        
        # Try JSON parsing for complex types
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            pass
        
        # Return as string
        return value
    
    def _merge_configurations(self) -> None:
        """Merge all configuration sources based on priority."""
        # Sort sources by priority (lower numbers first)
        sorted_sources = sorted(self._config_sources, key=lambda x: x.priority)
        
        # Merge configurations
        merged = {}
        for source in sorted_sources:
            merged = self._deep_merge(merged, source.data)
        
        self._merged_config = merged
        
        # Re-validate configurations if schemas are registered
        self._revalidate_all_configs()
    
    def _deep_merge(self, base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries."""
        result = base.copy()
        
        for key, value in update.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def register_schema(self, section: str, schema: Type[BaseModel]) -> None:
        """
        Register a Pydantic schema for configuration validation.
        
        Args:
            section: Configuration section name
            schema: Pydantic model class for validation
        """
        if not PYDANTIC_AVAILABLE:
            logger.warning("Pydantic not available, schema validation disabled")
            return
        
        with self._lock:
            self._schemas[section] = schema
            
            # Validate existing configuration for this section
            if section in self._merged_config:
                self._validate_section(section)
        
        logger.info(f"Registered schema for section: {section}")
    
    def _validate_section(self, section: str) -> None:
        """Validate a configuration section using its registered schema."""
        if section not in self._schemas or not PYDANTIC_AVAILABLE:
            return
        
        schema = self._schemas[section]
        config_data = self._merged_config.get(section, {})
        
        try:
            validated_config = schema(**config_data)
            self._validated_configs[section] = validated_config
            logger.debug(f"Configuration section '{section}' validated successfully")
        except ValidationError as e:
            error_msg = f"Configuration validation failed for section '{section}': {e}"
            logger.error(error_msg)
            raise ConfigurationError(error_msg) from e
    
    def _revalidate_all_configs(self) -> None:
        """Re-validate all configuration sections."""
        for section in self._schemas:
            try:
                self._validate_section(section)
            except ConfigurationError:
                # Log error but don't stop processing other sections
                pass
    
    def get(self, key: str, default: Any = None, section: Optional[str] = None) -> Any:
        """
        Get a configuration value.
        
        Args:
            key: Configuration key (supports dot notation)
            default: Default value if key not found
            section: Optional section name for validated config
            
        Returns:
            Configuration value
        """
        with self._lock:
            # If section is specified and validated config exists, use it
            if section and section in self._validated_configs:
                config_obj = self._validated_configs[section]
                try:
                    return getattr(config_obj, key, default)
                except AttributeError:
                    return default
            
            # Navigate through merged config using dot notation
            config = self._merged_config
            keys = key.split('.')
            
            for k in keys:
                if isinstance(config, dict) and k in config:
                    config = config[k]
                else:
                    return default
            
            return config
    
    def get_section(self, section: str, validated: bool = True) -> Union[Dict[str, Any], BaseModel, None]:
        """
        Get an entire configuration section.
        
        Args:
            section: Section name
            validated: Whether to return validated config object if available
            
        Returns:
            Configuration section data or validated model
        """
        with self._lock:
            # Return validated config if available and requested
            if validated and section in self._validated_configs:
                return self._validated_configs[section]
            
            # Return raw config section
            return self._merged_config.get(section)
    
    def set(self, key: str, value: Any, section: Optional[str] = None) -> None:
        """
        Set a configuration value at runtime.
        
        Args:
            key: Configuration key (supports dot notation)
            value: Value to set
            section: Optional section name
        """
        with self._lock:
            # Navigate through config and set value
            config = self._merged_config
            keys = key.split('.')
            
            # Navigate to parent
            for k in keys[:-1]:
                if k not in config:
                    config[k] = {}
                config = config[k]
            
            # Set value
            config[keys[-1]] = value
            
            # Re-validate section if schema exists
            if section and section in self._schemas:
                try:
                    self._validate_section(section)
                except ConfigurationError as e:
                    # Revert change if validation fails
                    logger.error(f"Failed to set {key}={value}: {e}")
                    raise
            
            logger.debug(f"Set configuration: {key} = {value}")
            
            # Notify about configuration change
            self._notify_config_change('value_changed', {'key': key, 'value': value})
    
    def reload(self) -> None:
        """Reload all file-based configuration sources."""
        with self._lock:
            file_sources = [s for s in self._config_sources if s.source_type == 'file']
            
            for source in file_sources:
                try:
                    self.load_from_file(source.location, source.priority, source.watch_for_changes)
                except Exception as e:
                    logger.error(f"Failed to reload {source.location}: {e}")
        
        logger.info("Configuration reloaded")
        
        # Notify about reload
        self._notify_config_change('reloaded', {})
    
    def _start_file_watching(self) -> None:
        """Start background thread for watching configuration files."""
        if not self._watch_thread or not self._watch_thread.is_alive():
            self._stop_watching.clear()
            self._watch_thread = threading.Thread(target=self._watch_files, daemon=True)
            self._watch_thread.start()
            logger.info("Started configuration file watching")
    
    def _watch_files(self) -> None:
        """Background thread function for watching file changes."""
        while not self._stop_watching.wait(1.0):  # Check every second
            try:
                with self._lock:
                    files_to_check = self._file_watchers.copy()
                
                for file_path, last_known_mtime in files_to_check.items():
                    try:
                        current_mtime = Path(file_path).stat().st_mtime
                        if current_mtime > last_known_mtime:
                            logger.info(f"Detected change in {file_path}, reloading...")
                            
                            # Find the source and reload it
                            source = None
                            with self._lock:
                                for s in self._config_sources:
                                    if s.location == file_path:
                                        source = s
                                        break
                            
                            if source:
                                self.load_from_file(file_path, source.priority, True)
                            
                            # Update watcher
                            with self._lock:
                                self._file_watchers[file_path] = current_mtime
                            
                    except (OSError, IOError):
                        # File might have been deleted or is temporarily unavailable
                        pass
                        
            except Exception as e:
                logger.error(f"Error in file watching: {e}")
    
    def add_change_callback(self, callback: Callable[[str, Any], None]) -> None:
        """
        Add a callback for configuration changes.
        
        Args:
            callback: Function to call when configuration changes
        """
        with self._lock:
            self._change_callbacks.append(callback)
    
    def remove_change_callback(self, callback: Callable[[str, Any], None]) -> None:
        """
        Remove a configuration change callback.
        
        Args:
            callback: Callback function to remove
        """
        with self._lock:
            if callback in self._change_callbacks:
                self._change_callbacks.remove(callback)
    
    def _notify_config_change(self, change_type: str, details: Dict[str, Any]) -> None:
        """Notify about configuration changes."""
        # Call registered callbacks
        for callback in self._change_callbacks:
            try:
                callback(change_type, details)
            except Exception as e:
                logger.error(f"Error in configuration change callback: {e}")
        
        # Publish event
        try:
            self.event_bus.publish_system_event(
                event_type=f"config_{change_type}",
                source="ConfigManager",
                data=details
            )
        except Exception as e:
            logger.error(f"Failed to publish configuration change event: {e}")
    
    def get_config_summary(self) -> Dict[str, Any]:
        """
        Get a summary of current configuration.
        
        Returns:
            Configuration summary
        """
        with self._lock:
            return {
                'environment': self.environment,
                'config_dir': str(self.config_dir),
                'sources': [
                    {
                        'type': s.source_type,
                        'location': s.location,
                        'priority': s.priority,
                        'last_modified': s.last_modified
                    }
                    for s in sorted(self._config_sources, key=lambda x: x.priority)
                ],
                'registered_schemas': list(self._schemas.keys()),
                'validated_sections': list(self._validated_configs.keys()),
                'auto_reload': self.auto_reload,
                'watched_files': list(self._file_watchers.keys())
            }
    
    def cleanup(self) -> None:
        """Clean up configuration manager resources."""
        logger.info("Cleaning up ConfigManager")
        
        # Stop file watching
        if self._watch_thread and self._watch_thread.is_alive():
            self._stop_watching.set()
            self._watch_thread.join(timeout=5)
        
        with self._lock:
            # Clear all data
            self._config_sources.clear()
            self._merged_config.clear()
            self._schemas.clear()
            self._validated_configs.clear()
            self._file_watchers.clear()
            self._change_callbacks.clear()
        
        logger.info("ConfigManager cleanup completed")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup()
    
    def __del__(self):
        """Destructor with cleanup."""
        try:
            self.cleanup()
        except Exception:
            pass  # Ignore cleanup errors during destruction