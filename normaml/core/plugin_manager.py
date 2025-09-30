"""
Plugin management system for NormaML.

This module implements the PluginManager that handles plugin discovery,
loading, lifecycle management, and dependency resolution for all plugin types.
"""

import os
import sys
import importlib
import importlib.util
import inspect
import logging
from pathlib import Path
from typing import Dict, List, Any, Type, Optional, Set, Union
from collections import defaultdict
import threading

from .interfaces import (
    Plugin, DataLoaderPlugin, FeatureEngineerPlugin, 
    ModelTrainerPlugin, EvaluatorPlugin,
    PluginError, PluginInitializationError, PluginConfigurationError
)


logger = logging.getLogger(__name__)


from .plugin_registry import plugin_registry as canonical_plugin_registry, PluginMetadata, PluginType
class PluginRegistry:
    """
    Adapter wrapper around the canonical PluginRegistry (normaml.core.plugin_registry).

    This thin adapter preserves the smaller API shape expected by the legacy PluginManager
    while delegating canonical metadata storage to the canonical registry. It keeps local
    mappings for runtime plugin instances and types so existing PluginManager logic remains
    compatible.
    """
    def __init__(self):
        self._canonical = canonical_plugin_registry
        self.plugins: Dict[str, Plugin] = {}
        self.plugin_types: Dict[str, Type[Plugin]] = {}
        self.plugin_configs: Dict[str, Dict[str, Any]] = {}
        self.plugin_dependencies: Dict[str, List[str]] = {}
        self.initialization_order: List[str] = []
        self._lock = threading.RLock()

    def register_plugin_type(self, plugin_type: Type[Plugin]) -> None:
        """Register a plugin type class and also register minimal metadata in canonical registry."""
        with self._lock:
            name = getattr(plugin_type, "name", plugin_type.__name__)
            self.plugin_types[name] = plugin_type
            self.plugin_dependencies[name] = list(getattr(plugin_type, "dependencies", []))
            # Build a PluginMetadata for canonical registry (best-effort)
            try:
                ptype = getattr(plugin_type, "plugin_type", None)
                canonical_ptype = ptype if isinstance(ptype, PluginType) else PluginType.DATA_LOADER
            except Exception:
                canonical_ptype = PluginType.DATA_LOADER
            metadata = PluginMetadata(
                name=str(name),
                plugin_type=canonical_ptype,
                cls=plugin_type,
                version=getattr(plugin_type, "version", None),
                capabilities=list(getattr(plugin_type, "capabilities", [])),
                dependencies=list(getattr(plugin_type, "dependencies", [])),
            )
            try:
                # Register metadata into canonical registry for discovery/inspection
                self._canonical.register_plugin(metadata)
            except Exception:
                # Resist failure if canonical registry does not accept metadata
                pass

    def register_plugin_instance(self, plugin: Plugin) -> None:
        """Register a plugin instance (runtime)."""
        with self._lock:
            self.plugins[plugin.name] = plugin
            self.plugin_configs[plugin.name] = {}

    def get_plugin(self, name: str) -> Optional[Plugin]:
        """Get plugin instance by name."""
        with self._lock:
            return self.plugins.get(name)

    def get_plugins_by_type(self, plugin_type: Type[Plugin]) -> List[Plugin]:
        """Get all plugins of a specific runtime type."""
        with self._lock:
            return [p for p in self.plugins.values() if isinstance(p, plugin_type)]

    def list_plugins(self) -> List[str]:
        """List all registered runtime plugin names."""
        with self._lock:
            return list(self.plugins.keys())

    def remove_plugin(self, name: str) -> None:
        """Remove plugin from runtime registry."""
        with self._lock:
            if name in self.plugins:
                del self.plugins[name]
            if name in self.plugin_configs:
                del self.plugin_configs[name]
            if name in self.initialization_order:
                self.initialization_order.remove(name)


class PluginManager:
    """
    Manages plugin discovery, loading, and lifecycle.
    
    The PluginManager is responsible for:
    - Discovering plugins from specified directories
    - Loading and validating plugin classes
    - Managing plugin dependencies
    - Handling plugin initialization and cleanup
    - Providing plugin access to other components
    """
    
    def __init__(self, plugin_dirs: Optional[List[str]] = None, 
                 auto_discover: bool = True):
        """
        Initialize the PluginManager.
        
        Args:
            plugin_dirs: List of directories to search for plugins
            auto_discover: Whether to automatically discover plugins on init
        """
        self.registry = PluginRegistry()
        self.plugin_dirs = plugin_dirs or []
        self.auto_discover = auto_discover
        self.initialized_plugins: Set[str] = set()
        self._shared_memory_config: Dict[str, Any] = {}
        
        # Default plugin directories
        self._setup_default_plugin_dirs()
        
        # Plugin type mapping
        self.plugin_type_mapping = {
            'data_loader': DataLoaderPlugin,
            'feature_engineer': FeatureEngineerPlugin,
            'model_trainer': ModelTrainerPlugin,
            'evaluator': EvaluatorPlugin
        }
        
        if self.auto_discover:
            self.discover_plugins()
        
        logger.info(f"PluginManager initialized with {len(self.plugin_dirs)} plugin directories")
    
    def _setup_default_plugin_dirs(self) -> None:
        """Setup default plugin directories."""
        # Add standard plugin directories
        current_dir = Path(__file__).parent.parent
        default_dirs = [
            str(current_dir / "plugins" / "data_loaders"),
            str(current_dir / "plugins" / "feature_engineers"),
            str(current_dir / "plugins" / "model_trainers"),
            str(current_dir / "plugins" / "evaluators")
        ]
        
        for directory in default_dirs:
            if directory not in self.plugin_dirs:
                self.plugin_dirs.append(directory)
    
    def discover_plugins(self) -> int:
        """
        Discover and load plugins from plugin directories.
        
        Returns:
            Number of plugins discovered
        """
        discovered_count = 0
        
        for plugin_dir in self.plugin_dirs:
            if not os.path.exists(plugin_dir):
                logger.debug(f"Plugin directory does not exist: {plugin_dir}")
                continue
            
            logger.info(f"Discovering plugins in: {plugin_dir}")
            discovered_count += self._discover_from_directory(plugin_dir)
        
        logger.info(f"Discovered {discovered_count} plugins total")
        return discovered_count
    
    def _discover_from_directory(self, directory: str) -> int:
        """
        Discover plugins from a specific directory.
        
        Args:
            directory: Directory path to search
            
        Returns:
            Number of plugins discovered in this directory
        """
        discovered_count = 0
        directory_path = Path(directory)
        
        # Look for Python files
        for py_file in directory_path.glob("*.py"):
            if py_file.name.startswith("__"):
                continue  # Skip __init__.py and __pycache__
            
            try:
                plugins = self._load_plugins_from_file(py_file)
                discovered_count += len(plugins)
                
                for plugin_class in plugins:
                    self.registry.register_plugin_type(plugin_class)
                    logger.debug(f"Registered plugin type: {plugin_class.name}")
                    
            except Exception as e:
                logger.error(f"Error loading plugins from {py_file}: {e}")
        
        # Look for subdirectories (plugin packages)
        for subdir in directory_path.iterdir():
            if subdir.is_dir() and not subdir.name.startswith("__"):
                try:
                    plugins = self._load_plugins_from_package(subdir)
                    discovered_count += len(plugins)
                    
                    for plugin_class in plugins:
                        self.registry.register_plugin_type(plugin_class)
                        logger.debug(f"Registered plugin type: {plugin_class.name}")
                        
                except Exception as e:
                    logger.error(f"Error loading plugins from package {subdir}: {e}")
        
        return discovered_count
    
    def _load_plugins_from_file(self, file_path: Path) -> List[Type[Plugin]]:
        """
        Load plugin classes from a Python file.
        
        Args:
            file_path: Path to the Python file
            
        Returns:
            List of plugin classes found in the file
        """
        plugins = []
        
        # Create module spec and load
        spec = importlib.util.spec_from_file_location(file_path.stem, file_path)
        if spec is None or spec.loader is None:
            return plugins
        
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Find plugin classes
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if (issubclass(obj, Plugin) and 
                obj is not Plugin and 
                hasattr(obj, 'name') and 
                hasattr(obj, 'version')):
                plugins.append(obj)
        
        return plugins
    
    def _load_plugins_from_package(self, package_path: Path) -> List[Type[Plugin]]:
        """
        Load plugin classes from a package directory.
        
        Args:
            package_path: Path to the package directory
            
        Returns:
            List of plugin classes found in the package
        """
        plugins = []
        
        # Check if package has __init__.py
        init_file = package_path / "__init__.py"
        if not init_file.exists():
            return plugins
        
        # Add package to sys.path temporarily
        parent_dir = str(package_path.parent)
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
            path_added = True
        else:
            path_added = False
        
        try:
            # Import the package
            module = importlib.import_module(package_path.name)
            
            # Look for plugin classes in the module
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if (issubclass(obj, Plugin) and 
                    obj is not Plugin and 
                    hasattr(obj, 'name') and 
                    hasattr(obj, 'version')):
                    plugins.append(obj)
            
            # Also check submodules
            for py_file in package_path.glob("*.py"):
                if py_file.name.startswith("__"):
                    continue
                
                submodule_name = f"{package_path.name}.{py_file.stem}"
                try:
                    submodule = importlib.import_module(submodule_name)
                    for name, obj in inspect.getmembers(submodule, inspect.isclass):
                        if (issubclass(obj, Plugin) and 
                            obj is not Plugin and 
                            hasattr(obj, 'name') and 
                            hasattr(obj, 'version')):
                            plugins.append(obj)
                except Exception as e:
                    logger.debug(f"Could not load submodule {submodule_name}: {e}")
        
        finally:
            # Remove from sys.path if we added it
            if path_added:
                sys.path.remove(parent_dir)
        
        return plugins
    
    def load_plugin(self, plugin_name: str, config: Optional[Dict[str, Any]] = None) -> Plugin:
        """
        Load and initialize a specific plugin.
        
        Args:
            plugin_name: Name of the plugin to load
            config: Configuration for the plugin
            
        Returns:
            Initialized plugin instance
            
        Raises:
            PluginError: If plugin cannot be loaded or initialized
        """
        if plugin_name in self.registry.plugins:
            return self.registry.plugins[plugin_name]
        
        if plugin_name not in self.registry.plugin_types:
            raise PluginError(f"Plugin type '{plugin_name}' not found")
        
        plugin_type = self.registry.plugin_types[plugin_name]
        config = config or {}
        
        try:
            # Create instance first to validate configuration
            plugin_instance = plugin_type()
            
            # Validate configuration
            if not plugin_instance.validate_config(config):
                raise PluginConfigurationError(f"Invalid configuration for plugin '{plugin_name}'")
            
            # Setup shared memory if supported
            if plugin_instance.supports_shared_memory and self._shared_memory_config:
                plugin_instance.setup_shared_memory(self._shared_memory_config)
            
            # Initialize plugin
            plugin_instance.initialize(config)
            
            # Register instance
            self.registry.register_plugin_instance(plugin_instance)
            self.registry.plugin_configs[plugin_name] = config
            self.initialized_plugins.add(plugin_name)
            
            logger.info(f"Successfully loaded and initialized plugin: {plugin_name}")
            return plugin_instance
            
        except Exception as e:
            raise PluginInitializationError(f"Failed to initialize plugin '{plugin_name}': {e}") from e
    
    def load_plugins_by_type(self, plugin_type: Union[str, Type[Plugin]],
                           configs: Optional[Dict[str, Dict[str, Any]]] = None) -> List[Plugin]:
        """
        Load all plugins of a specific type.
        
        Args:
            plugin_type: Plugin type class or string identifier
            configs: Optional configurations for each plugin
            
        Returns:
            List of loaded plugin instances
        """
        configs = configs or {}
        plugins = []
        
        # Convert string to class if needed
        target_plugin_type: Type[Plugin]
        if isinstance(plugin_type, str):
            if plugin_type not in self.plugin_type_mapping:
                raise PluginError(f"Unknown plugin type string: {plugin_type}")
            target_plugin_type = self.plugin_type_mapping[plugin_type]
        else:
            target_plugin_type = plugin_type
        
        # Find all plugins of this type
        for name, plugin_class in self.registry.plugin_types.items():
            if issubclass(plugin_class, target_plugin_type):
                try:
                    plugin_config = configs.get(name, {})
                    plugin = self.load_plugin(name, plugin_config)
                    plugins.append(plugin)
                except Exception as e:
                    logger.error(f"Failed to load plugin '{name}': {e}")
        
        return plugins
    
    def get_plugin(self, name: str) -> Optional[Plugin]:
        """
        Get a loaded plugin instance by name.
        
        Args:
            name: Plugin name
            
        Returns:
            Plugin instance or None if not found
        """
        return self.registry.get_plugin(name)
    
    def get_plugins_by_type(self, plugin_type: Type[Plugin]) -> List[Plugin]:
        """
        Get all loaded plugins of a specific type.
        
        Args:
            plugin_type: Plugin type class
            
        Returns:
            List of plugin instances
        """
        return self.registry.get_plugins_by_type(plugin_type)
    
    def list_available_plugins(self) -> Dict[str, Dict[str, Any]]:
        """
        List all available plugin types.
        
        Returns:
            Dictionary with plugin information
        """
        plugins_info = {}
        
        for name, plugin_type in self.registry.plugin_types.items():
            plugins_info[name] = {
                'name': name,
                'version': getattr(plugin_type, 'version', 'unknown'),
                'dependencies': getattr(plugin_type, 'dependencies', []),
                'supports_shared_memory': getattr(plugin_type, 'supports_shared_memory', False),
                'type': plugin_type.__name__,
                'loaded': name in self.registry.plugins
            }
        
        return plugins_info
    
    def set_shared_memory_config(self, config: Dict[str, Any]) -> None:
        """
        Set shared memory configuration for plugins.
        
        Args:
            config: Shared memory configuration
        """
        self._shared_memory_config = config.copy()
        
        # Apply to already loaded plugins
        for plugin in self.registry.plugins.values():
            if plugin.supports_shared_memory:
                try:
                    plugin.setup_shared_memory(self._shared_memory_config)
                except Exception as e:
                    logger.error(f"Failed to setup shared memory for plugin '{plugin.name}': {e}")
    
    def unload_plugin(self, name: str) -> None:
        """
        Unload a plugin and clean up its resources.
        
        Args:
            name: Plugin name to unload
        """
        plugin = self.registry.get_plugin(name)
        if plugin:
            try:
                plugin.cleanup()
                logger.info(f"Cleaned up plugin: {name}")
            except Exception as e:
                logger.error(f"Error during plugin cleanup for '{name}': {e}")
        
        self.registry.remove_plugin(name)
        self.initialized_plugins.discard(name)
    
    def cleanup_all_plugins(self) -> None:
        """Clean up all loaded plugins."""
        logger.info("Cleaning up all plugins")
        
        for name in list(self.registry.plugins.keys()):
            self.unload_plugin(name)
        
        self.initialized_plugins.clear()
        logger.info("All plugins cleaned up")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup_all_plugins()