from __future__ import annotations
"""
Plugin lifecycle manager.

Provides:
- initialize_plugin
- execute_plugin (supports thread / process isolation; docker not implemented)
- get_plugin_instance, cleanup_plugin, cleanup_all
"""
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, TimeoutError as FutureTimeout
from typing import Any, Dict, Optional
import functools
import logging

from normaml.core import plugin_registry
from normaml.core import interfaces

logger = logging.getLogger(__name__)


def _process_runner(plugin_cls, config: Dict[str, Any], method_name: str, args, kwargs):
    """
    Helper executed inside a ProcessPoolExecutor. Recreate plugin instance,
    initialize it with config and call requested method.

    This lives at module top-level so it can be pickled for ProcessPoolExecutor.
    """
    inst = plugin_cls()
    if hasattr(inst, "initialize"):
        inst.initialize(config or {})
    fn = getattr(inst, method_name)
    return fn(*args, **kwargs)


class PluginLifecycleManager:
    """
    Manage plugin instances and execution isolation.

    Usage:
        mgr = PluginLifecycleManager()
        inst = mgr.initialize_plugin("my_loader", config={})
        res = mgr.execute_plugin("my_loader", "load", "/path", isolation="thread")
    """

    def __init__(self) -> None:
        self._instances: Dict[str, interfaces.Plugin] = {}
        self._configs: Dict[str, Dict] = {}
        self._thread_pool = ThreadPoolExecutor(max_workers=8)
        self._process_pool = ProcessPoolExecutor(max_workers=2)

    def initialize_plugin(self, plugin_name: str, config: Optional[Dict] = None) -> interfaces.Plugin:
        """
        Initialize a plugin by name and store its instance.

        Args:
            plugin_name: Registered plugin name.
            config: Optional configuration dict passed to plugin.initialize.

        Returns:
            Initialized plugin instance.

        Raises:
            KeyError: If plugin not registered.
            interfaces.PluginConfigurationError: If validation fails.
        """
        meta = plugin_registry.plugin_registry.get_plugin_metadata(plugin_name)
        cls = meta.cls
        inst: interfaces.Plugin = cls()
        cfg = config or {}
        # validate
        if hasattr(inst, "validate_config"):
            valid = inst.validate_config(cfg)
            if not valid:
                raise interfaces.PluginConfigurationError(f"Invalid configuration for plugin {plugin_name}")
        # initialize
        inst.initialize(cfg)
        self._instances[plugin_name] = inst
        self._configs[plugin_name] = cfg
        return inst

    def execute_plugin(self, plugin_name: str, method: str, *args, isolation: str = "thread", timeout: Optional[float] = None, **kwargs) -> Any:
        """
        Execute a plugin method with optional isolation.

        Args:
            plugin_name: Registered plugin name (must be initialized or will be initialized on-the-fly).
            method: Method name to call on the plugin instance.
            *args, **kwargs: Passed to the method.
            isolation: "thread" | "process" | "docker" (docker not supported -> NotImplementedError)
            timeout: Optional timeout in seconds.

        Returns:
            Result of the method call.

        Raises:
            NotImplementedError: If docker isolation requested.
            KeyError: If plugin not registered.
        """
        if isolation not in ("thread", "process", "docker"):
            raise ValueError("isolation must be 'thread', 'process' or 'docker'")

        if isolation == "docker":
            raise NotImplementedError("Docker isolation is not implemented")

        # Ensure plugin is initialized
        if plugin_name not in self._instances:
            self.initialize_plugin(plugin_name, config=self._configs.get(plugin_name, {}))

        inst = self._instances[plugin_name]
        if not hasattr(inst, method):
            raise AttributeError(f"Plugin '{plugin_name}' has no method '{method}'")

        fn = getattr(inst, method)

        if isolation == "thread":
            future = self._thread_pool.submit(fn, *args, **kwargs)
            try:
                return future.result(timeout=timeout)
            except FutureTimeout:
                future.cancel()
                raise
        else:  # process
            # Use helper to recreate instance in process. Pass class and config.
            meta = plugin_registry.plugin_registry.get_plugin_metadata(plugin_name)
            plugin_cls = meta.cls
            cfg = self._configs.get(plugin_name, {})
            future = self._process_pool.submit(_process_runner, plugin_cls, cfg, method, args, kwargs)
            try:
                return future.result(timeout=timeout)
            except FutureTimeout:
                future.cancel()
                raise

    def get_plugin_instance(self, plugin_name: str) -> interfaces.Plugin:
        """
        Return the in-memory plugin instance.

        Raises KeyError if not initialized.
        """
        if plugin_name not in self._instances:
            raise KeyError(f"Plugin '{plugin_name}' not initialized")
        return self._instances[plugin_name]

    def cleanup_plugin(self, plugin_name: str) -> None:
        """
        Cleanup a single plugin instance (call its cleanup and remove it).
        """
        inst = self._instances.get(plugin_name)
        if inst:
            try:
                inst.cleanup()
            except Exception:
                logger.exception("Error while cleaning up plugin %s", plugin_name)
            finally:
                del self._instances[plugin_name]
                if plugin_name in self._configs:
                    del self._configs[plugin_name]

    def cleanup_all(self) -> None:
        """
        Cleanup all plugin instances and shutdown executors.
        """
        names = list(self._instances.keys())
        for n in names:
            self.cleanup_plugin(n)
        try:
            self._thread_pool.shutdown(wait=False)
        except Exception:
            pass
        try:
            self._process_pool.shutdown(wait=False)
        except Exception:
            pass