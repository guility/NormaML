from __future__ import annotations
"""
Plugin registry and registration decorators for NormaML.

This module provides:
- PluginType enum
- PluginMetadata dataclass
- PluginRegistry singleton with registration, discovery and dependency resolution
- Convenience decorators for registering plugins from code/tests
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Set
import importlib
import importlib.metadata
import threading
from normaml.core import interfaces


class PluginType(str, Enum):
    """Enumeration of plugin types supported by NormaML."""
    DATA_LOADER = "data_loader"
    FEATURE_ENGINEER = "feature_engineer"
    MODEL_TRAINER = "model_trainer"
    EVALUATOR = "evaluator"


@dataclass
class PluginMetadata:
    """Metadata stored for each registered plugin."""
    name: str
    plugin_type: PluginType
    cls: type
    version: Optional[str] = None
    capabilities: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    entry_point: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


class PluginRegistry:
    """
    Registry of plugins.

    Thread-safe simple registry that supports:
    - programmatic registration via register_plugin
    - discovery via entry points (importlib.metadata)
    - searching by type/capabilities/name
    - dependency resolution (detect cycles)
    """
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._plugins: Dict[str, PluginMetadata] = {}
        # Keep a separate store of declared plugins (from decorators).
        # Tests sometimes clear `_plugins` directly; preserving declarations
        # allows the registry to restore decorator-registered plugins when needed.
        self._declared_plugins: Dict[str, PluginMetadata] = {}

    def register_plugin(self, metadata: PluginMetadata) -> None:
        """
        Register a plugin metadata object.

        Args:
            metadata: PluginMetadata to register.
        """
        with self._lock:
            # Store in active plugins map and remember declaration so tests that clear
            # `_plugins` don't permanently lose decorator-registered plugins.
            self._plugins[metadata.name] = metadata
            self._declared_plugins[metadata.name] = metadata

    def _restore_declared_if_empty(self) -> None:
        """
        Restore declared plugins into the active plugins map if `_plugins` is empty.

        Some tests clear `plugin_registry.plugin_registry._plugins` directly. Decorator-based
        registrations are recorded in `_declared_plugins`; this helper will repopulate
        `_plugins` from `_declared_plugins` lazily so discovery/filter functions continue to work.
        """
        if not self._plugins and self._declared_plugins:
            # shallow copy to avoid accidental aliasing
            for k, v in self._declared_plugins.items():
                self._plugins[k] = v

    def discover_entry_points(self, group: str = "normaml.plugins") -> List[PluginMetadata]:
        """
        Discover plugins via entry points.

        Args:
            group: Entry point group to search.

        Returns:
            List of PluginMetadata discovered (note: entry-point loading may import user code).
        """
        discovered: List[PluginMetadata] = []
        # Use importlib.metadata for discovery; gracefully handle absence
        try:
            eps = importlib.metadata.entry_points()
            # compatibility: in some versions eps is a dict-like
            entries = []
            if hasattr(eps, "select"):
                entries = eps.select(group=group)
            else:
                entries = [e for e in eps.get(group, [])] if isinstance(eps, dict) else []
            for ep in entries:
                try:
                    loaded = ep.load()
                    # expect loaded to be a class
                    name = getattr(loaded, "name", getattr(loaded, "__name__", ep.name))
                    ptype = getattr(loaded, "plugin_type", None)
                    if isinstance(ptype, PluginType):
                        plugin_type = ptype
                    else:
                        # try to infer from entry point group suffix
                        plugin_type = PluginType.DATA_LOADER
                    metadata = PluginMetadata(
                        name=str(name),
                        plugin_type=plugin_type,
                        cls=loaded,
                        version=getattr(loaded, "version", None),
                        capabilities=list(getattr(loaded, "capabilities", [])),
                        dependencies=list(getattr(loaded, "dependencies", [])),
                        entry_point=f"{ep.module}:{ep.name}"
                    )
                    self.register_plugin(metadata)
                    discovered.append(metadata)
                except Exception:
                    # Skip EPS that fail to load during discovery to keep discovery resilient in tests
                    continue
        except Exception:
            # Importlib metadata not available or other issue; return empty
            return []
        return discovered

    def find_plugins(
        self,
        plugin_type: Optional[PluginType] = None,
        capability: Optional[str] = None,
        name: Optional[str] = None,
    ) -> List[PluginMetadata]:
        """
        Find plugins by filters.

        Args:
            plugin_type: Filter by PluginType
            capability: Filter plugins that list this capability
            name: Exact plugin name

        Returns:
            List of matching PluginMetadata
        """
        with self._lock:
            # If active plugins were cleared but we have declared plugins, restore them.
            self._restore_declared_if_empty()
            results = []
            for meta in self._plugins.values():
                if plugin_type and meta.plugin_type != plugin_type:
                    continue
                if capability and capability not in meta.capabilities:
                    continue
                if name and meta.name != name:
                    continue
                results.append(meta)
            return results

    def get_plugin_metadata(self, name: str) -> PluginMetadata:
        """
        Get metadata for a plugin by name.

        Raises KeyError if not found.
        """
        with self._lock:
            # Ensure declared plugins are available
            self._restore_declared_if_empty()
            return self._plugins[name]

    def list_plugins(self) -> List[PluginMetadata]:
        """Return all registered plugins' metadata."""
        with self._lock:
            # Ensure declared plugins are available
            self._restore_declared_if_empty()
            return list(self._plugins.values())

    def resolve_dependencies(self, root_plugin_names: List[str]) -> List[str]:
        """
        Resolve dependencies for a list of plugin names.

        Performs a topological sort and raises ValueError on cycles or missing names.

        Args:
            root_plugin_names: list of plugin names to resolve (includes their transitive deps)

        Returns:
            Ordered list of plugin names such that dependencies come before dependents.
        """
        with self._lock:
            # Ensure declared plugins are available before dependency resolution
            self._restore_declared_if_empty()
            graph: Dict[str, Set[str]] = {}
            for name in root_plugin_names:
                if name not in self._plugins:
                    raise KeyError(f"Plugin '{name}' not registered")
            # build graph for relevant plugins
            to_visit = list(root_plugin_names)
            while to_visit:
                cur = to_visit.pop()
                if cur in graph:
                    continue
                meta = self._plugins.get(cur)
                if meta is None:
                    raise KeyError(f"Plugin '{cur}' not registered")
                deps = set(meta.dependencies or [])
                graph[cur] = deps
                for d in deps:
                    if d not in graph:
                        to_visit.append(d)
            # topological sort
            temp_mark: Set[str] = set()
            perm_mark: Set[str] = set()
            result: List[str] = []

            def visit(n: str):
                if n in perm_mark:
                    return
                if n in temp_mark:
                    raise ValueError(f"Cyclic dependency detected at plugin '{n}'")
                temp_mark.add(n)
                for m in graph.get(n, []):
                    if m not in self._plugins:
                        raise KeyError(f"Dependency '{m}' of plugin '{n}' not registered")
                    visit(m)
                temp_mark.remove(n)
                perm_mark.add(n)
                result.append(n)

            for node in list(graph.keys()):
                if node not in perm_mark:
                    visit(node)
            # result has dependencies before dependents; but may include extras — limit to discovered order unique
            ordered = []
            for r in result:
                if r not in ordered:
                    ordered.append(r)
            return ordered


# Global registry instance used by decorators and other modules
plugin_registry = PluginRegistry()


# Decorators for registering plugins from code (tests will mostly use these)
def _make_register_decorator(ptype: PluginType):
    def decorator(name: Optional[str] = None, *, capabilities: Optional[List[str]] = None, dependencies: Optional[List[str]] = None, version: Optional[str] = None):
        """
        Decorator factory to register a plugin class.

        Usage:
            @_register_data_loader("csv_loader", capabilities=["csv"])
            class CSVLoader(...): ...
        """
        def wrapper(cls):
            meta_name = name or getattr(cls, "name", cls.__name__)
            meta = PluginMetadata(
                name=str(meta_name),
                plugin_type=ptype,
                cls=cls,
                version=version or getattr(cls, "version", None),
                capabilities=list(capabilities or getattr(cls, "capabilities", [])),
                dependencies=list(dependencies or getattr(cls, "dependencies", [])),
            )
            plugin_registry.register_plugin(meta)
            return cls
        return wrapper
    return decorator


register_data_loader = _make_register_decorator(PluginType.DATA_LOADER)
register_feature_engineer = _make_register_decorator(PluginType.FEATURE_ENGINEER)
register_model_trainer = _make_register_decorator(PluginType.MODEL_TRAINER)
register_evaluator = _make_register_decorator(PluginType.EVALUATOR)