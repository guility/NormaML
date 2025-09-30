from __future__ import annotations
"""
Asynchronous PluginFactory for creating plugin instances.

This factory chooses an appropriate plugin via PluginRegistry.find_plugins
and initializes it via PluginLifecycleManager.
"""
from typing import Any, Dict, List, Optional
import asyncio

from normaml.core import plugin_registry
from normaml.core import plugin_lifecycle
from normaml.core import interfaces


class PluginFactory:
    """
    Factory responsible for creating ready-to-use plugin instances.

    The factory methods are async to allow integration in async workflows.
    """

    def __init__(self, lifecycle: Optional[plugin_lifecycle.PluginLifecycleManager] = None):
        self.registry = plugin_registry.plugin_registry
        self.lifecycle = lifecycle or plugin_lifecycle.PluginLifecycleManager()

    async def _select_plugin(self, plugin_type: plugin_registry.PluginType, name: Optional[str], capability_matching: Optional[List[str]]) -> plugin_registry.PluginMetadata:
        """
        Select a plugin metadata based on preference.

        Args:
            plugin_type: type of plugin to search
            name: optional explicit plugin name
            capability_matching: optional list of capability strings to match (any match accepted)

        Returns:
            PluginMetadata of chosen plugin.

        Raises:
            LookupError if no plugin found.
        """
        # explicit name
        if name:
            metas = self.registry.find_plugins(plugin_type=plugin_type, name=name)
            if metas:
                return metas[0]
            raise LookupError(f"No plugin named '{name}' for type {plugin_type}")

        # capability matching
        if capability_matching:
            for cap in capability_matching:
                metas = self.registry.find_plugins(plugin_type=plugin_type, capability=cap)
                if metas:
                    return metas[0]

        # fallback: any plugin of type
        metas = self.registry.find_plugins(plugin_type=plugin_type)
        if metas:
            return metas[0]
        raise LookupError(f"No plugin found for type {plugin_type}")

    async def create_data_loader(self, source: str, loader_name: Optional[str] = None, **config) -> interfaces.DataLoaderPlugin:
        """
        Create and initialize a DataLoader plugin instance.

        Args:
            source: data source identifier (used to match capabilities)
            loader_name: optional explicit loader name
            **config: configuration forwarded to plugin.initialize

        Returns:
            Initialized DataLoaderPlugin instance.
        """
        # use source token as capability hint
        capability_hints = [source]
        meta = await self._select_plugin(plugin_registry.PluginType.DATA_LOADER, loader_name, capability_hints)
        # initialize via lifecycle
        inst = self.lifecycle.initialize_plugin(meta.name, config=config)
        return inst  # type: ignore

    async def create_feature_engineer(self, capabilities: Optional[List[str]] = None, engineer_name: Optional[str] = None, **config) -> interfaces.FeatureEngineerPlugin:
        """
        Create and initialize a FeatureEngineer plugin.

        Args:
            capabilities: list of desired capabilities (match any)
            engineer_name: optional explicit plugin name
            **config: forwarded to plugin.initialize
        """
        hints = capabilities or []
        meta = await self._select_plugin(plugin_registry.PluginType.FEATURE_ENGINEER, engineer_name, hints)
        inst = self.lifecycle.initialize_plugin(meta.name, config=config)
        return inst  # type: ignore

    async def create_model_trainer(self, task_type: str, backend: Optional[str] = None, trainer_name: Optional[str] = None, **config) -> interfaces.ModelTrainerPlugin:
        """
        Create a ModelTrainer suitable for a task_type.

        Args:
            task_type: e.g., 'classification' or 'regression'
            backend: optional backend hint (capability)
            trainer_name: explicit trainer name
            **config: forwarded to plugin.initialize
        """
        hints = [task_type]
        if backend:
            hints.append(backend)
        meta = await self._select_plugin(plugin_registry.PluginType.MODEL_TRAINER, trainer_name, hints)
        inst = self.lifecycle.initialize_plugin(meta.name, config=config)
        return inst  # type: ignore

    async def create_evaluator(self, task_type: str, metrics: Optional[List[str]] = None, evaluator_name: Optional[str] = None, **config) -> interfaces.EvaluatorPlugin:
        """
        Create an Evaluator plugin instance.

        Args:
            task_type: ML task type
            metrics: optional list of desired metrics (capabilities)
            evaluator_name: explicit name
            **config: forwarded to plugin.initialize
        """
        hints = [task_type] + (metrics or [])
        meta = await self._select_plugin(plugin_registry.PluginType.EVALUATOR, evaluator_name, hints)
        inst = self.lifecycle.initialize_plugin(meta.name, config=config)
        return inst  # type: ignore