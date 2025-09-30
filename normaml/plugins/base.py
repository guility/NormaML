from __future__ import annotations
"""
BasePlugin common utilities for NormaML plugins.
"""
from typing import Dict, Any, List
import logging

from normaml.core import interfaces


logger = logging.getLogger(__name__)


class BasePlugin(interfaces.Plugin):
    """
    Simple base implementation for plugins.

    Provides:
    - name, version, capabilities, dependencies attributes
    - basic logging via self.logger
    - default setup_shared_memory and cleanup implementations
    """

    name: str = "base_plugin"
    version: str = "0.0.0"
    capabilities: List[str] = []
    dependencies: List[str] = []
    supports_shared_memory: bool = False

    def __init__(self) -> None:
        self.logger = logger.getChild(self.__class__.__name__)
        self._initialized = False
        self._config: Dict[str, Any] = {}

    def initialize(self, config: Dict[str, Any]) -> None:
        """
        Store config and mark as initialized.

        Args:
            config: configuration dict
        """
        self._config = config or {}
        self._initialized = True
        self.logger.debug("Initialized plugin %s with config %s", self.name, self._config)

    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        Default validation accepts any dict.

        Override in concrete plugins.
        """
        return isinstance(config, dict)

    def get_capabilities(self) -> Dict[str, Any]:
        """Return list of capabilities."""
        return {"capabilities": list(self.capabilities)}

    def setup_shared_memory(self, shm_config: Dict[str, Any]) -> None:
        """Default no-op shared memory setup."""
        self.logger.debug("setup_shared_memory called with %s", shm_config)

    def cleanup(self) -> None:
        """Default cleanup - clear state."""
        self._initialized = False
        self._config = {}
        self.logger.debug("cleanup called for %s", self.name)