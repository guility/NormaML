import asyncio
import pytest

from normaml.core import plugin_registry
from normaml.core.plugin_lifecycle import PluginLifecycleManager
from normaml.plugins.base import BasePlugin
from normaml.core import interfaces


# Ensure clean registry per test
def setup_function():
    plugin_registry.plugin_registry._plugins.clear()


@plugin_registry.register_data_loader("exec_plugin", capabilities=["exec"])
class ExecPlugin(BasePlugin):
    name = "exec_plugin"
    capabilities = ["exec"]

    def validate_config(self, config: dict) -> bool:
        # accept any dict
        return isinstance(config, dict)

    def compute(self, a: int, b: int) -> int:
        return a + b


@plugin_registry.register_data_loader("validator_plugin", capabilities=["validate"])
class ValidatorPlugin(BasePlugin):
    name = "validator_plugin"
    capabilities = ["validate"]

    def validate_config(self, config: dict) -> bool:
        # require 'required' key
        return isinstance(config, dict) and "required" in config

    def ping(self) -> int:
        return 1


def test_initialize_and_execute_thread_and_process():
    mgr = PluginLifecycleManager()
    # initialize
    inst = mgr.initialize_plugin("exec_plugin", config={"x": 1})
    assert isinstance(inst, ExecPlugin)
    # execute in thread
    res = mgr.execute_plugin("exec_plugin", "compute", 2, 3, isolation="thread")
    assert res == 5
    # execute in process (should be quick)
    res2 = mgr.execute_plugin("exec_plugin", "compute", 10, 7, isolation="process")
    assert res2 == 17
    mgr.cleanup_all()


def test_validate_config_success_and_failure_and_cleanup():
    mgr = PluginLifecycleManager()
    # valid config
    inst = mgr.initialize_plugin("validator_plugin", config={"required": True})
    assert isinstance(inst, ValidatorPlugin)
    # invalid config should raise PluginConfigurationError
    with pytest.raises(interfaces.PluginConfigurationError):
        mgr.initialize_plugin("validator_plugin", config={})
    # cleanup individual
    mgr.cleanup_plugin("validator_plugin")
    with pytest.raises(KeyError):
        mgr.get_plugin_instance("validator_plugin")


def test_execute_docker_not_implemented():
    mgr = PluginLifecycleManager()
    mgr.initialize_plugin("exec_plugin", config={})
    with pytest.raises(NotImplementedError):
        mgr.execute_plugin("exec_plugin", "compute", 1, 2, isolation="docker")
    mgr.cleanup_all()