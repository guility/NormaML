import pytest

from normaml.core import plugin_registry
from normaml.core.plugin_registry import PluginMetadata, PluginType
from normaml.plugins.base import BasePlugin


def setup_function():
    # clear registry between tests
    plugin_registry.plugin_registry._plugins.clear()


def test_decorator_registration_and_find():
    @plugin_registry.register_data_loader("csv_loader", capabilities=["csv"])
    class CSVLoader:
        pass

    metas = plugin_registry.plugin_registry.find_plugins(plugin_type=PluginType.DATA_LOADER)
    names = [m.name for m in metas]
    assert "csv_loader" in names

    metas_csv = plugin_registry.plugin_registry.find_plugins(plugin_type=PluginType.DATA_LOADER, capability="csv")
    assert len(metas_csv) == 1
    assert metas_csv[0].name == "csv_loader"


def test_manual_register_and_find_by_name_and_capability():
    meta = PluginMetadata(name="manual_loader", plugin_type=PluginType.DATA_LOADER, cls=BasePlugin, capabilities=["manual"])
    plugin_registry.plugin_registry.register_plugin(meta)

    found = plugin_registry.plugin_registry.find_plugins(name="manual_loader")
    assert len(found) == 1
    assert found[0].name == "manual_loader"

    found_cap = plugin_registry.plugin_registry.find_plugins(plugin_type=PluginType.DATA_LOADER, capability="manual")
    assert found_cap[0].name == "manual_loader"


def test_resolve_dependencies_and_cycle_detection():
    # A depends on B
    meta_b = PluginMetadata(name="B", plugin_type=PluginType.FEATURE_ENGINEER, cls=BasePlugin, dependencies=[])
    meta_a = PluginMetadata(name="A", plugin_type=PluginType.FEATURE_ENGINEER, cls=BasePlugin, dependencies=["B"])
    plugin_registry.plugin_registry.register_plugin(meta_a)
    plugin_registry.plugin_registry.register_plugin(meta_b)

    order = plugin_registry.plugin_registry.resolve_dependencies(["A"])
    assert order.index("B") < order.index("A")
    assert order == ["B", "A"]

    # create cycle C -> D -> C
    meta_c = PluginMetadata(name="C", plugin_type=PluginType.FEATURE_ENGINEER, cls=BasePlugin, dependencies=["D"])
    meta_d = PluginMetadata(name="D", plugin_type=PluginType.FEATURE_ENGINEER, cls=BasePlugin, dependencies=["C"])
    plugin_registry.plugin_registry.register_plugin(meta_c)
    plugin_registry.plugin_registry.register_plugin(meta_d)

    with pytest.raises(ValueError):
        plugin_registry.plugin_registry.resolve_dependencies(["C"])