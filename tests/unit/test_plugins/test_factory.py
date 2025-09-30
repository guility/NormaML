import asyncio

from normaml.core import plugin_registry
from normaml.core.plugin_factory import PluginFactory
from normaml.plugins.base import BasePlugin


def setup_function():
    plugin_registry.plugin_registry._plugins.clear()


@plugin_registry.register_data_loader("csv_loader", capabilities=["csv", "file"])
class CSVLoader(BasePlugin):
    name = "csv_loader"
    capabilities = ["csv", "file"]

    def validate_config(self, config: dict) -> bool:
        return True

    def load(self, source: str, **kwargs):
        return f"loaded:{source}"


@plugin_registry.register_feature_engineer("simple_engineer", capabilities=["basic"])
class SimpleEngineer(BasePlugin):
    name = "simple_engineer"
    capabilities = ["basic"]

    def validate_config(self, config: dict) -> bool:
        return True

    def analyze_features(self, data_handle, **kwargs):
        return {"num_features": 1}


@plugin_registry.register_model_trainer("dummy_trainer", capabilities=["classification"])
class DummyTrainer(BasePlugin):
    name = "dummy_trainer"
    capabilities = ["classification"]

    def validate_config(self, config: dict) -> bool:
        return True

    def train(self, data_handle, target: str, config: dict = None):
        return {"model": "dummy", "target": target}


@plugin_registry.register_evaluator("simple_evaluator", capabilities=["accuracy"])
class SimpleEvaluator(BasePlugin):
    name = "simple_evaluator"
    capabilities = ["accuracy"]

    def validate_config(self, config: dict) -> bool:
        return True

    def evaluate(self, model, test_data_handle, target: str):
        return {"accuracy": 1.0}


def test_factory_creates_data_loader_and_calls_load():
    factory = PluginFactory()
    loader = asyncio.run(factory.create_data_loader("csv"))
    assert loader.name == "csv_loader"
    res = loader.load("somefile.csv")
    assert res == "loaded:somefile.csv"


def test_factory_creates_feature_engineer_and_trainer_and_evaluator():
    factory = PluginFactory()
    engineer = asyncio.run(factory.create_feature_engineer(capabilities=["basic"]))
    assert engineer.name == "simple_engineer"
    feats = engineer.analyze_features(None)
    assert feats.get("num_features") == 1

    trainer = asyncio.run(factory.create_model_trainer("classification"))
    assert trainer.name == "dummy_trainer"
    model = trainer.train(None, target="y")
    assert model["model"] == "dummy"

    evaluator = asyncio.run(factory.create_evaluator("classification", metrics=["accuracy"]))
    assert evaluator.name == "simple_evaluator"
    metrics = evaluator.evaluate(model, None, target="y")
    assert metrics["accuracy"] == 1.0