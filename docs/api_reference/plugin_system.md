# Plugin System Architecture для NormaML

## Обзор plugin системы

Plugin система NormaML обеспечивает модульность, расширяемость и независимое тестирование компонентов. Система поддерживает multiple discovery механизмы, dependency resolution, версионирование и изоляцию.

```
┌─────────────────────────────────────────────────────────────┐
│                   Plugin Discovery Layer                   │
├─────────────────┬───────────────┬─────────────────────────────┤
│  Entry Points   │  Decorators   │    Config-based Loading    │
└─────────────────┴───────────────┴─────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                   Plugin Registry                          │
├─────────────────┬───────────────┬─────────────────────────────┤
│   Metadata      │ Dependencies  │      Capabilities          │
│   Storage       │  Resolution   │      Validation            │
└─────────────────┴───────────────┴─────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                 Plugin Lifecycle Manager                   │
├─────────────────┬───────────────┬─────────────────────────────┤
│ Initialization  │   Execution   │      Cleanup               │
│   & Validation  │   & Isolation │   & Resource Management    │
└─────────────────┴───────────────┴─────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                     Plugin Factory                         │
├─────────────────┬───────────────┬─────────────────────────────┤
│ Instance        │ Configuration │      Shared Memory         │
│ Creation        │  Management   │      Setup                 │
└─────────────────┴───────────────┴─────────────────────────────┘
```

## Plugin Discovery Mechanisms

### 1. Entry Points Discovery

```python
# setup.py для plugin пакета
setup(
    name="normaml-sklearn-plugin",
    entry_points={
        'normaml.data_loaders': [
            'csv = normaml_sklearn.loaders:CSVLoader',
            'parquet = normaml_sklearn.loaders:ParquetLoader',
        ],
        'normaml.feature_engineers': [
            'sklearn_selector = normaml_sklearn.features:SklearnFeatureSelector',
            'sklearn_transformer = normaml_sklearn.features:SklearnTransformer',
        ],
        'normaml.model_trainers': [
            'sklearn_classifier = normaml_sklearn.trainers:SklearnClassifier',
            'sklearn_regressor = normaml_sklearn.trainers:SklearnRegressor',
        ],
        'normaml.evaluators': [
            'sklearn_evaluator = normaml_sklearn.evaluators:SklearnEvaluator',
        ],
        'normaml.backends': [
            'sklearn = normaml_sklearn.backend:SklearnBackend',
        ]
    }
)
```

### 2. Decorator-based Registration

```python
from normaml.core.plugin_manager import plugin_registry

@plugin_registry.register_data_loader(
    name="advanced_csv_loader",
    version="1.0.0",
    capabilities=["streaming", "chunked_loading", "schema_inference"],
    dependencies=["polars>=0.19.0", "pyarrow>=10.0.0"]
)
class AdvancedCSVLoader(DataLoaderPlugin):
    def __init__(self):
        super().__init__()
        self.name = "advanced_csv_loader"
        self.version = "1.0.0"
    
    def supports_source(self, source: str) -> bool:
        return source.endswith('.csv') or source.startswith('http')
    
    def load(self, source: str, **kwargs) -> pl.LazyFrame:
        return pl.scan_csv(source, **kwargs)

@plugin_registry.register_feature_engineer(
    name="polars_feature_engineer", 
    version="2.1.0",
    capabilities=["lazy_evaluation", "parallel_processing"],
    dependencies=["polars>=0.19.0"]
)
class PolarsFeatureEngineer(FeatureEngineerPlugin):
    # Implementation
    pass
```

### 3. Configuration-based Loading

```yaml
# normaml_config.yaml
plugins:
  discovery:
    - type: "entry_points"
      enabled: true
    - type: "decorators" 
      enabled: true
    - type: "config_based"
      enabled: true
      
  modules:
    - name: "custom_data_loader"
      module: "myproject.plugins.data_loaders"
      class: "CustomCSVLoader"
      config:
        chunk_size: 10000
        parallel_loading: true
        
    - name: "xgboost_trainer"
      module: "myproject.plugins.trainers"
      class: "XGBoostTrainer"
      docker_image: "normaml/xgboost:latest"
      config:
        n_estimators: 100
        max_depth: 6
        
  dependencies:
    resolution_strategy: "strict"  # strict, permissive, latest
    allow_conflicts: false
    
  isolation:
    default_mode: "process"  # thread, process, docker
    shared_memory: true
    resource_limits:
      max_memory: "4GB"
      max_cpu: 4
```

## Plugin Registry Implementation

```python
from typing import Dict, List, Any, Optional, Type, Union
from dataclasses import dataclass, field
from enum import Enum
import importlib
import pkg_resources
from packaging import version

class PluginType(Enum):
    DATA_LOADER = "data_loader"
    FEATURE_ENGINEER = "feature_engineer"
    MODEL_TRAINER = "model_trainer"
    EVALUATOR = "evaluator"
    BACKEND = "backend"
    CUSTOM = "custom"

@dataclass
class PluginMetadata:
    name: str
    version: str
    plugin_type: PluginType
    class_path: str
    module_path: str
    capabilities: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    docker_image: Optional[str] = None
    config_schema: Optional[Dict[str, Any]] = None
    supports_shared_memory: bool = True
    isolation_mode: str = "process"  # thread, process, docker
    resource_requirements: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        self.version_obj = version.parse(self.version)

@dataclass 
class PluginInstance:
    metadata: PluginMetadata
    instance: Plugin
    config: Dict[str, Any] = field(default_factory=dict)
    is_initialized: bool = False
    shared_memory_handle: Optional[str] = None

class PluginRegistry:
    """Центральный реестр всех plugins"""
    
    def __init__(self):
        self.plugins: Dict[str, PluginMetadata] = {}
        self.plugin_instances: Dict[str, PluginInstance] = {}
        self.type_index: Dict[PluginType, List[str]] = {
            plugin_type: [] for plugin_type in PluginType
        }
        self.capability_index: Dict[str, List[str]] = {}
        self.dependency_graph: Dict[str, List[str]] = {}
    
    def register_plugin(self, metadata: PluginMetadata) -> None:
        """Регистрация plugin в реестре"""
        if metadata.name in self.plugins:
            existing = self.plugins[metadata.name]
            if existing.version_obj >= metadata.version_obj:
                return  # Существующая версия новее или равна
        
        self.plugins[metadata.name] = metadata
        self.type_index[metadata.plugin_type].append(metadata.name)
        
        # Индексирование capabilities
        for capability in metadata.capabilities:
            if capability not in self.capability_index:
                self.capability_index[capability] = []
            self.capability_index[capability].append(metadata.name)
        
        # Построение dependency graph
        self.dependency_graph[metadata.name] = metadata.dependencies
    
    def discover_entry_points(self) -> None:
        """Обнаружение plugins через entry points"""
        entry_point_groups = {
            'normaml.data_loaders': PluginType.DATA_LOADER,
            'normaml.feature_engineers': PluginType.FEATURE_ENGINEER,
            'normaml.model_trainers': PluginType.MODEL_TRAINER,
            'normaml.evaluators': PluginType.EVALUATOR,
            'normaml.backends': PluginType.BACKEND,
        }
        
        for group_name, plugin_type in entry_point_groups.items():
            for entry_point in pkg_resources.iter_entry_points(group_name):
                try:
                    plugin_class = entry_point.load()
                    metadata = self._extract_metadata_from_class(
                        plugin_class, plugin_type, entry_point.name
                    )
                    self.register_plugin(metadata)
                except Exception as e:
                    print(f"Failed to load plugin {entry_point.name}: {e}")
    
    def _extract_metadata_from_class(self, plugin_class: Type[Plugin], 
                                   plugin_type: PluginType, 
                                   name: str) -> PluginMetadata:
        """Извлечение metadata из класса plugin"""
        # Получаем metadata из атрибутов класса или аннотаций
        metadata = PluginMetadata(
            name=getattr(plugin_class, 'name', name),
            version=getattr(plugin_class, 'version', '1.0.0'),
            plugin_type=plugin_type,
            class_path=f"{plugin_class.__module__}.{plugin_class.__name__}",
            module_path=plugin_class.__module__,
            capabilities=getattr(plugin_class, 'capabilities', []),
            dependencies=getattr(plugin_class, 'dependencies', []),
            docker_image=getattr(plugin_class, 'docker_image', None),
            supports_shared_memory=getattr(plugin_class, 'supports_shared_memory', True),
            isolation_mode=getattr(plugin_class, 'isolation_mode', 'process')
        )
        return metadata
    
    def find_plugins(self, plugin_type: PluginType = None,
                    capabilities: List[str] = None,
                    name_pattern: str = None) -> List[PluginMetadata]:
        """Поиск plugins по критериям"""
        candidates = list(self.plugins.values())
        
        if plugin_type:
            candidates = [p for p in candidates if p.plugin_type == plugin_type]
        
        if capabilities:
            candidates = [
                p for p in candidates 
                if all(cap in p.capabilities for cap in capabilities)
            ]
        
        if name_pattern:
            import re
            pattern = re.compile(name_pattern)
            candidates = [p for p in candidates if pattern.search(p.name)]
        
        return sorted(candidates, key=lambda p: p.version_obj, reverse=True)
    
    def resolve_dependencies(self, plugin_names: List[str]) -> List[str]:
        """Разрешение зависимостей между plugins"""
        resolved = []
        visiting = set()
        visited = set()
        
        def dfs(plugin_name: str):
            if plugin_name in visiting:
                raise ValueError(f"Circular dependency detected: {plugin_name}")
            if plugin_name in visited:
                return
            
            visiting.add(plugin_name)
            
            for dependency in self.dependency_graph.get(plugin_name, []):
                dfs(dependency)
            
            visiting.remove(plugin_name)
            visited.add(plugin_name)
            resolved.append(plugin_name)
        
        for plugin_name in plugin_names:
            dfs(plugin_name)
        
        return resolved
    
    def get_plugin_metadata(self, name: str) -> Optional[PluginMetadata]:
        """Получение metadata plugin по имени"""
        return self.plugins.get(name)
    
    def list_plugins(self, plugin_type: PluginType = None) -> List[PluginMetadata]:
        """Список всех зарегистрированных plugins"""
        if plugin_type:
            plugin_names = self.type_index[plugin_type]
            return [self.plugins[name] for name in plugin_names]
        return list(self.plugins.values())

# Глобальный реестр
plugin_registry = PluginRegistry()

# Декораторы для регистрации
def register_data_loader(name: str, version: str = "1.0.0", **kwargs):
    def decorator(cls):
        metadata = PluginMetadata(
            name=name,
            version=version,
            plugin_type=PluginType.DATA_LOADER,
            class_path=f"{cls.__module__}.{cls.__name__}",
            module_path=cls.__module__,
            **kwargs
        )
        plugin_registry.register_plugin(metadata)
        return cls
    return decorator

def register_feature_engineer(name: str, version: str = "1.0.0", **kwargs):
    def decorator(cls):
        metadata = PluginMetadata(
            name=name,
            version=version,
            plugin_type=PluginType.FEATURE_ENGINEER,
            class_path=f"{cls.__module__}.{cls.__name__}",
            module_path=cls.__module__,
            **kwargs
        )
        plugin_registry.register_plugin(metadata)
        return cls
    return decorator

# Аналогично для других типов plugins...
```

## Plugin Lifecycle Manager

```python
import asyncio
from typing import Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

class PluginLifecycleManager:
    """Управление жизненным циклом plugins"""
    
    def __init__(self, registry: PluginRegistry, 
                 data_manager: 'SharedMemoryDataManager'):
        self.registry = registry
        self.data_manager = data_manager
        self.active_plugins: Dict[str, PluginInstance] = {}
        self.thread_pool = ThreadPoolExecutor()
        self.process_pool = ProcessPoolExecutor()
    
    async def initialize_plugin(self, plugin_name: str, 
                               config: Dict[str, Any] = None) -> PluginInstance:
        """Инициализация plugin"""
        metadata = self.registry.get_plugin_metadata(plugin_name)
        if not metadata:
            raise ValueError(f"Plugin {plugin_name} not found")
        
        # Проверяем зависимости
        dependencies = self.registry.resolve_dependencies([plugin_name])
        for dep in dependencies[:-1]:  # Исключаем сам plugin
            if dep not in self.active_plugins:
                await self.initialize_plugin(dep)
        
        # Загружаем класс plugin
        plugin_class = self._load_plugin_class(metadata)
        
        # Создаем экземпляр
        instance = plugin_class()
        
        # Валидируем конфигурацию
        plugin_config = config or {}
        if not instance.validate_config(plugin_config):
            raise ValueError(f"Invalid config for plugin {plugin_name}")
        
        # Настраиваем shared memory если поддерживается
        if metadata.supports_shared_memory:
            shm_config = self._setup_shared_memory(plugin_name, metadata)
            instance.setup_shared_memory(shm_config)
        
        # Инициализируем plugin
        instance.initialize(plugin_config)
        
        # Создаем instance wrapper
        plugin_instance = PluginInstance(
            metadata=metadata,
            instance=instance,
            config=plugin_config,
            is_initialized=True
        )
        
        self.active_plugins[plugin_name] = plugin_instance
        return plugin_instance
    
    def _load_plugin_class(self, metadata: PluginMetadata) -> Type[Plugin]:
        """Загрузка класса plugin"""
        module = importlib.import_module(metadata.module_path)
        class_name = metadata.class_path.split('.')[-1]
        return getattr(module, class_name)
    
    def _setup_shared_memory(self, plugin_name: str, 
                           metadata: PluginMetadata) -> Dict[str, Any]:
        """Настройка shared memory для plugin"""
        # Создаем именованную область shared memory для plugin
        shm_name = f"normaml_plugin_{plugin_name}"
        return {
            'shm_name': shm_name,
            'data_manager': self.data_manager,
            'max_size': metadata.resource_requirements.get('max_memory', '1GB')
        }
    
    async def execute_plugin(self, plugin_name: str, method: str, 
                           *args, **kwargs) -> Any:
        """Выполнение метода plugin"""
        if plugin_name not in self.active_plugins:
            raise ValueError(f"Plugin {plugin_name} not initialized")
        
        plugin_instance = self.active_plugins[plugin_name]
        metadata = plugin_instance.metadata
        instance = plugin_instance.instance
        
        # Получаем метод
        if not hasattr(instance, method):
            raise AttributeError(f"Plugin {plugin_name} has no method {method}")
        
        plugin_method = getattr(instance, method)
        
        # Выбираем режим выполнения
        if metadata.isolation_mode == "thread":
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                self.thread_pool, plugin_method, *args, **kwargs
            )
        elif metadata.isolation_mode == "process":
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                self.process_pool, plugin_method, *args, **kwargs
            )
        elif metadata.isolation_mode == "docker":
            # Выполнение в Docker контейнере будет реализовано в ExecutionEngine
            raise NotImplementedError("Docker execution via ExecutionEngine")
        else:
            # Прямое выполнение
            return plugin_method(*args, **kwargs)
    
    def get_plugin_instance(self, plugin_name: str) -> Optional[PluginInstance]:
        """Получение экземпляра plugin"""
        return self.active_plugins.get(plugin_name)
    
    async def cleanup_plugin(self, plugin_name: str) -> None:
        """Очистка plugin"""
        if plugin_name in self.active_plugins:
            plugin_instance = self.active_plugins[plugin_name]
            try:
                plugin_instance.instance.cleanup()
            except Exception as e:
                print(f"Error during cleanup of {plugin_name}: {e}")
            finally:
                del self.active_plugins[plugin_name]
    
    async def cleanup_all(self) -> None:
        """Очистка всех активных plugins"""
        cleanup_tasks = [
            self.cleanup_plugin(name) 
            for name in list(self.active_plugins.keys())
        ]
        await asyncio.gather(*cleanup_tasks, return_exceptions=True)
```

## Plugin Factory

```python
class PluginFactory:
    """Factory для создания и конфигурирования plugins"""
    
    def __init__(self, lifecycle_manager: PluginLifecycleManager):
        self.lifecycle_manager = lifecycle_manager
        self.registry = lifecycle_manager.registry
    
    async def create_data_loader(self, source: str, 
                               loader_name: str = None,
                               **config) -> DataLoaderPlugin:
        """Создание data loader plugin"""
        if loader_name:
            plugins = [self.registry.get_plugin_metadata(loader_name)]
        else:
            # Автоматический выбор подходящего loader
            loaders = self.registry.find_plugins(PluginType.DATA_LOADER)
            plugins = []
            for loader_meta in loaders:
                # Проверяем, может ли loader обработать источник
                try:
                    temp_instance = await self.lifecycle_manager.initialize_plugin(
                        loader_meta.name, config
                    )
                    if temp_instance.instance.supports_source(source):
                        plugins.append(loader_meta)
                        break
                except:
                    continue
        
        if not plugins:
            raise ValueError(f"No suitable data loader found for {source}")
        
        plugin_meta = plugins[0]
        instance = await self.lifecycle_manager.initialize_plugin(
            plugin_meta.name, config
        )
        return instance.instance
    
    async def create_feature_engineer(self, capabilities: List[str] = None,
                                    engineer_name: str = None,
                                    **config) -> FeatureEngineerPlugin:
        """Создание feature engineer plugin"""
        if engineer_name:
            plugins = [self.registry.get_plugin_metadata(engineer_name)]
        else:
            plugins = self.registry.find_plugins(
                PluginType.FEATURE_ENGINEER, 
                capabilities=capabilities
            )
        
        if not plugins:
            cap_str = ", ".join(capabilities) if capabilities else "any"
            raise ValueError(f"No feature engineer found with capabilities: {cap_str}")
        
        plugin_meta = plugins[0]
        instance = await self.lifecycle_manager.initialize_plugin(
            plugin_meta.name, config
        )
        return instance.instance
    
    async def create_model_trainer(self, task_type: str,
                                 backend: str = None,
                                 trainer_name: str = None,
                                 **config) -> ModelTrainerPlugin:
        """Создание model trainer plugin"""
        if trainer_name:
            plugins = [self.registry.get_plugin_metadata(trainer_name)]
        else:
            trainers = self.registry.find_plugins(PluginType.MODEL_TRAINER)
            plugins = []
            for trainer_meta in trainers:
                try:
                    temp_instance = await self.lifecycle_manager.initialize_plugin(
                        trainer_meta.name, config
                    )
                    if temp_instance.instance.supports_task_type(task_type):
                        plugins.append(trainer_meta)
                        break
                except:
                    continue
        
        if not plugins:
            raise ValueError(f"No suitable trainer found for task type: {task_type}")
        
        plugin_meta = plugins[0]
        instance = await self.lifecycle_manager.initialize_plugin(
            plugin_meta.name, config
        )
        return instance.instance
    
    async def create_evaluator(self, task_type: str,
                             metrics: List[str] = None,
                             evaluator_name: str = None,
                             **config) -> EvaluatorPlugin:
        """Создание evaluator plugin"""
        if evaluator_name:
            plugins = [self.registry.get_plugin_metadata(evaluator_name)]
        else:
            evaluators = self.registry.find_plugins(PluginType.EVALUATOR)
            plugins = []
            for eval_meta in evaluators:
                try:
                    temp_instance = await self.lifecycle_manager.initialize_plugin(
                        eval_meta.name, config
                    )
                    supported_metrics = temp_instance.instance.get_supported_metrics(task_type)
                    if not metrics or any(m in supported_metrics for m in metrics):
                        plugins.append(eval_meta)
                        break
                except:
                    continue
        
        if not plugins:
            raise ValueError(f"No suitable evaluator found for task type: {task_type}")
        
        plugin_meta = plugins[0]
        instance = await self.lifecycle_manager.initialize_plugin(
            plugin_meta.name, config
        )
        return instance.instance
```

## Plugin Configuration Management

```python
from pydantic import BaseModel, ValidationError
from typing import Dict, Any, Optional

class PluginConfigValidator:
    """Валидация конфигурации plugins"""
    
    @staticmethod
    def validate_config(plugin_metadata: PluginMetadata, 
                       config: Dict[str, Any]) -> bool:
        """Валидация конфигурации plugin по schema"""
        if not plugin_metadata.config_schema:
            return True  # Нет schema - считаем валидной
        
        try:
            # Можно использовать jsonschema или pydantic для валидации
            import jsonschema
            jsonschema.validate(config, plugin_metadata.config_schema)
            return True
        except ValidationError:
            return False
        except Exception:
            return True  # Если schema невалидна, пропускаем валидацию

class PluginConfigManager:
    """Управление конфигурациями plugins"""
    
    def __init__(self):
        self.configs: Dict[str, Dict[str, Any]] = {}
        self.defaults: Dict[str, Dict[str, Any]] = {}
    
    def set_plugin_config(self, plugin_name: str, config: Dict[str, Any]) -> None:
        """Установка конфигурации для plugin"""
        self.configs[plugin_name] = config
    
    def get_plugin_config(self, plugin_name: str) -> Dict[str, Any]:
        """Получение конфигурации plugin"""
        return self.configs.get(plugin_name, {})
    
    def merge_with_defaults(self, plugin_name: str, 
                           user_config: Dict[str, Any]) -> Dict[str, Any]:
        """Слияние пользовательской конфигурации с defaults"""
        defaults = self.defaults.get(plugin_name, {})
        merged = defaults.copy()
        merged.update(user_config)
        return merged
    
    def load_from_file(self, config_path: str) -> None:
        """Загрузка конфигурации из файла"""
        import yaml
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        for plugin_name, plugin_config in config_data.get('plugins', {}).items():
            self.set_plugin_config(plugin_name, plugin_config)
```

## Примеры использования Plugin System

```python
# Инициализация системы
registry = PluginRegistry()
data_manager = SharedMemoryDataManager()
lifecycle_manager = PluginLifecycleManager(registry, data_manager)
factory = PluginFactory(lifecycle_manager)

# Обнаружение plugins
registry.discover_entry_points()

# Создание data loader
loader = await factory.create_data_loader(
    source="data.csv",
    chunk_size=10000,
    parallel_loading=True
)

# Создание feature engineer с определенными возможностями
feature_engineer = await factory.create_feature_engineer(
    capabilities=["feature_selection", "dimensionality_reduction"],
    selection_method="mutual_info"
)

# Создание model trainer
trainer = await factory.create_model_trainer(
    task_type="classification",
    backend="sklearn",
    model_type="random_forest"
)

# Создание evaluator
evaluator = await factory.create_evaluator(
    task_type="classification",
    metrics=["accuracy", "f1_score", "roc_auc"]
)
```

Эта plugin система обеспечивает:

1. **Гибкое обнаружение** - через entry points, декораторы и конфигурацию
2. **Управление зависимостями** - автоматическое разрешение и инициализация 
3. **Изоляцию** - поддержка thread, process и docker isolation
4. **Shared Memory** - zero-copy передача данных между plugins
5. **Конфигурируемость** - валидация и управление конфигурациями
6. **Расширяемость** - простое добавление новых типов plugins
7. **Безопасность** - контролируемый lifecycle и resource management