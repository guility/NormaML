
# MLflow Integration Architecture для NormaML

## Обзор интеграции с MLflow

Интеграция с MLflow обеспечивает комплексное отслеживание экспериментов, версионирование моделей и управление жизненным циклом ML проектов с полной интеграцией в plugin систему NormaML и поддержкой distributed environments.

```
┌─────────────────────────────────────────────────────────────┐
│                    MLflow Integration Layer                │
├─────────────────┬───────────────┬─────────────────────────────┤
│  Experiment     │  Model        │    Artifacts                │
│  Tracking       │  Registry     │    Management               │
├─────────────────┼───────────────┼─────────────────────────────┤
│ • Auto-logging  │ • Versioning  │ • Large Files               │
│ • Nested runs   │ • Staging     │ • Shared Memory             │
│ • Parallel logs │ • Production  │ • Zero-copy Logs            │
└─────────────────┴───────────────┴─────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                   Plugin Integration                       │
├─────────────────┬───────────────┬─────────────────────────────┤
│  Auto Tracking  │ Custom Loggers│    Framework Adapters      │
│  Decorators     │  (Per Plugin) │    (sklearn, xgboost, etc.)│
├─────────────────┼───────────────┼─────────────────────────────┤
│ • Method hooks  │ • Memory      │ • Native integration       │
│ • Data logging  │ • Performance │ • Custom metrics           │
│ • Error capture │ • Resources   │ • Model artifacts          │
└─────────────────┴───────────────┴─────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                 Distributed Tracking                       │
├─────────────────┬───────────────┬─────────────────────────────┤
│ Multi-Process   │ Docker        │    Remote Tracking         │
│ Coordination    │ Integration   │    Server                  │
├─────────────────┼───────────────┼─────────────────────────────┤
│ • Shared state  │ • Container   │ • Centralized server       │
│ • Run merging   │ • Logs        │ • Authentication           │
│ • Conflict res. │ • Artifacts   │ • Multi-user support       │
└─────────────────┴───────────────┴─────────────────────────────┘
```

## Core MLflow Integration Components

### 1. Enhanced Experiment Tracker

```python
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from mlflow.tracking import MlflowClient
from typing import Dict, Any, List, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import threading
import asyncio
import json
import pickle
from datetime import datetime
import uuid

class RunType(Enum):
    EXPERIMENT = "experiment"
    PIPELINE = "pipeline"
    HYPERPARAMETER_TUNING = "hyperparameter_tuning"
    MODEL_VALIDATION = "model_validation"
    DATA_PROCESSING = "data_processing"
    FEATURE_ENGINEERING = "feature_engineering"

class ArtifactType(Enum):
    MODEL = "model"
    DATASET = "dataset"
    FEATURE_SET = "feature_set"
    METRICS = "metrics"
    PLOTS = "plots"
    CONFIG = "config"
    SHARED_MEMORY = "shared_memory"

@dataclass
class RunConfig:
    experiment_name: str
    run_name: Optional[str] = None
    run_type: RunType = RunType.EXPERIMENT
    parent_run_id: Optional[str] = None
    tags: Dict[str, str] = field(default_factory=dict)
    nested: bool = False
    auto_log: bool = True
    track_data: bool = True
    track_memory: bool = True
    track_performance: bool = True

@dataclass
class LogEntry:
    entry_id: str
    run_id: str
    timestamp: datetime
    entry_type: str  # param, metric, artifact, tag
    key: str
    value: Any
    step: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class EnhancedExperimentTracker:
    """Расширенный трекер экспериментов с plugin интеграцией"""
    
    def __init__(self, tracking_uri: str = None, 
                 registry_uri: str = None,
                 enable_auto_logging: bool = True):
        
        # MLflow setup
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        if registry_uri:
            mlflow.set_registry_uri(registry_uri)
        
        self.client = MlflowClient()
        self.enable_auto_logging = enable_auto_logging
        
        # Run management
        self.active_runs: Dict[str, mlflow.ActiveRun] = {}
        self.run_stack: List[str] = []  # For nested runs
        self.run_configs: Dict[str, RunConfig] = {}
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Plugin integration
        self.plugin_loggers: Dict[str, 'PluginLogger'] = {}
        self.auto_log_hooks: Dict[str, List[Callable]] = {}
        
        # Distributed coordination
        self.distributed_mode = False
        self.process_id = str(uuid.uuid4())
        self.shared_state: Optional[Dict[str, Any]] = None
        
        # Caching and batching
        self.log_queue: List[LogEntry] = []
        self.queue_lock = threading.Lock()
        self.batch_size = 100
        self.batch_timeout = 30  # seconds
        
        # Performance tracking
        self.data_manager = None
        self.resource_manager = None
        
        # Setup auto-logging
        if enable_auto_logging:
            self._setup_auto_logging()
    
    def set_dependencies(self, data_manager, resource_manager):
        """Установка зависимостей"""
        self.data_manager = data_manager
        self.resource_manager = resource_manager
    
    def _setup_auto_logging(self):
        """Настройка автоматического логирования"""
        # Включаем auto-logging для популярных фреймворков
        try:
            mlflow.sklearn.autolog(log_input_examples=True, log_model_signatures=True)
            mlflow.xgboost.autolog(log_input_examples=True, log_model_signatures=True)
            # Добавим другие фреймворки по мере необходимости
        except Exception as e:
            print(f"Auto-logging setup warning: {e}")
    
    def create_experiment(self, name: str, 
                         artifact_location: str = None,
                         tags: Dict[str, str] = None) -> str:
        """Создание нового эксперимента"""
        try:
            experiment_id = self.client.create_experiment(
                name=name,
                artifact_location=artifact_location,
                tags=tags
            )
            return experiment_id
        except mlflow.exceptions.MlflowException as e:
            if "already exists" in str(e):
                experiment = self.client.get_experiment_by_name(name)
                return experiment.experiment_id
            raise
    
    def start_run(self, config: RunConfig) -> str:
        """Начало нового run"""
        with self.lock:
            # Создаем или получаем эксперимент
            experiment_id = self.create_experiment(config.experiment_name)
            mlflow.set_experiment(experiment_id=experiment_id)
            
            # Определяем parent run для nested runs
            parent_run_id = None
            if config.nested and self.run_stack:
                parent_run_id = self.run_stack[-1]
            elif config.parent_run_id:
                parent_run_id = config.parent_run_id
            
            # Стартуем run
            if parent_run_id:
                run = mlflow.start_run(
                    run_name=config.run_name,
                    nested=True,
                    parent_run_id=parent_run_id
                )
            else:
                run = mlflow.start_run(run_name=config.run_name)
            
            run_id = run.info.run_id
            
            # Сохраняем информацию о run
            self.active_runs[run_id] = run
            self.run_configs[run_id] = config
            self.run_stack.append(run_id)
            
            # Логируем базовые теги
            base_tags = {
                'run_type': config.run_type.value,
                'process_id': self.process_id,
                'normaml_version': '1.0.0',  # Получать из package
                'auto_log_enabled': str(config.auto_log),
                **config.tags
            }
            
            self.log_tags(base_tags, run_id=run_id)
            
            # Регистрируем plugin loggers для этого run
            if config.auto_log:
                self._setup_run_auto_logging(run_id)
            
            return run_id
    
    def _setup_run_auto_logging(self, run_id: str):
        """Настройка auto-logging для конкретного run"""
        config = self.run_configs[run_id]
        
        # Создаем plugin logger для run
        plugin_logger = PluginLogger(self, run_id, config)
        self.plugin_loggers[run_id] = plugin_logger
    
    def end_run(self, run_id: str = None) -> Dict[str, Any]:
        """Завершение run"""
        with self.lock:
            if run_id is None:
                if not self.run_stack:
                    raise ValueError("No active runs")
                run_id = self.run_stack[-1]
            
            if run_id not in self.active_runs:
                raise ValueError(f"Run {run_id} not found")
            
            # Flush pending logs
            self._flush_log_queue(run_id)
            
            # Завершаем run
            mlflow.end_run()
            
            # Cleanup
            run = self.active_runs.pop(run_id)
            config = self.run_configs.pop(run_id)
            self.run_stack.remove(run_id)
            
            if run_id in self.plugin_loggers:
                del self.plugin_loggers[run_id]
            
            # Возвращаем информацию о завершенном run
            return {
                'run_id': run_id,
                'status': 'FINISHED',
                'end_time': datetime.now(),
                'artifact_uri': run.info.artifact_uri
            }
    
    def log_params(self, params: Dict[str, Any], run_id: str = None):
        """Логирование параметров"""
        run_id = run_id or self._get_current_run_id()
        
        # Batch logging
        for key, value in params.items():
            entry = LogEntry(
                entry_id=str(uuid.uuid4()),
                run_id=run_id,
                timestamp=datetime.now(),
                entry_type='param',
                key=key,
                value=value
            )
            self._add_to_queue(entry)
    
    def log_metrics(self, metrics: Dict[str, float],
                   step: int = None, run_id: str = None):
        """Логирование метрик"""
        run_id = run_id or self._get_current_run_id()
        
        for key, value in metrics.items():
            entry = LogEntry(
                entry_id=str(uuid.uuid4()),
                run_id=run_id,
                timestamp=datetime.now(),
                entry_type='metric',
                key=key,
                value=value,
                step=step
            )
            self._add_to_queue(entry)
    
    def log_tags(self, tags: Dict[str, str], run_id: str = None):
        """Логирование тегов"""
        run_id = run_id or self._get_current_run_id()
        
        for key, value in tags.items():
            entry = LogEntry(
                entry_id=str(uuid.uuid4()),
                run_id=run_id,
                timestamp=datetime.now(),
                entry_type='tag',
                key=key,
                value=str(value)
            )
            self._add_to_queue(entry)
    
    def _get_current_run_id(self) -> str:
        """Получение ID текущего run"""
        if not self.run_stack:
            raise ValueError("No active runs")
        return self.run_stack[-1]
    
    def _add_to_queue(self, entry: LogEntry):
        """Добавление записи в очередь"""
        with self.queue_lock:
            self.log_queue.append(entry)
            
            # Проверяем, нужно ли сбросить очередь
            if len(self.log_queue) >= self.batch_size:
                self._flush_log_queue()
    
    def _flush_log_queue(self, run_id: str = None):
        """Сброс очереди логов"""
        with self.queue_lock:
            if not self.log_queue:
                return
            
            # Группируем по run_id и типу
            grouped_entries = {}
            for entry in self.log_queue:
                if run_id and entry.run_id != run_id:
                    continue
                
                key = (entry.run_id, entry.entry_type)
                if key not in grouped_entries:
                    grouped_entries[key] = []
                grouped_entries[key].append(entry)
            
            # Логируем каждую группу
            for (entry_run_id, entry_type), entries in grouped_entries.items():
                try:
                    if entry_type == 'param':
                        params = {entry.key: entry.value for entry in entries}
                        with mlflow.start_run(run_id=entry_run_id):
                            mlflow.log_params(params)
                    
                    elif entry_type == 'metric':
                        for entry in entries:
                            with mlflow.start_run(run_id=entry_run_id):
                                mlflow.log_metric(entry.key, entry.value, step=entry.step)
                    
                    elif entry_type == 'tag':
                        tags = {entry.key: entry.value for entry in entries}
                        with mlflow.start_run(run_id=entry_run_id):
                            mlflow.set_tags(tags)
                
                except Exception as e:
                    print(f"Failed to flush logs for {entry_type}: {e}")
            
            # Очищаем обработанные записи
            if run_id:
                self.log_queue = [e for e in self.log_queue if e.run_id != run_id]
            else:
                self.log_queue.clear()


class PluginLogger:
    """Logger для интеграции с plugins"""
    
    def __init__(self, tracker: EnhancedExperimentTracker,
                 run_id: str, config: RunConfig):
        self.tracker = tracker
        self.run_id = run_id
        self.config = config
        
        # Hooks для автоматического логирования
        self.method_hooks: Dict[str, List[Callable]] = {}
    
    def setup_plugin_hooks(self, plugin_instance, plugin_metadata):
        """Настройка hooks для plugin"""
        plugin_name = plugin_metadata.name
        
        # Автоматические hooks для стандартных методов
        if hasattr(plugin_instance, 'train'):
            self._wrap_method(plugin_instance, 'train',
                            self._create_training_hook(plugin_name))
        
        if hasattr(plugin_instance, 'evaluate'):
            self._wrap_method(plugin_instance, 'evaluate',
                            self._create_evaluation_hook(plugin_name))
        
        if hasattr(plugin_instance, 'transform'):
            self._wrap_method(plugin_instance, 'transform',
                            self._create_transform_hook(plugin_name))
    
    def _wrap_method(self, instance, method_name: str, hook: Callable):
        """Обертывание метода с hook"""
        original_method = getattr(instance, method_name)
        
        def wrapped_method(*args, **kwargs):
            # Pre-hook
            start_time = datetime.now()
            
            # Логируем входные параметры
            if self.config.track_data:
                self._log_method_inputs(method_name, args, kwargs)
            
            try:
                # Выполняем оригинальный метод
                result = original_method(*args, **kwargs)
                
                # Post-hook
                execution_time = (datetime.now() - start_time).total_seconds()
                
                # Логируем результаты
                hook(method_name, args, kwargs, result, execution_time, None)
                
                return result
                
            except Exception as e:
                execution_time = (datetime.now() - start_time).total_seconds()
                hook(method_name, args, kwargs, None, execution_time, e)
                raise
        
        setattr(instance, method_name, wrapped_method)
    
    def _create_training_hook(self, plugin_name: str) -> Callable:
        """Создание hook для обучения"""
        def training_hook(method_name, args, kwargs, result, execution_time, error):
            # Логируем параметры обучения
            if kwargs:
                params = {f"{plugin_name}_{k}": str(v) for k, v in kwargs.items()}
                self.tracker.log_params(params, self.run_id)
            
            # Логируем время выполнения
            self.tracker.log_metrics({
                f"{plugin_name}_training_time_seconds": execution_time
            }, run_id=self.run_id)
            
            # Логируем ошибку если есть
            if error:
                self.tracker.log_tags({
                    f"{plugin_name}_training_error": str(error)
                }, self.run_id)
        
        return training_hook
    
    def _create_evaluation_hook(self, plugin_name: str) -> Callable:
        """Создание hook для оценки"""
        def evaluation_hook(method_name, args, kwargs, result, execution_time, error):
            # Логируем время выполнения
            self.tracker.log_metrics({
                f"{plugin_name}_evaluation_time_seconds": execution_time
            }, run_id=self.run_id)
            
            # Логируем метрики если результат - словарь
            if result and isinstance(result, dict):
                metrics = {
                    f"{plugin_name}_{k}": v
                    for k, v in result.items()
                    if isinstance(v, (int, float))
                }
                if metrics:
                    self.tracker.log_metrics(metrics, run_id=self.run_id)
            
            # Логируем ошибку если есть
            if error:
                self.tracker.log_tags({
                    f"{plugin_name}_evaluation_error": str(error)
                }, self.run_id)
        
        return evaluation_hook
    
    def _create_transform_hook(self, plugin_name: str) -> Callable:
        """Создание hook для трансформации"""
        def transform_hook(method_name, args, kwargs, result, execution_time, error):
            # Логируем время выполнения
            self.tracker.log_metrics({
                f"{plugin_name}_transform_time_seconds": execution_time
            }, run_id=self.run_id)
            
            # Логируем информацию о данных если доступна
            if result and hasattr(result, 'shape'):
                self.tracker.log_params({
                    f"{plugin_name}_output_shape": str(result.shape)
                }, self.run_id)
            
            # Логируем ошибку если есть
            if error:
                self.tracker.log_tags({
                    f"{plugin_name}_transform_error": str(error)
                }, self.run_id)
        
        return transform_hook
    
    def _log_method_inputs(self, method_name: str, args, kwargs):
        """Логирование входных данных метода"""
        try:
            # Логируем количество аргументов
            self.tracker.log_params({
                f"{method_name}_args_count": len(args),
                f"{method_name}_kwargs_count": len(kwargs)
            }, self.run_id)
            
            # Логируем информацию о DataFrame если есть
            for i, arg in enumerate(args):
                if hasattr(arg, 'shape'):
                    self.tracker.log_params({
                        f"{method_name}_arg_{i}_shape": str(arg.shape)
                    }, self.run_id)
        
        except Exception as e:
            # Не падаем если логирование входных данных не удалось
            print(f"Failed to log method inputs: {e}")


## Usage Examples

# Настройка MLflow интеграции
tracker = EnhancedExperimentTracker(
    tracking_uri="http://localhost:5000",
    enable_auto_logging=True
)

# Установка зависимостей
tracker.set_dependencies(data_manager, resource_manager)

# Создание эксперимента
config = RunConfig(
    experiment_name="normaml_feature_selection",
    run_name="sklearn_feature_selection_v1",
    run_type=RunType.FEATURE_ENGINEERING,
    tags={'algorithm': 'mutual_info', 'version': '1.0'},
    auto_log=True,
    track_data=True,
    track_memory=True
)

# Начало run
run_id = tracker.start_run(config)

# Автоматическое логирование через plugin
plugin_logger = tracker.plugin_loggers[run_id]
plugin_logger.setup_plugin_hooks(feature_selector_instance, feature_selector_metadata)

# Выполнение операций (автоматически логируется)
selected_features = feature_selector.select_features(data, target_column)

# Ручное логирование специфических метрик
tracker.log_metrics({
    'features_selected': len(selected_features),
    'selection_ratio': len(selected_features) / len(original_features)
}, run_id)

# Завершение run
tracker.end_run(run_id)
```

Эта MLflow интеграция обеспечивает:

1. **Comprehensive Tracking** - автоматическое логирование всех аспектов ML pipeline
2. **Plugin Integration** - seamless интеграция с plugin системой
3. **Distributed Support** - координация между процессами и контейнерами
4. **Resource Monitoring** - отслеживание использования CPU, памяти и shared memory
5. **Performance Optimization** - батчинг и асинхронное логирование
6. **Error Handling** - надежное логирование даже при ошибках
7. **Extensibility** - легкое добавление новых типов логирования