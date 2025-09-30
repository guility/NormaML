# API Спецификация и примеры использования NormaML

## Публичный API

### Основные классы и функции

```python
from normaml import (
    # Core components
    NormaML,
    Pipeline,
    DataManager,
    ExperimentTracker,
    
    # Plugin factories
    PluginFactory,
    
    # Configuration
    Config,
    RunConfig,
    
    # Utilities
    auto_plugin_discovery,
    create_pipeline_from_config,
)

# Type hints
from normaml.core.interfaces import (
    DataLoaderPlugin,
    FeatureEngineerPlugin, 
    ModelTrainerPlugin,
    EvaluatorPlugin,
)
```

### Основной класс NormaML

```python
class NormaML:
    """
    Главный класс для работы с NormaML библиотекой.
    
    Provides unified interface for:
    - Plugin management
    - Data processing with zero-copy
    - Parallel execution
    - Experiment tracking
    """
    
    def __init__(self, 
                 config: Optional[Config] = None,
                 tracking_uri: Optional[str] = None,
                 enable_auto_logging: bool = True,
                 max_workers: int = None):
        """
        Initialize NormaML instance.
        
        Args:
            config: Configuration object or path to config file
            tracking_uri: MLflow tracking server URI
            enable_auto_logging: Enable automatic experiment logging
            max_workers: Maximum number of parallel workers
        """
        
    def load_data(self, source: str, 
                  loader: str = "auto", 
                  **kwargs) -> str:
        """
        Load data from source.
        
        Args:
            source: Data source path/URL
            loader: Plugin name or "auto" for automatic detection
            **kwargs: Additional arguments for data loader
            
        Returns:
            Data ID for zero-copy operations
        """
        
    def engineer_features(self, data_id: str,
                         engineer: str = "auto",
                         **kwargs) -> str:
        """
        Perform feature engineering.
        
        Args:
            data_id: Input data ID
            engineer: Feature engineer plugin name
            **kwargs: Engineer configuration
            
        Returns:
            Transformed data ID
        """
        
    def train_model(self, data_id: str,
                   target_column: str,
                   trainer: str = "auto",
                   **kwargs) -> 'TrainedModel':
        """
        Train machine learning model.
        
        Args:
            data_id: Training data ID
            target_column: Target variable name
            trainer: Model trainer plugin name
            **kwargs: Training configuration
            
        Returns:
            Trained model object
        """
        
    def evaluate_model(self, model: 'TrainedModel',
                      test_data_id: str,
                      target_column: str,
                      evaluator: str = "auto",
                      **kwargs) -> Dict[str, float]:
        """
        Evaluate trained model.
        
        Args:
            model: Trained model
            test_data_id: Test data ID
            target_column: Target variable name
            evaluator: Evaluator plugin name
            **kwargs: Evaluation configuration
            
        Returns:
            Evaluation metrics
        """
        
    def create_pipeline(self, steps: List[Dict[str, Any]]) -> 'Pipeline':
        """
        Create ML pipeline from step definitions.
        
        Args:
            steps: List of pipeline steps
            
        Returns:
            Pipeline object
        """
        
    def run_experiment(self, experiment_config: Dict[str, Any]) -> str:
        """
        Run complete ML experiment.
        
        Args:
            experiment_config: Experiment configuration
            
        Returns:
            MLflow run ID
        """
```

## Базовые примеры использования

### 1. Простой пример классификации

```python
import normaml

# Инициализация
ml = normaml.NormaML(
    tracking_uri="http://localhost:5000",
    enable_auto_logging=True
)

# Загрузка данных
data_id = ml.load_data("dataset.csv")

# Feature engineering
processed_data_id = ml.engineer_features(
    data_id,
    engineer="sklearn",
    feature_selection=True,
    selection_method="mutual_info",
    k_best=10
)

# Обучение модели
model = ml.train_model(
    processed_data_id,
    target_column="target",
    trainer="sklearn_classifier",
    algorithm="random_forest",
    n_estimators=100,
    random_state=42
)

# Оценка модели
metrics = ml.evaluate_model(
    model,
    processed_data_id,  # Используем те же данные для простоты
    target_column="target",
    evaluator="classification",
    metrics=["accuracy", "f1_score", "roc_auc"]
)

print(f"Model performance: {metrics}")
```

### 2. Pipeline-based подход

```python
import normaml

# Определение pipeline
pipeline_config = {
    "steps": [
        {
            "name": "data_loading",
            "plugin": "csv_loader",
            "config": {
                "source": "large_dataset.csv",
                "chunk_size": 10000
            }
        },
        {
            "name": "feature_engineering", 
            "plugin": "polars_engineer",
            "config": {
                "operations": [
                    {"type": "drop_nulls"},
                    {"type": "normalize", "columns": ["numeric_features"]},
                    {"type": "encode_categorical", "columns": ["cat_features"]}
                ]
            }
        },
        {
            "name": "feature_selection",
            "plugin": "sklearn_feature_selector", 
            "config": {
                "method": "mutual_info",
                "k_best": 15
            }
        },
        {
            "name": "model_training",
            "plugin": "xgboost_classifier",
            "config": {
                "n_estimators": 200,
                "max_depth": 6,
                "learning_rate": 0.1
            }
        },
        {
            "name": "model_evaluation",
            "plugin": "classification_evaluator",
            "config": {
                "metrics": ["accuracy", "precision", "recall", "f1_score"],
                "cross_validation": {"cv": 5, "scoring": "f1_macro"}
            }
        }
    ],
    "target_column": "target",
    "experiment_name": "xgboost_classification_pipeline"
}

# Создание и выполнение pipeline
ml = normaml.NormaML()
pipeline = ml.create_pipeline(pipeline_config["steps"])

# Запуск с автоматическим логированием
run_id = ml.run_experiment(pipeline_config)
print(f"Experiment completed. MLflow run ID: {run_id}")
```

### 3. Работа с большими данными (Zero-Copy)

```python
import normaml
from normaml.data import StreamingProcessor

# Инициализация с shared memory support
ml = normaml.NormaML(
    config={
        "data_manager": {
            "max_memory_mb": 8192,
            "enable_shared_memory": True,
            "enable_memory_mapping": True
        }
    }
)

# Загрузка большого датасета
large_data_id = ml.load_data(
    "huge_dataset.parquet",
    loader="parquet_loader",
    strategy="memory_mapped"  # Автоматически выберется для больших файлов
)

# Streaming feature engineering
streaming_processor = StreamingProcessor(ml.data_manager)

def feature_transform(chunk):
    return chunk.select([
        "important_feature_1",
        "important_feature_2", 
        "target"
    ]).drop_nulls()

# Обработка потоком
processed_data_id = streaming_processor.process_streaming(
    large_data_id,
    feature_transform,
    output_strategy="shared_memory"
)

# Обучение модели с shared memory
model = ml.train_model(
    processed_data_id,
    target_column="target",
    trainer="sklearn_classifier",
    algorithm="sgd",  # Подходит для больших данных
    use_shared_memory=True
)

# Получение статистики памяти
memory_stats = ml.data_manager.get_memory_stats()
print(f"Memory usage: {memory_stats}")
```

### 4. Distributed execution с Docker

```python
import normaml

# Конфигурация для distributed execution
config = normaml.Config({
    "execution": {
        "default_mode": "docker",
        "max_concurrent_tasks": 5,
        "docker": {
            "registry": "normaml",
            "shared_memory_size": "4g"
        }
    },
    "plugins": {
        "model_trainers": {
            "xgboost_trainer": {
                "docker_image": "normaml/xgboost:latest",
                "resource_limits": {
                    "cpu_cores": 4,
                    "memory_mb": 4096
                }
            }
        }
    }
})

ml = normaml.NormaML(config=config)

# Grid search с distributed execution
grid_search_config = {
    "data_source": "training_data.csv",
    "target_column": "target", 
    "trainer": "xgboost_trainer",
    "param_grid": {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 6, 9],
        "learning_rate": [0.01, 0.1, 0.2]
    },
    "cv_folds": 5,
    "scoring": "f1_macro",
    "parallel_jobs": 3  # Количество Docker контейнеров
}

# Запуск distributed grid search
best_model, best_params, cv_results = ml.grid_search(grid_search_config)

print(f"Best parameters: {best_params}")
print(f"Best CV score: {cv_results['best_score']}")
```

### 5. Custom Plugin разработка

```python
from normaml.core.interfaces import FeatureEngineerPlugin
from normaml.plugins.base import BasePlugin
import polars as pl
from typing import Dict, Any, List

class CustomFeatureEngineer(FeatureEngineerPlugin, BasePlugin):
    """Custom feature engineering plugin with domain-specific logic."""
    
    name = "custom_feature_engineer"
    version = "1.0.0"
    capabilities = ["feature_creation", "feature_selection", "domain_specific"]
    dependencies = ["polars>=0.19.0", "scikit-learn>=1.3.0"]
    
    def __init__(self):
        super().__init__()
        self.created_features = []
    
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize with domain-specific configuration."""
        self.domain = config.get("domain", "general")
        self.feature_types = config.get("feature_types", ["numerical", "categorical"])
        
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate configuration."""
        required_keys = ["domain"]
        return all(key in config for key in required_keys)
    
    def analyze_features(self, data: pl.LazyFrame) -> Dict[str, Any]:
        """Analyze features for domain-specific insights."""
        schema = data.collect_schema()
        
        analysis = {
            "total_features": len(schema),
            "feature_types": {name: str(dtype) for name, dtype in schema.items()},
            "suggested_operations": self._suggest_operations(schema)
        }
        
        return analysis
    
    def create_features(self, data: pl.LazyFrame, 
                       config: Dict[str, Any]) -> pl.LazyFrame:
        """Create domain-specific features."""
        
        if self.domain == "financial":
            # Financial domain features
            data = self._create_financial_features(data)
        elif self.domain == "text":
            # Text domain features  
            data = self._create_text_features(data)
        else:
            # General features
            data = self._create_general_features(data)
            
        return data
    
    def select_features(self, data: pl.LazyFrame, 
                       target: str, **kwargs) -> List[str]:
        """Select best features using domain knowledge."""
        from sklearn.feature_selection import SelectKBest, mutual_info_classif
        
        # Collect data for sklearn
        df = data.collect()
        X = df.drop(target).to_pandas()
        y = df.select(target).to_pandas().iloc[:, 0]
        
        # Feature selection
        k_best = kwargs.get("k_best", 10)
        selector = SelectKBest(mutual_info_classif, k=k_best)
        selector.fit(X, y)
        
        selected_features = X.columns[selector.get_support()].tolist()
        return selected_features
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Return feature importance scores."""
        # Implementation depends on the method used
        return {}
    
    def _suggest_operations(self, schema: Dict[str, Any]) -> List[str]:
        """Suggest operations based on schema analysis."""
        operations = []
        
        for name, dtype in schema.items():
            if "Int" in str(dtype) or "Float" in str(dtype):
                operations.append(f"normalize_{name}")
            elif "Utf8" in str(dtype):
                operations.append(f"encode_{name}")
                
        return operations
    
    def _create_financial_features(self, data: pl.LazyFrame) -> pl.LazyFrame:
        """Create financial domain-specific features."""
        return data.with_columns([
            # Example: moving averages, ratios, etc.
            pl.col("price").rolling_mean(window_size=7).alias("price_ma_7d"),
            pl.col("price").rolling_mean(window_size=30).alias("price_ma_30d"),
            (pl.col("revenue") / pl.col("expenses")).alias("profit_ratio"),
        ])
    
    def _create_text_features(self, data: pl.LazyFrame) -> pl.LazyFrame:
        """Create text domain-specific features."""
        return data.with_columns([
            # Example: text length, word count, etc.
            pl.col("text").str.lengths().alias("text_length"),
            pl.col("text").str.count_match(r'\w+').alias("word_count"),
        ])
    
    def _create_general_features(self, data: pl.LazyFrame) -> pl.LazyFrame:
        """Create general features."""
        return data.with_columns([
            # Example: interaction features, polynomials
            (pl.col("feature1") * pl.col("feature2")).alias("feature1_x_feature2"),
        ])

# Регистрация plugin
from normaml.core.plugin_manager import plugin_registry

@plugin_registry.register_feature_engineer(
    name="custom_feature_engineer",
    version="1.0.0",
    capabilities=["feature_creation", "feature_selection", "domain_specific"]
)
class RegisteredCustomFeatureEngineer(CustomFeatureEngineer):
    pass

# Использование custom plugin
ml = normaml.NormaML()

# Plugin будет автоматически обнаружен
processed_data_id = ml.engineer_features(
    data_id,
    engineer="custom_feature_engineer",
    domain="financial",
    feature_types=["numerical"],
    k_best=15
)
```

### 6. Hyperparameter tuning с MLflow

```python
import normaml
from normaml.tracking import RunConfig, RunType

# Конфигурация эксперимента
experiment_config = {
    "name": "hyperparameter_optimization",
    "data_source": "dataset.csv",
    "target_column": "target",
    "algorithms": ["xgboost", "sklearn_rf", "sklearn_svm"],
    "optimization": {
        "method": "bayesian",  # grid, random, bayesian
        "n_trials": 100,
        "parallel_trials": 4
    }
}

ml = normaml.NormaML(
    tracking_uri="http://localhost:5000",
    enable_auto_logging=True
)

# Создание главного эксперимента
main_run_config = RunConfig(
    experiment_name=experiment_config["name"],
    run_name="hyperparameter_optimization_main",
    run_type=RunType.HYPERPARAMETER_TUNING,
    tags={
        "optimization_method": experiment_config["optimization"]["method"],
        "n_trials": str(experiment_config["optimization"]["n_trials"])
    }
)

main_run_id = ml.experiment_tracker.start_run(main_run_config)

try:
    # Загрузка и подготовка данных
    data_id = ml.load_data(experiment_config["data_source"])
    processed_data_id = ml.engineer_features(data_id, engineer="sklearn")
    
    best_results = {}
    
    # Оптимизация для каждого алгоритма
    for algorithm in experiment_config["algorithms"]:
        
        # Nested run для каждого алгоритма
        algo_run_config = RunConfig(
            experiment_name=f"{experiment_config['name']}_{algorithm}",
            run_name=f"{algorithm}_optimization",
            run_type=RunType.HYPERPARAMETER_TUNING,
            parent_run_id=main_run_id,
            nested=True,
            tags={"algorithm": algorithm}
        )
        
        algo_run_id = ml.experiment_tracker.start_run(algo_run_config)
        
        # Определение пространства параметров
        param_space = ml.get_param_space(algorithm)
        
        # Bayesian optimization
        best_params, best_score = ml.optimize_hyperparameters(
            data_id=processed_data_id,
            target_column=experiment_config["target_column"],
            trainer=algorithm,
            param_space=param_space,
            optimization_config=experiment_config["optimization"],
            run_id=algo_run_id
        )
        
        best_results[algorithm] = {
            "params": best_params,
            "score": best_score,
            "run_id": algo_run_id
        }
        
        # Логирование лучших результатов
        ml.experiment_tracker.log_params({
            f"best_{algorithm}_params": str(best_params)
        }, main_run_id)
        
        ml.experiment_tracker.log_metrics({
            f"best_{algorithm}_score": best_score
        }, main_run_id)
        
        ml.experiment_tracker.end_run(algo_run_id)
    
    # Выбор лучшего алгоритма
    best_algorithm = max(best_results.keys(), 
                        key=lambda k: best_results[k]["score"])
    
    ml.experiment_tracker.log_params({
        "best_algorithm": best_algorithm,
        "best_overall_score": best_results[best_algorithm]["score"]
    }, main_run_id)
    
    print(f"Best algorithm: {best_algorithm}")
    print(f"Best score: {best_results[best_algorithm]['score']}")
    print(f"Best params: {best_results[best_algorithm]['params']}")

finally:
    ml.experiment_tracker.end_run(main_run_id)
```

### 7. Конфигурационный подход

```python
# config.yaml
experiment:
  name: "production_model_training"
  description: "Production model with optimized features"
  
data:
  source: "s3://my-bucket/production-data.parquet"
  loader: "parquet_loader"
  preprocessing:
    drop_nulls: true
    normalize_numeric: true
    encode_categorical: true
    
features:
  engineer: "sklearn_engineer"
  selection:
    method: "mutual_info"
    k_best: 20
  creation:
    polynomial_features: true
    interaction_features: true
    
models:
  - name: "primary_model"
    trainer: "xgboost_classifier"
    params:
      n_estimators: 500
      max_depth: 8
      learning_rate: 0.05
      subsample: 0.8
      colsample_bytree: 0.8
      
  - name: "backup_model"  
    trainer: "sklearn_classifier"
    params:
      algorithm: "random_forest"
      n_estimators: 300
      max_depth: 10
      
evaluation:
  evaluator: "classification_evaluator"
  metrics: ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
  cross_validation:
    cv: 5
    scoring: "f1_macro"
    
execution:
  mode: "docker"
  parallel_jobs: 3
  resource_limits:
    cpu_cores: 4
    memory_mb: 8192
    
tracking:
  mlflow_uri: "http://mlflow-server:5000"
  auto_log: true
  log_artifacts: true
  register_best_model: true

# Python код
import normaml

# Загрузка конфигурации
config = normaml.Config.from_file("config.yaml")
ml = normaml.NormaML(config=config)

# Создание pipeline из конфигурации
pipeline = normaml.create_pipeline_from_config(config)

# Запуск эксперимента
results = pipeline.run()

print(f"Experiment completed successfully!")
print(f"Results: {results}")
```

### 8. Advanced API для экспертов

```python
import normaml
from normaml.core import (
    PluginManager, 
    DataManager, 
    ExecutionEngine,
    ExperimentTracker
)
from normaml.execution import TaskScheduler, TaskDefinition

# Продвинутая настройка компонентов
data_manager = DataManager(
    max_memory_mb=16384,
    enable_shared_memory=True,
    enable_memory_mapping=True,
    cleanup_timeout=3600
)

plugin_manager = PluginManager()
plugin_manager.discover_plugins()

execution_engine = ExecutionEngine(
    max_workers=8,
    default_mode="docker",
    docker_registry="my-registry"
)

experiment_tracker = ExperimentTracker(
    tracking_uri="http://mlflow:5000",
    enable_auto_logging=True,
    batch_size=50
)

# Создание scheduler для complex workflows
scheduler = TaskScheduler(max_concurrent_tasks=5)
scheduler.register_execution_engines(
    async_engine=execution_engine.async_engine,
    process_engine=execution_engine.process_engine,
    docker_engine=execution_engine.docker_engine
)

# Определение complex task dependencies
tasks = [
    TaskDefinition(
        id="data_loading",
        plugin_name="parquet_loader",
        plugin_method="load",
        plugin_config={"source": "data.parquet"},
        execution_mode="async",
        resources={"memory_mb": 2048}
    ),
    TaskDefinition(
        id="feature_engineering_1",
        plugin_name="polars_engineer", 
        plugin_method="create_features",
        dependencies=["data_loading"],
        execution_mode="process",
        resources={"cpu_cores": 2, "memory_mb": 4096}
    ),
    TaskDefinition(
        id="feature_engineering_2",
        plugin_name="sklearn_engineer",
        plugin_method="select_features", 
        dependencies=["feature_engineering_1"],
        execution_mode="docker",
        docker_image="normaml/sklearn:latest",
        resources={"cpu_cores": 4, "memory_mb": 8192}
    ),
    TaskDefinition(
        id="model_training",
        plugin_name="xgboost_trainer",
        plugin_method="train",
        dependencies=["feature_engineering_2"],
        execution_mode="docker",
        docker_image="normaml/xgboost:latest",
        resources={"cpu_cores": 8, "memory_mb": 16384}
    )
]

# Запуск scheduler
import asyncio

async def run_complex_workflow():
    await scheduler.start_scheduler()
    
    # Отправка задач
    task_ids = []
    for task in tasks:
        task_id = scheduler.submit_task(task)
        task_ids.append(task_id)
    
    # Мониторинг выполнения
    while True:
        status = scheduler.get_queue_status()
        print(f"Queue status: {status}")
        
        if status['active_tasks'] == 0 and status['queue_size'] == 0:
            break
            
        await asyncio.sleep(5)
    
    await scheduler.stop_scheduler()
    
    # Получение результатов
    results = {}
    for task_id in task_ids:
        task_status = scheduler.get_task_status(task_id)
        results[task_id] = task_status
    
    return results

# Запуск workflow
results = asyncio.run(run_complex_workflow())
print(f"Workflow completed: {results}")
```

## API Reference Summary

### Core Classes

- **`NormaML`** - Главный класс для работы с библиотекой
- **`Pipeline`** - Класс для создания ML pipelines
- **`Config`** - Конфигурационный класс
- **`DataManager`** - Управление данными с zero-copy
- **`PluginManager`** - Управление plugins
- **`ExecutionEngine`** - Параллельное выполнение
- **`ExperimentTracker`** - Интеграция с MLflow

### Plugin Interfaces

- **`DataLoaderPlugin`** - Интерфейс для загрузчиков данных
- **`FeatureEngineerPlugin`** - Интерфейс для feature engineering
- **`ModelTrainerPlugin`** - Интерфейс для обучения моделей
- **`EvaluatorPlugin`** - Интерфейс для оценки моделей

### Utility Functions

- **`auto_plugin_discovery()`** - Автоматическое обнаружение plugins
- **`create_pipeline_from_config()`** - Создание pipeline из конфигурации
- **`optimize_hyperparameters()`** - Оптимизация гиперпараметров

### Configuration Options

- **Data Management**: memory limits, shared memory, caching
- **Execution**: parallel modes, resource limits, Docker settings
- **Tracking**: MLflow integration, logging levels
- **Plugins**: discovery methods, custom plugins

Эта API спецификация обеспечивает:

1. **Простоту использования** - минимальный код для базовых задач
2. **Гибкость** - продвинутые возможности для экспертов
3. **Модульность** - независимые компоненты
4. **Производительность** - zero-copy операции и параллелизм
5. **Отслеживаемость** - полная интеграция с MLflow
6. **Расширяемость** - простое создание custom plugins
7. **Конфигурируемость** - декларативный подход через YAML/JSON
8. **Production-ready** - Docker support и resource management