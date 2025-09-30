# Архитектура ядра NormaML

## Обзор архитектуры

NormaML построена на принципах модульности, расширяемости и производительности с поддержкой zero-copy операций через shared memory. Основная архитектура состоит из 5 ключевых компонентов и plugin-системы.

```
┌─────────────────────────────────────────────────────────────┐
│                     NormaML Core                            │
├─────────────────┬───────────────┬─────────────────────────────┤
│   DataManager   │ PluginManager │    ExecutionEngine         │
│   (SharedMem)   │               │    (Docker+SharedMem)      │
├─────────────────┼───────────────┼─────────────────────────────┤
│ ExperimentTracker │ ConfigManager │       EventBus           │
└─────────────────┴───────────────┴─────────────────────────────┘
                           ▲
                           │
┌─────────────────────────────────────────────────────────────┐
│                      Plugin Layer                          │
├─────────────┬─────────────┬─────────────┬─────────────────────┤
│ DataLoaders │FeatureEng  │ModelTrainers│    Evaluators      │
├─────────────┼─────────────┼─────────────┼─────────────────────┤
│  Backends   │ Transforms  │ Validators  │    Exporters       │
└─────────────┴─────────────┴─────────────┴─────────────────────┘
                           ▲
                           │
┌─────────────────────────────────────────────────────────────┐
│              Shared Memory Layer                           │
├─────────────┬─────────────┬─────────────┬─────────────────────┤
│ Arrow Buffers│Memory Maps │ POSIX SharedMem │ Container Volumes│
└─────────────┴─────────────┴─────────────┴─────────────────────┘
```

## Core Interfaces

### 1. DataManager Interface (с поддержкой Shared Memory)

```python
from abc import ABC, abstractmethod
from typing import Union, Dict, Any, Optional
import polars as pl
import pyarrow as pa
from multiprocessing import shared_memory
import mmap

class SharedMemoryHandle:
    """Дескриптор shared memory для zero-copy передачи данных"""
    def __init__(self, name: str, size: int, dtype: str, shape: tuple):
        self.name = name
        self.size = size
        self.dtype = dtype
        self.shape = shape
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'size': self.size,
            'dtype': self.dtype,
            'shape': self.shape
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SharedMemoryHandle':
        return cls(data['name'], data['size'], data['dtype'], data['shape'])

class DataManager(ABC):
    """Управление данными с zero-copy операциями через shared memory"""
    
    @abstractmethod
    def load_data(self, source: str, **kwargs) -> pl.LazyFrame:
        """Ленивая загрузка данных"""
        pass
    
    @abstractmethod
    def create_shared_memory_view(self, data: pl.DataFrame, 
                                 name: str) -> SharedMemoryHandle:
        """Создание shared memory представления данных"""
        pass
    
    @abstractmethod
    def load_from_shared_memory(self, handle: SharedMemoryHandle) -> pl.LazyFrame:
        """Загрузка данных из shared memory"""
        pass
    
    @abstractmethod
    def get_memory_usage(self) -> Dict[str, int]:
        """Получение информации о использовании памяти"""
        pass
    
    @abstractmethod
    def create_arrow_buffer(self, data: pl.DataFrame) -> pa.Buffer:
        """Создание Arrow buffer для межпроцессного обмена"""
        pass
    
    @abstractmethod
    def share_across_containers(self, data_id: str, 
                               container_ids: List[str]) -> Dict[str, Any]:
        """Подготовка данных для обмена между Docker контейнерами"""
        pass
```

### 2. Plugin Interface

```python
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

class Plugin(ABC):
    """Базовый интерфейс для всех plugins"""
    
    name: str
    version: str
    dependencies: List[str] = []
    supports_shared_memory: bool = True
    
    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> None:
        """Инициализация plugin"""
        pass
    
    @abstractmethod
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Валидация конфигурации"""
        pass
    
    @abstractmethod
    def get_capabilities(self) -> Dict[str, Any]:
        """Возвращает возможности plugin"""
        pass
    
    def setup_shared_memory(self, shm_config: Dict[str, Any]) -> None:
        """Настройка shared memory для plugin"""
        pass
    
    def cleanup(self) -> None:
        """Очистка ресурсов включая shared memory"""
        pass
```

### 3. DataLoader Plugin Interface

```python
class DataLoaderPlugin(Plugin):
    """Plugin для загрузки данных с поддержкой shared memory"""
    
    @abstractmethod
    def supports_source(self, source: str) -> bool:
        """Проверяет, поддерживает ли loader данный источник"""
        pass
    
    @abstractmethod
    def load(self, source: str, **kwargs) -> pl.LazyFrame:
        """Загружает данные из источника"""
        pass
    
    @abstractmethod
    def load_to_shared_memory(self, source: str, 
                             shm_name: str, **kwargs) -> SharedMemoryHandle:
        """Загружает данные напрямую в shared memory"""
        pass
    
    @abstractmethod
    def get_schema(self, source: str) -> Dict[str, str]:
        """Возвращает схему данных"""
        pass
    
    @abstractmethod
    def estimate_size(self, source: str) -> int:
        """Оценивает размер данных в байтах"""
        pass
    
    def supports_streaming(self) -> bool:
        """Поддерживает ли потоковую загрузку"""
        return False
```

### 4. FeatureEngineer Plugin Interface

```python
class FeatureEngineerPlugin(Plugin):
    """Plugin для feature engineering с поддержкой shared memory"""
    
    @abstractmethod
    def analyze_features(self, data_handle: Union[pl.LazyFrame, SharedMemoryHandle]) -> Dict[str, Any]:
        """Анализ признаков"""
        pass
    
    @abstractmethod
    def select_features(self, data_handle: Union[pl.LazyFrame, SharedMemoryHandle], 
                       target: str, **kwargs) -> List[str]:
        """Отбор признаков"""
        pass
    
    @abstractmethod
    def create_features(self, data_handle: Union[pl.LazyFrame, SharedMemoryHandle], 
                       config: Dict[str, Any]) -> Union[pl.LazyFrame, SharedMemoryHandle]:
        """Создание новых признаков"""
        pass
    
    @abstractmethod
    def get_feature_importance(self) -> Dict[str, float]:
        """Важность признаков"""
        pass
```

### 5. ModelTrainer Plugin Interface

```python
class ModelTrainerPlugin(Plugin):
    """Plugin для обучения моделей с поддержкой shared memory"""
    
    @abstractmethod
    def train(self, data_handle: Union[pl.LazyFrame, SharedMemoryHandle], 
              target: str, config: Dict[str, Any]) -> 'TrainedModel':
        """Обучение модели"""
        pass
    
    @abstractmethod
    def grid_search(self, data_handle: Union[pl.LazyFrame, SharedMemoryHandle], 
                   target: str, param_grid: Dict[str, List[Any]]) -> Dict[str, Any]:
        """Grid search для подбора гиперпараметров"""
        pass
    
    @abstractmethod
    def supports_task_type(self, task_type: str) -> bool:
        """Поддерживает ли trainer тип задачи"""
        pass
    
    @abstractmethod
    def get_default_params(self, task_type: str) -> Dict[str, Any]:
        """Параметры по умолчанию"""
        pass
    
    def supports_incremental_training(self) -> bool:
        """Поддерживает ли инкрементальное обучение"""
        return False
```

### 6. Evaluator Plugin Interface

```python
class EvaluatorPlugin(Plugin):
    """Plugin для оценки моделей с поддержкой shared memory"""
    
    @abstractmethod
    def evaluate(self, model: 'TrainedModel', 
                test_data_handle: Union[pl.LazyFrame, SharedMemoryHandle], 
                target: str) -> Dict[str, float]:
        """Оценка модели"""
        pass
    
    @abstractmethod
    def cross_validate(self, trainer: ModelTrainerPlugin,
                      data_handle: Union[pl.LazyFrame, SharedMemoryHandle], 
                      target: str, cv_config: Dict[str, Any]) -> Dict[str, Any]:
        """Кросс-валидация"""
        pass
    
    @abstractmethod
    def get_supported_metrics(self, task_type: str) -> List[str]:
        """Поддерживаемые метрики"""
        pass
```

## Core Components Implementation

### 1. SharedMemoryDataManager

```python
import polars as pl
import pyarrow as pa
import numpy as np
from multiprocessing import shared_memory
import tempfile
import os
from typing import Dict, Any, List

class SharedMemoryDataManager(DataManager):
    """Управление данными через shared memory"""
    
    def __init__(self):
        self.shared_blocks: Dict[str, shared_memory.SharedMemory] = {}
        self.data_registry: Dict[str, SharedMemoryHandle] = {}
        self.temp_dir = tempfile.mkdtemp(prefix="normaml_shm_")
    
    def load_data(self, source: str, **kwargs) -> pl.LazyFrame:
        """Ленивая загрузка данных"""
        return pl.scan_csv(source, **kwargs)
    
    def create_shared_memory_view(self, data: pl.DataFrame, 
                                 name: str) -> SharedMemoryHandle:
        """Создание shared memory представления данных"""
        # Конвертируем в Arrow для эффективного представления
        arrow_table = data.to_arrow()
        
        # Сериализуем в bytes
        sink = pa.BufferOutputStream()
        with pa.ipc.new_stream(sink, arrow_table.schema) as writer:
            writer.write_table(arrow_table)
        
        buffer_bytes = sink.getvalue().to_pybytes()
        
        # Создаем shared memory
        shm = shared_memory.SharedMemory(
            create=True, 
            size=len(buffer_bytes),
            name=f"normaml_{name}"
        )
        
        # Копируем данные в shared memory
        shm.buf[:len(buffer_bytes)] = buffer_bytes
        
        # Сохраняем ссылку
        self.shared_blocks[name] = shm
        
        handle = SharedMemoryHandle(
            name=shm.name,
            size=len(buffer_bytes),
            dtype="arrow_ipc",
            shape=arrow_table.shape
        )
        
        self.data_registry[name] = handle
        return handle
    
    def load_from_shared_memory(self, handle: SharedMemoryHandle) -> pl.LazyFrame:
        """Загрузка данных из shared memory"""
        # Подключаемся к существующему shared memory
        shm = shared_memory.SharedMemory(name=handle.name)
        
        # Читаем данные как Arrow IPC
        buffer = pa.py_buffer(shm.buf[:handle.size])
        reader = pa.ipc.open_stream(buffer)
        arrow_table = reader.read_all()
        
        # Конвертируем в Polars LazyFrame
        return pl.from_arrow(arrow_table).lazy()
    
    def create_arrow_buffer(self, data: pl.DataFrame) -> pa.Buffer:
        """Создание Arrow buffer для межпроцессного обмена"""
        arrow_table = data.to_arrow()
        sink = pa.BufferOutputStream()
        with pa.ipc.new_stream(sink, arrow_table.schema) as writer:
            writer.write_table(arrow_table)
        return sink.getvalue()
    
    def share_across_containers(self, data_id: str, 
                               container_ids: List[str]) -> Dict[str, Any]:
        """Подготовка данных для обмена между Docker контейнерами"""
        if data_id not in self.data_registry:
            raise ValueError(f"Data {data_id} not found in registry")
        
        handle = self.data_registry[data_id]
        
        # Создаем memory-mapped файл для контейнеров
        mmap_path = os.path.join(self.temp_dir, f"{data_id}.mmap")
        
        # Копируем данные из shared memory в mmap файл
        shm = self.shared_blocks[data_id]
        with open(mmap_path, 'wb') as f:
            f.write(shm.buf[:handle.size])
        
        return {
            'type': 'memory_mapped_file',
            'path': mmap_path,
            'handle': handle.to_dict(),
            'volume_mount': f"{mmap_path}:/shared_data/{data_id}.mmap:ro",
            'container_ids': container_ids
        }
    
    def get_memory_usage(self) -> Dict[str, int]:
        """Получение информации о использовании памяти"""
        total_size = sum(handle.size for handle in self.data_registry.values())
        return {
            'total_shared_memory': total_size,
            'active_blocks': len(self.shared_blocks),
            'registered_datasets': len(self.data_registry)
        }
    
    def cleanup(self):
        """Очистка всех shared memory блоков"""
        for shm in self.shared_blocks.values():
            try:
                shm.close()
                shm.unlink()
            except FileNotFoundError:
                pass  # Уже удален
        
        self.shared_blocks.clear()
        self.data_registry.clear()
        
        # Очистка temporary files
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
```

### 2. ExecutionEngine с Docker + Shared Memory

```python
import asyncio
import docker
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Callable, Any, List, Dict, Optional

class ExecutionEngine:
    """Управление параллельным выполнением с поддержкой Docker и shared memory"""
    
    def __init__(self, max_workers: int = None):
        self.process_pool = ProcessPoolExecutor(max_workers)
        self.thread_pool = ThreadPoolExecutor(max_workers)
        self.docker_client = docker.from_env()
        self.data_manager: Optional[SharedMemoryDataManager] = None
        self.active_containers: Dict[str, docker.models.containers.Container] = {}
    
    def set_data_manager(self, data_manager: SharedMemoryDataManager):
        """Устанавливает data manager для работы с shared memory"""
        self.data_manager = data_manager
    
    async def execute_parallel(self, tasks: List[Callable], 
                              execution_type: str = "thread") -> List[Any]:
        """Параллельное выполнение задач"""
        if execution_type == "process":
            loop = asyncio.get_event_loop()
            futures = [
                loop.run_in_executor(self.process_pool, task)
                for task in tasks
            ]
        elif execution_type == "thread":
            loop = asyncio.get_event_loop()
            futures = [
                loop.run_in_executor(self.thread_pool, task)
                for task in tasks
            ]
        else:
            raise ValueError(f"Unknown execution type: {execution_type}")
        
        return await asyncio.gather(*futures)
    
    def execute_plugin_in_docker(self, plugin_name: str, image: str, 
                                command: str, data_id: str = None,
                                environment: Dict[str, str] = None) -> Dict[str, Any]:
        """Выполнение plugin в Docker контейнере с shared memory"""
        
        volumes = {}
        shared_data_info = None
        
        if data_id and self.data_manager:
            # Подготавливаем shared memory для контейнера
            container_id = f"{plugin_name}_{data_id}"
            shared_data_info = self.data_manager.share_across_containers(
                data_id, [container_id]
            )
            
            # Добавляем volume mount для shared data
            volumes[shared_data_info['path']] = {
                'bind': f'/shared_data/{data_id}.mmap',
                'mode': 'ro'
            }
        
        # Добавляем volume для /dev/shm (shared memory)
        volumes['/dev/shm'] = {'bind': '/dev/shm', 'mode': 'rw'}
        
        container_config = {
            'image': image,
            'command': command,
            'volumes': volumes,
            'environment': environment or {},
            'detach': True,
            'remove': True,
            'shm_size': '2g'  # Увеличиваем размер shared memory
        }
        
        if shared_data_info:
            container_config['environment']['SHARED_DATA_PATH'] = f'/shared_data/{data_id}.mmap'
            container_config['environment']['SHARED_DATA_HANDLE'] = str(shared_data_info['handle'])
        
        try:
            container = self.docker_client.containers.run(**container_config)
            self.active_containers[f"{plugin_name}_{data_id}"] = container
            
            # Ждем завершения
            result = container.wait()
            logs = container.logs().decode('utf-8')
            
            return {
                'exit_code': result['StatusCode'],
                'logs': logs,
                'shared_data_info': shared_data_info
            }
        
        except Exception as e:
            return {
                'exit_code': -1,
                'error': str(e),
                'shared_data_info': shared_data_info
            }
    
    def cleanup_containers(self):
        """Остановка и очистка всех активных контейнеров"""
        for container_id, container in self.active_containers.items():
            try:
                container.stop(timeout=10)
                container.remove()
            except:
                pass
        self.active_containers.clear()
```

### 3. ExperimentTracker

```python
import mlflow
from typing import Dict, Any, Optional

class ExperimentTracker:
    """Интеграция с MLflow с поддержкой shared memory метрик"""
    
    def __init__(self, experiment_name: str, tracking_uri: str = None):
        self.experiment_name = experiment_name
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        self.current_run_id = None
    
    def start_run(self, run_name: str = None) -> str:
        """Начало эксперимента"""
        run = mlflow.start_run(run_name=run_name)
        self.current_run_id = run.info.run_id
        return self.current_run_id
    
    def log_params(self, params: Dict[str, Any]) -> None:
        """Логирование параметров"""
        mlflow.log_params(params)
    
    def log_metrics(self, metrics: Dict[str, float], step: int = None) -> None:
        """Логирование метрик"""
        mlflow.log_metrics(metrics, step)
    
    def log_shared_memory_usage(self, memory_info: Dict[str, int]) -> None:
        """Логирование использования shared memory"""
        mlflow.log_metrics({
            f"memory_{key}": value for key, value in memory_info.items()
        })
    
    def log_model(self, model: Any, artifact_path: str, 
                  registered_model_name: str = None) -> None:
        """Логирование модели"""
        mlflow.log_model(model, artifact_path, 
                        registered_model_name=registered_model_name)
    
    def end_run(self) -> None:
        """Завершение эксперимента"""
        mlflow.end_run()
        self.current_run_id = None
```

## Zero-Copy Data Flow

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Data Source   │───▶│  SharedMemory    │───▶│  Docker         │
│   (CSV/Parquet) │    │  Buffer          │    │  Container      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │  Memory Mapped   │
                       │  File            │
                       └──────────────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │  Plugin Process  │
                       │  (Zero Copy)     │
                       └──────────────────┘
```

## Component Lifecycle

1. **Initialization Phase**:
   - ConfigManager загружает конфигурацию
   - SharedMemoryDataManager инициализирует shared memory пулы
   - PluginManager обнаруживает и регистрирует plugins
   - ExecutionEngine настраивает Docker client и pools
   - ExperimentTracker подключается к MLflow

2. **Runtime Phase**:
   - Данные загружаются в shared memory (zero-copy)
   - Plugins выполняются в Docker контейнерах с доступом к shared data
   - Memory-mapped файлы используются для передачи данных между контейнерами
   - Все операции логируются в MLflow включая использование памяти

3. **Cleanup Phase**:
   - Plugins вызывают cleanup()
   - ExecutionEngine останавливает все контейнеры
   - SharedMemoryDataManager очищает все shared memory блоки
   - ExperimentTracker завершает активные runs