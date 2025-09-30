# Архитектура параллельного выполнения и Docker интеграции

## Обзор системы параллельного выполнения

Система параллельного выполнения NormaML обеспечивает эффективное использование ресурсов через комбинацию процессов, потоков, асинхронного выполнения и Docker контейнеров с поддержкой shared memory для zero-copy операций.

```
┌─────────────────────────────────────────────────────────────┐
│                 Execution Orchestrator                     │
├─────────────────┬───────────────┬─────────────────────────────┤
│  Task Scheduler │ Resource Mgr  │    Load Balancer           │
│  (Queue-based)  │ (CPU/Memory)  │    (Dynamic)               │
└─────────────────┴───────────────┴─────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                 Execution Engines                          │
├─────────────────┬───────────────┬─────────────────────────────┤
│   AsyncIO       │ ProcessPool   │    Docker Containers       │
│   Engine        │ Engine        │    Engine                  │
├─────────────────┼───────────────┼─────────────────────────────┤
│ • Fast I/O      │ • CPU-bound   │ • Isolation                │
│ • Networking    │ • Parallel    │ • Reproducibility          │
│ • Orchestration │ • Multicore   │ • Resource Limits          │
└─────────────────┴───────────────┴─────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              Shared Memory Integration                      │
├─────────────────┬───────────────┬─────────────────────────────┤
│  Memory Pools   │ Cross-Process │    Container Volumes       │
│  (Per Engine)   │  Sharing      │    (Bind Mounts)           │
├─────────────────┼───────────────┼─────────────────────────────┤
│ • POSIX SHM     │ • IPC Queues  │ • /dev/shm mounting        │
│ • Memory Maps   │ • Named Pipes │ • Volume sharing           │
│ • Arrow Buffers │ • Unix Sockets│ • Network sharing          │
└─────────────────┴───────────────┴─────────────────────────────┘
```

## Task Scheduling System

### 1. Task Definition

```python
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import uuid
from datetime import datetime

class TaskType(Enum):
    DATA_LOADING = "data_loading"
    FEATURE_ENGINEERING = "feature_engineering"
    MODEL_TRAINING = "model_training"
    MODEL_EVALUATION = "model_evaluation"
    HYPERPARAMETER_TUNING = "hyperparameter_tuning"
    PIPELINE_EXECUTION = "pipeline_execution"

class TaskPriority(Enum):
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4

class ExecutionMode(Enum):
    ASYNC = "async"
    THREAD = "thread"
    PROCESS = "process"
    DOCKER = "docker"

@dataclass
class ResourceRequirements:
    cpu_cores: Optional[int] = None
    memory_mb: Optional[int] = None
    gpu_count: Optional[int] = None
    disk_space_mb: Optional[int] = None
    max_execution_time: Optional[int] = None  # seconds
    
@dataclass
class TaskDefinition:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    task_type: TaskType = TaskType.PIPELINE_EXECUTION
    priority: TaskPriority = TaskPriority.NORMAL
    execution_mode: ExecutionMode = ExecutionMode.PROCESS
    
    # Plugin information
    plugin_name: str = ""
    plugin_method: str = ""
    plugin_config: Dict[str, Any] = field(default_factory=dict)
    
    # Data references (for zero-copy)
    input_data_ids: List[str] = field(default_factory=list)
    output_data_id: Optional[str] = None
    shared_memory_config: Dict[str, Any] = field(default_factory=dict)
    
    # Resource requirements
    resources: ResourceRequirements = field(default_factory=ResourceRequirements)
    
    # Docker specific
    docker_image: Optional[str] = None
    docker_volumes: Dict[str, str] = field(default_factory=dict)
    docker_environment: Dict[str, str] = field(default_factory=dict)
    
    # Dependencies
    dependencies: List[str] = field(default_factory=list)
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    scheduled_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Retry configuration
    max_retries: int = 3
    retry_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'task_type': self.task_type.value,
            'priority': self.priority.value,
            'execution_mode': self.execution_mode.value,
            'plugin_name': self.plugin_name,
            'plugin_method': self.plugin_method,
            'plugin_config': self.plugin_config,
            'input_data_ids': self.input_data_ids,
            'output_data_id': self.output_data_id,
            'shared_memory_config': self.shared_memory_config,
            'resources': self.resources.__dict__,
            'docker_image': self.docker_image,
            'docker_volumes': self.docker_volumes,
            'docker_environment': self.docker_environment,
            'dependencies': self.dependencies,
            'max_retries': self.max_retries,
            'retry_count': self.retry_count
        }

@dataclass
class TaskResult:
    task_id: str
    success: bool
    result_data: Any = None
    error_message: Optional[str] = None
    execution_time: float = 0.0
    memory_used: int = 0
    cpu_time: float = 0.0
    output_data_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
```

### 2. Task Scheduler

```python
import heapq
import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import Dict, List, Set, Callable
import threading
import time

class TaskScheduler:
    """Планировщик задач с поддержкой приоритетов и зависимостей"""
    
    def __init__(self, max_concurrent_tasks: int = 10):
        self.max_concurrent_tasks = max_concurrent_tasks
        
        # Priority queue для задач
        self.task_queue: List[tuple] = []  # (priority, timestamp, task)
        self.queue_lock = threading.Lock()
        
        # Активные задачи
        self.active_tasks: Dict[str, TaskDefinition] = {}
        self.completed_tasks: Dict[str, TaskResult] = {}
        self.failed_tasks: Dict[str, TaskResult] = {}
        
        # Dependency tracking
        self.dependency_graph: Dict[str, Set[str]] = {}
        self.dependent_tasks: Dict[str, Set[str]] = {}
        
        # Execution engines
        self.async_engine: Optional['AsyncExecutionEngine'] = None
        self.process_engine: Optional['ProcessExecutionEngine'] = None
        self.docker_engine: Optional['DockerExecutionEngine'] = None
        
        # Scheduling state
        self.running = False
        self.scheduler_task: Optional[asyncio.Task] = None
        
        # Callbacks
        self.task_callbacks: Dict[str, List[Callable]] = {
            'on_task_start': [],
            'on_task_complete': [],
            'on_task_error': [],
            'on_task_retry': []
        }
    
    def register_execution_engines(self, 
                                 async_engine: 'AsyncExecutionEngine',
                                 process_engine: 'ProcessExecutionEngine',
                                 docker_engine: 'DockerExecutionEngine'):
        """Регистрация execution engines"""
        self.async_engine = async_engine
        self.process_engine = process_engine
        self.docker_engine = docker_engine
    
    def submit_task(self, task: TaskDefinition) -> str:
        """Добавление задачи в очередь"""
        # Проверяем зависимости
        self._register_dependencies(task)
        
        # Добавляем в очередь с приоритетом
        priority_value = -task.priority.value  # Negative for max-heap behavior
        timestamp = time.time()
        
        with self.queue_lock:
            heapq.heappush(self.task_queue, (priority_value, timestamp, task))
        
        return task.id
    
    def _register_dependencies(self, task: TaskDefinition):
        """Регистрация зависимостей задачи"""
        task_id = task.id
        dependencies = set(task.dependencies)
        
        self.dependency_graph[task_id] = dependencies
        
        # Обновляем обратные зависимости
        for dep_id in dependencies:
            if dep_id not in self.dependent_tasks:
                self.dependent_tasks[dep_id] = set()
            self.dependent_tasks[dep_id].add(task_id)
    
    def _are_dependencies_satisfied(self, task: TaskDefinition) -> bool:
        """Проверка выполнения всех зависимостей"""
        dependencies = self.dependency_graph.get(task.id, set())
        
        for dep_id in dependencies:
            if dep_id not in self.completed_tasks:
                return False
        
        return True
    
    async def start_scheduler(self):
        """Запуск планировщика"""
        self.running = True
        self.scheduler_task = asyncio.create_task(self._scheduler_loop())
    
    async def stop_scheduler(self):
        """Остановка планировщика"""
        self.running = False
        
        if self.scheduler_task:
            self.scheduler_task.cancel()
            try:
                await self.scheduler_task
            except asyncio.CancelledError:
                pass
        
        # Ждем завершения активных задач
        while self.active_tasks:
            await asyncio.sleep(0.1)
    
    async def _scheduler_loop(self):
        """Основной цикл планировщика"""
        while self.running:
            try:
                # Проверяем, есть ли место для новых задач
                if len(self.active_tasks) >= self.max_concurrent_tasks:
                    await asyncio.sleep(0.1)
                    continue
                
                # Ищем готовую к выполнению задачу
                ready_task = self._get_ready_task()
                
                if ready_task:
                    # Запускаем задачу
                    await self._execute_task(ready_task)
                else:
                    # Нет готовых задач, ждем
                    await asyncio.sleep(0.1)
                    
            except Exception as e:
                print(f"Scheduler error: {e}")
                await asyncio.sleep(1)
    
    def _get_ready_task(self) -> Optional[TaskDefinition]:
        """Получение следующей готовой к выполнению задачи"""
        with self.queue_lock:
            ready_tasks = []
            
            # Временно сохраняем задачи, которые не готовы
            temp_queue = []
            
            while self.task_queue:
                priority, timestamp, task = heapq.heappop(self.task_queue)
                
                if self._are_dependencies_satisfied(task):
                    ready_tasks.append((priority, timestamp, task))
                else:
                    temp_queue.append((priority, timestamp, task))
            
            # Возвращаем неготовые задачи в очередь
            for item in temp_queue:
                heapq.heappush(self.task_queue, item)
            
            # Возвращаем задачу с наивысшим приоритетом
            if ready_tasks:
                ready_tasks.sort()  # Сортируем по приоритету
                _, _, task = ready_tasks[0]
                return task
        
        return None
    
    async def _execute_task(self, task: TaskDefinition):
        """Выполнение задачи"""
        task_id = task.id
        task.started_at = datetime.now()
        self.active_tasks[task_id] = task
        
        # Вызываем callbacks
        await self._call_callbacks('on_task_start', task)
        
        try:
            # Выбираем execution engine
            if task.execution_mode == ExecutionMode.ASYNC:
                engine = self.async_engine
            elif task.execution_mode == ExecutionMode.PROCESS:
                engine = self.process_engine
            elif task.execution_mode == ExecutionMode.DOCKER:
                engine = self.docker_engine
            else:
                raise ValueError(f"Unsupported execution mode: {task.execution_mode}")
            
            # Выполняем задачу
            result = await engine.execute_task(task)
            
            # Обработка результата
            await self._handle_task_completion(task, result)
            
        except Exception as e:
            # Обработка ошибки
            await self._handle_task_error(task, e)
    
    async def _handle_task_completion(self, task: TaskDefinition, result: TaskResult):
        """Обработка успешного завершения задачи"""
        task.completed_at = datetime.now()
        
        # Сохраняем результат
        self.completed_tasks[task.id] = result
        
        # Удаляем из активных
        del self.active_tasks[task.id]
        
        # Вызываем callbacks
        await self._call_callbacks('on_task_complete', task, result)
        
        # Проверяем зависимые задачи
        await self._check_dependent_tasks(task.id)
    
    async def _handle_task_error(self, task: TaskDefinition, error: Exception):
        """Обработка ошибки выполнения задачи"""
        task.retry_count += 1
        
        if task.retry_count <= task.max_retries:
            # Повторная попытка
            await self._call_callbacks('on_task_retry', task, error)
            
            # Возвращаем в очередь
            priority_value = -task.priority.value
            timestamp = time.time()
            
            with self.queue_lock:
                heapq.heappush(self.task_queue, (priority_value, timestamp, task))
        
        else:
            # Максимальное количество попыток исчерпано
            task.completed_at = datetime.now()
            
            result = TaskResult(
                task_id=task.id,
                success=False,
                error_message=str(error),
                execution_time=0.0
            )
            
            self.failed_tasks[task.id] = result
            
            # Удаляем из активных
            del self.active_tasks[task.id]
            
            # Вызываем callbacks
            await self._call_callbacks('on_task_error', task, error)
    
    async def _check_dependent_tasks(self, completed_task_id: str):
        """Проверка задач, зависящих от завершенной"""
        dependent_ids = self.dependent_tasks.get(completed_task_id, set())
        
        for dep_id in dependent_ids:
            # Проверяем, готова ли зависимая задача к выполнению
            # (это будет проверено в следующем цикле планировщика)
            pass
    
    async def _call_callbacks(self, event: str, *args):
        """Вызов callbacks для события"""
        callbacks = self.task_callbacks.get(event, [])
        
        for callback in callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(*args)
                else:
                    callback(*args)
            except Exception as e:
                print(f"Callback error for {event}: {e}")
    
    def add_callback(self, event: str, callback: Callable):
        """Добавление callback для события"""
        if event in self.task_callbacks:
            self.task_callbacks[event].append(callback)
    
    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Получение статуса задачи"""
        if task_id in self.active_tasks:
            return {'status': 'running', 'task': self.active_tasks[task_id]}
        elif task_id in self.completed_tasks:
            return {'status': 'completed', 'result': self.completed_tasks[task_id]}
        elif task_id in self.failed_tasks:
            return {'status': 'failed', 'result': self.failed_tasks[task_id]}
        else:
            return {'status': 'not_found'}
    
    def get_queue_status(self) -> Dict[str, Any]:
        """Получение статуса очереди"""
        with self.queue_lock:
            queue_size = len(self.task_queue)
        
        return {
            'queue_size': queue_size,
            'active_tasks': len(self.active_tasks),
            'completed_tasks': len(self.completed_tasks),
            'failed_tasks': len(self.failed_tasks)
        }
```

## Execution Engines

### 1. AsyncIO Execution Engine

```python
class AsyncExecutionEngine:
    """Асинхронный движок выполнения для I/O операций"""
    
    def __init__(self, max_concurrent: int = 50):
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.plugin_manager: Optional['PluginLifecycleManager'] = None
        self.data_manager: Optional['ZeroCopyDataManager'] = None
    
    def set_dependencies(self, plugin_manager, data_manager):
        self.plugin_manager = plugin_manager
        self.data_manager = data_manager
    
    async def execute_task(self, task: TaskDefinition) -> TaskResult:
        """Асинхронное выполнение задачи"""
        async with self.semaphore:
            start_time = time.time()
            
            try:
                # Получаем plugin
                plugin_instance = await self.plugin_manager.initialize_plugin(
                    task.plugin_name, task.plugin_config
                )
                
                # Подготавливаем входные данные
                input_data = await self._prepare_input_data(task)
                
                # Выполняем метод plugin
                method = getattr(plugin_instance.instance, task.plugin_method)
                
                if asyncio.iscoroutinefunction(method):
                    result = await method(*input_data)
                else:
                    # Выполняем в thread pool для синхронных методов
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(None, method, *input_data)
                
                # Сохраняем результат
                output_data_id = await self._save_result_data(task, result)
                
                execution_time = time.time() - start_time
                
                return TaskResult(
                    task_id=task.id,
                    success=True,
                    result_data=result,
                    execution_time=execution_time,
                    output_data_id=output_data_id
                )
                
            except Exception as e:
                execution_time = time.time() - start_time
                return TaskResult(
                    task_id=task.id,
                    success=False,
                    error_message=str(e),
                    execution_time=execution_time
                )
    
    async def _prepare_input_data(self, task: TaskDefinition) -> List[Any]:
        """Подготовка входных данных"""
        input_data = []
        
        for data_id in task.input_data_ids:
            if task.shared_memory_config.get('use_shared_memory', False):
                # Используем shared memory handle
                descriptor = self.data_manager.data_registry[data_id]
                from zero_copy_data_management import SharedMemoryHandle
                handle = SharedMemoryHandle(
                    name=descriptor.shared_memory_name,
                    size=descriptor.size_bytes,
                    dtype="arrow_ipc",
                    shape=descriptor.shape
                )
                input_data.append(handle)
            else:
                # Обычная загрузка данных
                data = self.data_manager.get_data(data_id, "polars")
                input_data.append(data)
        
        return input_data
    
    async def _save_result_data(self, task: TaskDefinition, result: Any) -> Optional[str]:
        """Сохранение результата"""
        if task.output_data_id and hasattr(result, 'to_pandas'):
            # Результат - DataFrame, сохраняем
            return self.data_manager.register_data(
                result, task.output_data_id, "auto"
            )
        return None
```

### 2. Process Execution Engine

```python
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

class ProcessExecutionEngine:
    """Движок выполнения в отдельных процессах"""
    
    def __init__(self, max_workers: int = None):
        self.max_workers = max_workers or mp.cpu_count()
        self.executor = ProcessPoolExecutor(max_workers=self.max_workers)
        self.plugin_manager = None
        self.data_manager = None
        
        # Shared memory configuration
        self.shared_memory_enabled = True
    
    def set_dependencies(self, plugin_manager, data_manager):
        self.plugin_manager = plugin_manager
        self.data_manager = data_manager
    
    async def execute_task(self, task: TaskDefinition) -> TaskResult:
        """Выполнение задачи в отдельном процессе"""
        start_time = time.time()
        
        try:
            # Подготавливаем данные для передачи в процесс
            process_config = await self._prepare_process_config(task)
            
            # Выполняем в процессе
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor,
                _execute_task_in_process,
                process_config
            )
            
            execution_time = time.time() - start_time
            
            # Обрабатываем результат
            if result['success']:
                return TaskResult(
                    task_id=task.id,
                    success=True,
                    result_data=result['data'],
                    execution_time=execution_time,
                    memory_used=result.get('memory_used', 0),
                    cpu_time=result.get('cpu_time', 0.0),
                    output_data_id=result.get('output_data_id')
                )
            else:
                return TaskResult(
                    task_id=task.id,
                    success=False,
                    error_message=result['error'],
                    execution_time=execution_time
                )
                
        except Exception as e:
            execution_time = time.time() - start_time
            return TaskResult(
                task_id=task.id,
                success=False,
                error_message=str(e),
                execution_time=execution_time
            )
    
    async def _prepare_process_config(self, task: TaskDefinition) -> Dict[str, Any]:
        """Подготовка конфигурации для процесса"""
        config = {
            'task_id': task.id,
            'plugin_name': task.plugin_name,
            'plugin_method': task.plugin_method,
            'plugin_config': task.plugin_config,
            'shared_memory_handles': [],
            'data_configs': []
        }
        
        # Подготавливаем shared memory handles для input данных
        for data_id in task.input_data_ids:
            if self.shared_memory_enabled:
                # Создаем shared memory view
                handle = self.data_manager.create_shared_memory_view(
                    self.data_manager.get_data(data_id, "polars").collect(),
                    f"{data_id}_process_{task.id}"
                )
                config['shared_memory_handles'].append(handle.to_dict())
            else:
                # Обычная сериализация (медленнее)
                data_config = {
                    'data_id': data_id,
                    'format': 'pickle',
                    'data': self.data_manager.get_data(data_id, "polars").collect()
                }
                config['data_configs'].append(data_config)
        
        return config

def _execute_task_in_process(config: Dict[str, Any]) -> Dict[str, Any]:
    """Функция выполнения в отдельном процессе"""
    import time
    import psutil
    import os
    
    start_time = time.time()
    process = psutil.Process(os.getpid())
    start_memory = process.memory_info().rss
    
    try:
        # Инициализируем plugin manager в процессе
        from plugin_system import PluginRegistry, PluginLifecycleManager
        from zero_copy_data_management import ZeroCopyDataManager, SharedMemoryHandle
        
        registry = PluginRegistry()
        data_manager = ZeroCopyDataManager()
        lifecycle_manager = PluginLifecycleManager(registry, data_manager)
        
        # Обнаруживаем plugins
        registry.discover_entry_points()
        
        # Инициализируем plugin
        plugin_instance = asyncio.run(lifecycle_manager.initialize_plugin(
            config['plugin_name'], config['plugin_config']
        ))
        
        # Загружаем входные данные
        input_data = []
        
        # Из shared memory
        for handle_dict in config['shared_memory_handles']:
            handle = SharedMemoryHandle.from_dict(handle_dict)
            data = data_manager.load_from_shared_memory(handle)
            input_data.append(data.collect())
        
        # Из обычных data configs
        for data_config in config['data_configs']:
            input_data.append(data_config['data'])
        
        # Выполняем метод
        method = getattr(plugin_instance.instance, config['plugin_method'])
        result = method(*input_data)
        
        # Сохраняем результат если нужно
        output_data_id = None
        if hasattr(result, 'to_pandas'):
            output_data_id = data_manager.register_data(result, strategy="shared_memory")
        
        # Статистика использования ресурсов
        end_memory = process.memory_info().rss
        memory_used = end_memory - start_memory
        cpu_time = time.time() - start_time
        
        return {
            'success': True,
            'data': result,
            'output_data_id': output_data_id,
            'memory_used': memory_used,
            'cpu_time': cpu_time
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }
```

### 3. Docker Execution Engine

```python
import docker
import json
import tempfile
import os
from pathlib import Path

class DockerExecutionEngine:
    """Движок выполнения в Docker контейнерах"""
    
    def __init__(self, docker_client: docker.DockerClient = None):
        self.docker_client = docker_client or docker.from_env()
        self.data_manager = None
        self.plugin_registry = None
        
        # Docker configuration
        self.default_volumes = {}
        self.default_environment = {
            'PYTHONPATH': '/app',
            'NORMAML_MODE': 'docker'
        }
        
        # Shared memory configuration
        self.shared_memory_size = '4g'
        self.shared_data_path = '/shared_data'
    
    def set_dependencies(self, plugin_registry, data_manager):
        self.plugin_registry = plugin_registry
        self.data_manager = data_manager
    
    async def execute_task(self, task: TaskDefinition) -> TaskResult:
        """Выполнение задачи в Docker контейнере"""
        start_time = time.time()
        container = None
        
        try:
            # Подготавливаем данные для контейнера
            container_config = await self._prepare_container_config(task)
            
            # Создаем и запускаем контейнер
            container = await self._create_and_run_container(container_config)
            
            # Ждем завершения
            result = container.wait()
            exit_code = result['StatusCode']
            
            # Получаем логи
            logs = container.logs().decode('utf-8')
            
            execution_time = time.time() - start_time
            
            if exit_code == 0:
                # Извлекаем результат
                result_data = await self._extract_result(container, task)
                
                return TaskResult(
                    task_id=task.id,
                    success=True,
                    result_data=result_data,
                    execution_time=execution_time,
                    metadata={'logs': logs, 'exit_code': exit_code}
                )
            else:
                return TaskResult(
                    task_id=task.id,
                    success=False,
                    error_message=f"Container exited with code {exit_code}. Logs: {logs}",
                    execution_time=execution_time
                )
                
        except Exception as e:
            execution_time = time.time() - start_time
            return TaskResult(
                task_id=task.id,
                success=False,
                error_message=str(e),
                execution_time=execution_time
            )
        finally:
            # Очистка контейнера
            if container:
                try:
                    container.remove(force=True)
                except:
                    pass
    
    async def _prepare_container_config(self, task: TaskDefinition) -> Dict[str, Any]:
        """Подготовка конфигурации контейнера"""
        # Базовая конфигурация
        config = {
            'image': task.docker_image,
            'environment': {**self.default_environment, **task.docker_environment},
            'volumes': {**self.default_volumes, **task.docker_volumes},
            'detach': True,
            'remove': False,  # Удалим вручную после извлечения результата
            'shm_size': self.shared_memory_size,
            'mem_limit': f"{task.resources.memory_mb}m" if task.resources.memory_mb else None,
            'cpu_count': task.resources.cpu_cores,
        }
        
        # Настройка shared memory для данных
        await self._setup_shared_data_volumes(task, config)
        
        # Создание task config файла
        task_config_path = await self._create_task_config_file(task)
        config['volumes'][task_config_path] = {
            'bind': '/app/task_config.json',
            'mode': 'ro'
        }
        
        # Команда выполнения
        config['command'] = [
            'python', '-c',
            '''
import json
import sys
sys.path.append("/app")

from normaml_docker_runner import run_task

with open("/app/task_config.json", "r") as f:
    task_config = json.load(f)

result = run_task(task_config)
print(json.dumps(result))
            '''
        ]
        
        return config
    
    async def _setup_shared_data_volumes(self, task: TaskDefinition, config: Dict[str, Any]):
        """Настройка shared volumes для данных"""
        # Создаем временную директорию для shared data
        temp_dir = tempfile.mkdtemp(prefix="normaml_docker_")
        
        # Подготавливаем данные для каждого input
        for i, data_id in enumerate(task.input_data_ids):
            # Экспортируем данные в shared memory или memory-mapped file
            shared_info = self.data_manager.share_across_containers(
                data_id, [f"container_{task.id}"]
            )
            
            if shared_info['type'] == 'memory_mapped_file':
                # Монтируем memory-mapped file в контейнер
                host_path = shared_info['path']
                container_path = f"{self.shared_data_path}/input_{i}.mmap"
                
                config['volumes'][host_path] = {
                    'bind': container_path,
                    'mode': 'ro'
                }
                
                # Добавляем информацию в environment
                config['environment'][f'INPUT_DATA_{i}_PATH'] = container_path
                config['environment'][f'INPUT_DATA_{i}_HANDLE'] = json.dumps(shared_info['handle'])
        
        # Монтируем /dev/shm для shared memory
        config['volumes']['/dev/shm'] = {
            'bind': '/dev/shm',
            'mode': 'rw'
        }
        
        # Директория для output данных
        output_dir = os.path.join(temp_dir, 'output')
        os.makedirs(output_dir, exist_ok=True)
        
        config['volumes'][output_dir] = {
            'bind': f"{self.shared_data_path}/output",
            'mode': 'rw'
        }
        
        config['environment']['OUTPUT_DATA_PATH'] = f"{self.shared_data_path}/output"
        config['environment']['SHARED_DATA_PATH'] = self.shared_data_path
        
        # Сохраняем temp_dir для последующей очистки
        config['_temp_dir'] = temp_dir
    
    async def _create_task_config_file(self, task: TaskDefinition) -> str:
        """Создание файла конфигурации задачи"""
        task_config = task.to_dict()
        
        # Создаем временный файл
        fd, temp_path = tempfile.mkstemp(suffix='.json', prefix='normaml_task_')
        
        try:
            with os.fdopen(fd, 'w') as f:
                json.dump(task_config, f, indent=2, default=str)
        except:
            os.close(fd)
            raise
        
        return temp_path
    
    async def _create_and_run_container(self, config: Dict[str, Any]) -> docker.models.containers.Container:
        """Создание и запуск контейнера"""
        # Убираем внутренние поля
        temp_dir = config.pop('_temp_dir', None)
        
        # Создаем контейнер
        container = self.docker_client.containers.create(**config)
        
        # Запускаем
        container.start()
        
        return container
    
    async def _extract_result(self, container: docker.models.containers.Container, 
                            task: TaskDefinition) -> Any:
        """Извлечение результата из контейнера"""
        try:
            # Получаем результат из logs
            logs = container.logs().decode('utf-8')
            
            # Ищем JSON результат в логах
            lines = logs.split('\n')
            for line in reversed(lines):
                if line.strip().startswith('{'):
                    try:
                        result = json.loads(line.strip())
                        if 'output_data_path' in result:
                            # Загружаем данные из shared output
                            return await self._load_output_data(result['output_data_path'])
                        else:
                            return result
                    except json.JSONDecodeError:
                        continue
            
            return {'raw_logs': logs}
            
        except Exception as e:
            return {'error': f"Failed to extract result: {e}"}
    
    async def _load_output_data(self, output_path: str) -> Any:
        """Загрузка выходных данных"""
        # Здесь можно реализовать загрузку данных из output директории
        # В зависимости от формата (parquet, arrow, etc.)
        if os.path.exists(output_path):
            # Пример для parquet
            if output_path.endswith('.parquet'):
                import polars as pl
                return pl.read_parquet(output_path)
        
        return None
```

## Resource Management

```python
import psutil
from typing import Dict, Any, List
import asyncio

class ResourceManager:
    """Управление ресурсами системы"""
    
    def __init__(self):
        self.cpu_cores = psutil.cpu_count()
        self.total_memory = psutil.virtual_memory().total
        self.available_memory = psutil.virtual_memory().available
        
        # Tracking активных ресурсов
        self.allocated_cpu = 0
        self.allocated_memory = 0
        self.active_allocations: Dict[str, ResourceRequirements] = {}
        
        # Пределы использования
        self.max_cpu_utilization = 0.8  # 80%
        self.max_memory_utilization = 0.8  # 80%
    
    def can_allocate_resources(self, requirements: ResourceRequirements) -> bool:
        """Проверка возможности выделения ресурсов"""
        required_cpu = requirements.cpu_cores or 1
        required_memory = requirements.memory_mb or 0
        
        # Проверяем CPU
        if self.allocated_cpu + required_cpu > self.cpu_cores * self.max_cpu_utilization:
            return False
        
        # Проверяем память
        required_memory_bytes = required_memory * 1024 * 1024
        if self.allocated_memory + required_memory_bytes > self.total_memory * self.max_memory_utilization:
            return False
        
        return True
    
    def allocate_resources(self, task_id: str, requirements: ResourceRequirements) -> bool:
        """Выделение ресурсов для задачи"""
        if not self.can_allocate_resources(requirements):
            return False
        
        required_cpu = requirements.cpu_cores or 1
        required_memory = requirements.memory_mb or 0
        
        self.allocated_cpu += required_cpu
        self.allocated_memory += required_memory * 1024 * 1024
        self.active_allocations[task_id] = requirements
        
        return True
    
    def release_resources(self, task_id: str):
        """Освобождение ресурсов задачи"""
        if task_id in self.active_allocations:
            requirements = self.active_allocations[task_id]
            
            self.allocated_cpu -= requirements.cpu_cores or 1
            self.allocated_memory -= (requirements.memory_mb or 0) * 1024 * 1024
            
            del self.active_allocations[task_id]
    
    def get_resource_stats(self) -> Dict[str, Any]:
        """Получение статистики ресурсов"""
        current_memory = psutil.virtual_memory()
        current_cpu = psutil.cpu_percent(interval=1)
        
        return {
            'cpu': {
                'total_cores': self.cpu_cores,
                'allocated_cores': self.allocated_cpu,
                'current_utilization': current_cpu,
                'available_cores': self.cpu_cores - self.allocated_cpu
            },
            'memory': {
                'total_bytes': self.total_memory,
                'allocated_bytes': self.allocated_memory,
                'current_usage_bytes': current_memory.used,
                'available_bytes': current_memory.available,
                'utilization_percent': current_memory.percent
            },
            'active_tasks': len(self.active_allocations)
        }

class LoadBalancer:
    """Балансировщик нагрузки между execution engines"""
    
    def __init__(self, resource_manager: ResourceManager):
        self.resource_manager = resource_manager
        self.engine_stats: Dict[str, Dict[str, Any]] = {
            'async': {'active_tasks': 0, 'avg_execution_time': 0.0},
            'process': {'active_tasks': 0, 'avg_execution_time': 0.0},
            'docker': {'active_tasks': 0, 'avg_execution_time': 0.0}
        }
    
    def choose_execution_mode(self, task: TaskDefinition) -> ExecutionMode:
        """Выбор оптимального режима выполнения"""
        # Если режим явно указан, используем его
        if task.execution_mode != ExecutionMode.PROCESS:  # default
            return task.execution_mode
        
        # Автоматический выбор на основе типа задачи и ресурсов
        if task.task_type == TaskType.DATA_LOADING:
            return ExecutionMode.ASYNC  # I/O bound
        
        elif task.task_type == TaskType.MODEL_TRAINING:
            if task.resources.cpu_cores and task.resources.cpu_cores > 4:
                return ExecutionMode.DOCKER  # Изоляция для тяжелых задач
            else:
                return ExecutionMode.PROCESS  # CPU bound
        
        elif task.task_type == TaskType.HYPERPARAMETER_TUNING:
            return ExecutionMode.DOCKER  # Изоляция и reproducibility
        
        else:
            # Выбор на основе загрузки engines
            min_load_engine = min(
                self.engine_stats.items(),
                key=lambda x: x[1]['active_tasks']
            )
            
            return ExecutionMode(min_load_engine[0])
    
    def update_engine_stats(self, engine_name: str, execution_time: float):
        """Обновление статистики engine"""
        stats = self.engine_stats[engine_name]
        stats['active_tasks'] = max(0, stats['active_tasks'] - 1)
        
        # Обновляем среднее время выполнения
        current_avg = stats['avg_execution_time']
        stats['avg_execution_time'] = (current_avg + execution_time) / 2
    
    def notify_task_start(self, engine_name: str):
        """Уведомление о начале задачи"""
        self.engine_stats[engine_name]['active_tasks'] += 1
```

## Integration Example

```python
# Пример интеграции всех компонентов
async def setup_parallel_execution_system():
    """Настройка системы параллельного выполнения"""
    
    # Инициализация компонентов
    data_manager = ZeroCopyDataManager()
    plugin_registry = PluginRegistry()
    plugin_lifecycle = PluginLifecycleManager(plugin_registry, data_manager)
    resource_manager = ResourceManager()
    load_balancer = LoadBalancer(resource_manager)
    
    # Execution engines
    async_engine = AsyncExecutionEngine()
    process_engine = ProcessExecutionEngine()
    docker_engine = DockerExecutionEngine()
    
    # Настройка зависимостей
    async_engine.set_dependencies(plugin_lifecycle, data_manager)
    process_engine.set_dependencies(plugin_lifecycle, data_manager)
    docker_engine.set_dependencies(plugin_registry, data_manager)
    
    # Task scheduler
    scheduler = TaskScheduler(max_concurrent_tasks=10)
    scheduler.register_execution_engines(async_engine, process_engine, docker_engine)
    
    # Callbacks для resource management
    def on_task_start(task):
        resource_manager.allocate_resources(task.id, task.resources)
        load_balancer.notify_task_start(task.execution_mode.value)
    
    def on_task_complete(task, result):
        resource_manager.release_resources(task.id)
        load_balancer.update_engine_stats(task.execution_mode.value, result.execution_time)
    
    scheduler.add_callback('on_task_start', on_task_start)
    scheduler.add_callback('on_task_complete', on_task_complete)
    
    # Запуск планировщика
    await scheduler.start_scheduler()
    
    return {
        'scheduler': scheduler,
        'data_manager': data_manager,
        'resource_manager': resource_manager,
        'load_balancer': load_balancer
    }

# Пример использования
async def run_ml_pipeline():
    system = await setup_parallel_execution_system()
    scheduler = system['scheduler']
    
    # Создание задач pipeline
    tasks = [
        TaskDefinition(
            task_type=TaskType.DATA_LOADING,
            execution_mode=ExecutionMode.ASYNC,
            plugin_name="csv_loader",
            plugin_method="load",
            plugin_config={"source": "data.csv"},
            resources=ResourceRequirements(memory_mb=512)
        ),
        TaskDefinition(
            task_type=TaskType.FEATURE_ENGINEERING,
            execution_mode=ExecutionMode.PROCESS,
            plugin_name="sklearn_feature_selector",
            plugin_method="select_features",
            dependencies=["data_loading_task"],
            resources=ResourceRequirements(cpu_cores=2, memory_mb=1024)
        ),
        TaskDefinition(
            task_type=TaskType.MODEL_TRAINING,
            execution_mode=ExecutionMode.DOCKER,
            plugin_name="xgboost_trainer",
            plugin_method="train",
            docker_image="normaml/xgboost:latest",
            dependencies=["feature_engineering_task"],
            resources=ResourceRequirements(cpu_cores=4, memory_mb=4096)
        )
    ]
    
    # Отправка задач
    for task in tasks:
        task_id = scheduler.submit_task(task)
        print(f"Submitted task {task_id}")
    
    # Мониторинг выполнения
    while True:
        status = scheduler.get_queue_status()
        print(f"Queue status: {status}")
        
        if status['active_tasks'] == 0 and status['queue_size'] == 0:
            break
        
        await asyncio.sleep(1)
    
    await scheduler.stop_scheduler()
```

Эта архитектура обеспечивает:

1. **Эффективное планирование** - приоритетная очередь с dependency resolution
2. **Multiple execution modes** - async, process, и docker для разных типов задач
3. **Resource management** - контроль использования CPU и памяти
4. **Zero-copy data sharing** - через shared memory и memory-mapped files
5. **Fault tolerance** - retry mechanisms и error handling
6. **Monitoring** - статистика выполнения и использования ресурсов
7. **Load balancing** - автоматический выбор optimal execution engine
8. **Docker isolation** - безопасное выполнение с shared memory support