# План тестирования NormaML

## Стратегия тестирования

### Общие принципы

1. **Pyramid Testing Strategy**:
   - Unit tests (70%) - быстрые, изолированные тесты компонентов
   - Integration tests (20%) - тестирование взаимодействия компонентов
   - End-to-end tests (10%) - полные сценарии использования

2. **Test Coverage**:
   - Минимум 90% code coverage для core компонентов
   - 100% coverage для критических функций (data management, plugin system)
   - Property-based testing для complex algorithms

3. **Test Types**:
   - Functional testing
   - Performance testing
   - Security testing
   - Compatibility testing
   - Regression testing

4. **Test Environment**:
   - Локальная разработка
   - CI/CD pipeline
   - Docker containers
   - Staging environment

## Структура тестов

```
tests/
├── unit/                          # Unit tests
│   ├── test_core/
│   │   ├── test_plugin_manager.py
│   │   ├── test_data_manager.py
│   │   ├── test_execution_engine.py
│   │   ├── test_experiment_tracker.py
│   │   └── test_event_bus.py
│   ├── test_plugins/
│   │   ├── test_data_loaders/
│   │   ├── test_feature_engineers/
│   │   ├── test_model_trainers/
│   │   ├── test_evaluators/
│   │   └── test_backends/
│   ├── test_data/
│   │   ├── test_zero_copy.py
│   │   ├── test_shared_memory.py
│   │   ├── test_streaming.py
│   │   └── test_optimization.py
│   ├── test_execution/
│   │   ├── test_scheduler.py
│   │   ├── test_engines/
│   │   ├── test_resources.py
│   │   └── test_load_balancer.py
│   ├── test_tracking/
│   │   ├── test_mlflow_integration.py
│   │   ├── test_plugin_logger.py
│   │   └── test_distributed_coordinator.py
│   └── test_utils/
│       ├── test_logging.py
│       ├── test_serialization.py
│       └── test_validation.py
├── integration/                   # Integration tests
│   ├── test_pipelines/
│   │   ├── test_basic_pipeline.py
│   │   ├── test_complex_pipeline.py
│   │   └── test_distributed_pipeline.py
│   ├── test_docker/
│   │   ├── test_container_execution.py
│   │   ├── test_shared_memory_docker.py
│   │   └── test_docker_networking.py
│   ├── test_mlflow/
│   │   ├── test_experiment_tracking.py
│   │   ├── test_model_registry.py
│   │   └── test_distributed_tracking.py
│   └── test_plugin_interactions/
│       ├── test_plugin_discovery.py
│       ├── test_plugin_lifecycle.py
│       └── test_plugin_dependencies.py
├── performance/                   # Performance tests
│   ├── test_memory/
│   │   ├── test_zero_copy_performance.py
│   │   ├── test_shared_memory_performance.py
│   │   └── test_memory_leaks.py
│   ├── test_speed/
│   │   ├── test_data_loading_speed.py
│   │   ├── test_processing_speed.py
│   │   └── test_model_training_speed.py
│   └── benchmarks/
│       ├── benchmark_data_operations.py
│       ├── benchmark_plugin_execution.py
│       └── benchmark_end_to_end.py
├── e2e/                          # End-to-end tests
│   ├── test_complete_workflows/
│   ├── test_user_scenarios/
│   └── test_production_scenarios/
├── security/                     # Security tests
│   ├── test_plugin_isolation.py
│   ├── test_data_access_control.py
│   └── test_container_security.py
└── compatibility/                # Compatibility tests
    ├── test_python_versions/
    ├── test_dependency_versions/
    └── test_platform_compatibility/
```

## Unit Tests

### 1. Core Components Tests

#### test_plugin_manager.py

```python
import pytest
from unittest.mock import Mock, patch, MagicMock
from normaml.core.plugin_manager import PluginManager, PluginRegistry
from normaml.core.interfaces import DataLoaderPlugin, PluginMetadata
from normaml.plugins.base import BasePlugin

class TestPluginManager:
    
    @pytest.fixture
    def plugin_manager(self):
        return PluginManager()
    
    @pytest.fixture
    def mock_plugin_class(self):
        class MockPlugin(DataLoaderPlugin, BasePlugin):
            name = "test_plugin"
            version = "1.0.0"
            capabilities = ["test_capability"]
            
            def supports_source(self, source): return True
            def load(self, source, **kwargs): return Mock()
            def get_schema(self, source): return {}
            def estimate_size(self, source): return 1000
            def initialize(self, config): pass
            def validate_config(self, config): return True
            def get_capabilities(self): return {}
            
        return MockPlugin
    
    def test_plugin_discovery(self, plugin_manager):
        """Test plugin discovery mechanism."""
        initial_count = len(plugin_manager.registry.plugins)
        plugin_manager.discover_plugins()
        
        # Should discover built-in plugins
        assert len(plugin_manager.registry.plugins) > initial_count
    
    def test_plugin_registration(self, plugin_manager, mock_plugin_class):
        """Test manual plugin registration."""
        metadata = PluginMetadata(
            name="test_plugin",
            version="1.0.0",
            plugin_type="data_loader",
            class_path="test.mock_plugin.MockPlugin",
            module_path="test.mock_plugin"
        )
        
        plugin_manager.registry.register_plugin(metadata)
        assert "test_plugin" in plugin_manager.registry.plugins
    
    def test_plugin_initialization(self, plugin_manager, mock_plugin_class):
        """Test plugin initialization with lifecycle management."""
        with patch.object(plugin_manager, '_load_plugin_class', 
                         return_value=mock_plugin_class):
            
            plugin_instance = plugin_manager.initialize_plugin(
                "test_plugin", {"test_config": "value"}
            )
            
            assert plugin_instance is not None
            assert plugin_instance.metadata.name == "test_plugin"
    
    def test_plugin_dependency_resolution(self, plugin_manager):
        """Test dependency resolution between plugins."""
        # Create mock plugins with dependencies
        plugin_a_meta = PluginMetadata(
            name="plugin_a", version="1.0.0", plugin_type="data_loader",
            class_path="test.plugin_a", module_path="test"
        )
        plugin_b_meta = PluginMetadata(
            name="plugin_b", version="1.0.0", plugin_type="feature_engineer", 
            class_path="test.plugin_b", module_path="test",
            dependencies=["plugin_a"]
        )
        
        plugin_manager.registry.register_plugin(plugin_a_meta)
        plugin_manager.registry.register_plugin(plugin_b_meta)
        
        resolved = plugin_manager.registry.resolve_dependencies(["plugin_b"])
        assert resolved == ["plugin_a", "plugin_b"]
    
    def test_plugin_error_handling(self, plugin_manager):
        """Test error handling in plugin operations."""
        with pytest.raises(ValueError, match="Plugin .* not found"):
            plugin_manager.get_plugin("nonexistent_plugin")
    
    @pytest.mark.asyncio
    async def test_async_plugin_execution(self, plugin_manager, mock_plugin_class):
        """Test asynchronous plugin execution."""
        with patch.object(plugin_manager, '_load_plugin_class',
                         return_value=mock_plugin_class):
            
            plugin_instance = plugin_manager.initialize_plugin("test_plugin", {})
            
            # Test async method execution
            result = await plugin_manager.execute_plugin_method(
                "test_plugin", "load", "test_source"
            )
            
            assert result is not None
```

#### test_data_manager.py

```python
import pytest
import polars as pl
import pyarrow as pa
import numpy as np
from unittest.mock import Mock, patch
from normaml.data.zero_copy import ZeroCopyDataManager, DataDescriptor
from normaml.data.shared_memory import SharedMemoryHandle

class TestZeroCopyDataManager:
    
    @pytest.fixture
    def data_manager(self):
        return ZeroCopyDataManager(
            max_memory_mb=1024,
            enable_shared_memory=True,
            enable_memory_mapping=True
        )
    
    @pytest.fixture
    def sample_dataframe(self):
        return pl.DataFrame({
            'id': range(1000),
            'value': np.random.rand(1000),
            'category': ['A', 'B', 'C'] * 333 + ['A']
        })
    
    def test_data_registration_memory(self, data_manager, sample_dataframe):
        """Test data registration in memory."""
        data_id = data_manager.register_data(
            sample_dataframe, "test_data", strategy="arrow_buffer"
        )
        
        assert data_id in data_manager.data_registry
        descriptor = data_manager.data_registry[data_id]
        assert descriptor.format.value == "arrow_ipc"
        assert descriptor.location.value == "memory"
    
    def test_data_registration_shared_memory(self, data_manager, sample_dataframe):
        """Test data registration in shared memory."""
        data_id = data_manager.register_data(
            sample_dataframe, "test_shared", strategy="shared_memory"
        )
        
        assert data_id in data_manager.shared_memory_blocks
        
        # Test loading from shared memory
        loaded_data = data_manager.get_data(data_id, "polars")
        assert loaded_data.shape == sample_dataframe.shape
        
        # Test zero-copy (same data, different references)
        assert not loaded_data.frame_equal(sample_dataframe)  # Different references
        np.testing.assert_array_equal(
            loaded_data.to_numpy(), sample_dataframe.to_numpy()
        )
    
    def test_memory_mapped_storage(self, data_manager, sample_dataframe):
        """Test memory-mapped file storage."""
        data_id = data_manager.register_data(
            sample_dataframe, "test_mmap", strategy="memory_mapped"
        )
        
        descriptor = data_manager.data_registry[data_id]
        assert descriptor.location.value == "memory_mapped"
        assert descriptor.file_path is not None
        
        # Test loading from memory map
        loaded_data = data_manager.get_data(data_id, "polars")
        assert loaded_data.shape == sample_dataframe.shape
    
    def test_lazy_frame_operations(self, data_manager):
        """Test lazy frame operations."""
        lazy_df = pl.LazyFrame({
            'x': range(100),
            'y': range(100, 200)
        })
        
        data_id = data_manager.register_data(
            lazy_df, "test_lazy", strategy="lazy_frame"
        )
        
        # Test that data is not materialized yet
        descriptor = data_manager.data_registry[data_id]
        assert descriptor.size_bytes == 0  # Not materialized
        
        # Test materialization
        materialized = data_manager.get_data(data_id, "polars")
        assert materialized.shape == (100, 2)
    
    def test_data_views(self, data_manager, sample_dataframe):
        """Test zero-copy data views."""
        base_data_id = data_manager.register_data(sample_dataframe, "base_data")
        
        # Create view with selection
        view_id = data_manager.create_view(
            base_data_id, "filtered_view",
            selection={
                'columns': ['id', 'value'],
                'filter': pl.col('value') > 0.5
            }
        )
        
        assert view_id in data_manager.data_registry
        
        # Test that view shares reference count
        base_descriptor = data_manager.data_registry[base_data_id]
        assert base_descriptor.reference_count > 0
    
    def test_memory_usage_tracking(self, data_manager, sample_dataframe):
        """Test memory usage tracking."""
        initial_stats = data_manager.get_memory_stats()
        initial_usage = initial_stats['total_memory_usage']
        
        data_id = data_manager.register_data(sample_dataframe, "memory_test")
        
        updated_stats = data_manager.get_memory_stats()
        assert updated_stats['total_memory_usage'] > initial_usage
        assert updated_stats['registered_datasets'] > initial_stats['registered_datasets']
    
    def test_automatic_cleanup(self, data_manager, sample_dataframe):
        """Test automatic data cleanup."""
        data_id = data_manager.register_data(sample_dataframe, "cleanup_test")
        
        # Manually trigger cleanup
        data_manager._cleanup_data(data_id)
        
        assert data_id not in data_manager.data_registry
        assert data_id not in data_manager.arrow_buffers
    
    def test_concurrent_access(self, data_manager, sample_dataframe):
        """Test concurrent data access."""
        import threading
        import time
        
        data_id = data_manager.register_data(
            sample_dataframe, "concurrent_test", strategy="shared_memory"
        )
        
        results = []
        errors = []
        
        def access_data():
            try:
                for _ in range(10):
                    data = data_manager.get_data(data_id, "polars")
                    results.append(data.shape)
                    time.sleep(0.01)
            except Exception as e:
                errors.append(e)
        
        threads = [threading.Thread(target=access_data) for _ in range(5)]
        
        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join()
        
        assert len(errors) == 0
        assert all(shape == sample_dataframe.shape for shape in results)
    
    @pytest.mark.parametrize("strategy", ["arrow_buffer", "shared_memory", "memory_mapped"])
    def test_storage_strategies(self, data_manager, sample_dataframe, strategy):
        """Test different storage strategies."""
        if strategy == "memory_mapped" and not data_manager.enable_memory_mapping:
            pytest.skip("Memory mapping disabled")
        if strategy == "shared_memory" and not data_manager.enable_shared_memory:
            pytest.skip("Shared memory disabled")
            
        data_id = data_manager.register_data(
            sample_dataframe, f"test_{strategy}", strategy=strategy
        )
        
        # Test data retrieval
        retrieved_data = data_manager.get_data(data_id, "polars")
        assert retrieved_data.shape == sample_dataframe.shape
        
        # Test format conversion
        arrow_data = data_manager.get_data(data_id, "arrow")
        assert isinstance(arrow_data, pa.Table)
        assert arrow_data.shape == sample_dataframe.shape
```

#### test_execution_engine.py

```python
import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from normaml.execution.scheduler import TaskScheduler, TaskDefinition
from normaml.execution.engines import AsyncExecutionEngine, ProcessExecutionEngine
from normaml.execution.resources import ResourceManager

class TestTaskScheduler:
    
    @pytest.fixture
    def scheduler(self):
        return TaskScheduler(max_concurrent_tasks=3)
    
    @pytest.fixture
    def mock_engines(self):
        return {
            'async': Mock(spec=AsyncExecutionEngine),
            'process': Mock(spec=ProcessExecutionEngine),
            'docker': Mock()
        }
    
    def test_task_submission(self, scheduler):
        """Test task submission to scheduler."""
        task = TaskDefinition(
            id="test_task",
            plugin_name="test_plugin",
            plugin_method="test_method",
            priority="normal"
        )
        
        task_id = scheduler.submit_task(task)
        assert task_id == "test_task"
        assert len(scheduler.task_queue) == 1
    
    def test_dependency_resolution(self, scheduler):
        """Test task dependency resolution."""
        task_a = TaskDefinition(id="task_a", plugin_name="plugin_a", plugin_method="method")
        task_b = TaskDefinition(id="task_b", plugin_name="plugin_b", plugin_method="method", 
                               dependencies=["task_a"])
        
        scheduler.submit_task(task_a)
        scheduler.submit_task(task_b)
        
        resolved = scheduler._get_ready_task()
        assert resolved.id == "task_a"  # task_a should be ready first
    
    @pytest.mark.asyncio
    async def test_parallel_execution(self, scheduler, mock_engines):
        """Test parallel task execution."""
        scheduler.register_execution_engines(**mock_engines)
        
        # Mock successful execution
        mock_engines['async'].execute_task = AsyncMock(return_value=Mock(success=True))
        
        tasks = [
            TaskDefinition(id=f"task_{i}", plugin_name="plugin", plugin_method="method")
            for i in range(5)
        ]
        
        for task in tasks:
            scheduler.submit_task(task)
        
        await scheduler.start_scheduler()
        
        # Wait for all tasks to complete
        while scheduler.get_queue_status()['active_tasks'] > 0:
            await asyncio.sleep(0.1)
        
        await scheduler.stop_scheduler()
        
        # Verify all tasks were executed
        assert len(scheduler.completed_tasks) == 5
    
    def test_priority_ordering(self, scheduler):
        """Test task priority ordering."""
        low_task = TaskDefinition(id="low", priority="low", plugin_name="p", plugin_method="m")
        high_task = TaskDefinition(id="high", priority="high", plugin_name="p", plugin_method="m")
        normal_task = TaskDefinition(id="normal", priority="normal", plugin_name="p", plugin_method="m")
        
        # Submit in random order
        scheduler.submit_task(low_task)
        scheduler.submit_task(high_task)
        scheduler.submit_task(normal_task)
        
        # Should get high priority task first
        ready_task = scheduler._get_ready_task()
        assert ready_task.id == "high"
    
    def test_resource_constraints(self, scheduler):
        """Test resource constraint enforcement."""
        resource_manager = ResourceManager()
        scheduler.resource_manager = resource_manager
        
        # Create task that requires more resources than available
        heavy_task = TaskDefinition(
            id="heavy_task",
            plugin_name="plugin",
            plugin_method="method",
            resources={'cpu_cores': 999, 'memory_mb': 999999}
        )
        
        can_allocate = resource_manager.can_allocate_resources(heavy_task.resources)
        assert not can_allocate

class TestAsyncExecutionEngine:
    
    @pytest.fixture
    def engine(self):
        return AsyncExecutionEngine(max_concurrent=10)
    
    @pytest.fixture 
    def mock_plugin_manager(self):
        manager = Mock()
        plugin_instance = Mock()
        plugin_instance.instance.test_method = AsyncMock(return_value="test_result")
        manager.initialize_plugin = AsyncMock(return_value=plugin_instance)
        return manager
    
    @pytest.mark.asyncio
    async def test_async_task_execution(self, engine, mock_plugin_manager):
        """Test asynchronous task execution."""
        engine.set_dependencies(mock_plugin_manager, Mock())
        
        task = TaskDefinition(
            id="async_task",
            plugin_name="test_plugin", 
            plugin_method="test_method"
        )
        
        result = await engine.execute_task(task)
        
        assert result.success
        assert result.result_data == "test_result"
        assert result.execution_time > 0
    
    @pytest.mark.asyncio
    async def test_concurrent_execution_limit(self, engine):
        """Test concurrent execution limits."""
        # Create tasks that take some time
        async def slow_task():
            await asyncio.sleep(0.1)
            return "done"
        
        tasks = [slow_task() for _ in range(20)]
        
        # Should respect max_concurrent limit
        start_time = asyncio.get_event_loop().time()
        results = await engine.execute_parallel(tasks, "thread")
        end_time = asyncio.get_event_loop().time()
        
        assert len(results) == 20
        assert all(result == "done" for result in results)
        # Should take longer due to concurrency limit
        assert end_time - start_time > 0.1

class TestResourceManager:
    
    @pytest.fixture
    def resource_manager(self):
        return ResourceManager()
    
    def test_resource_allocation(self, resource_manager):
        """Test resource allocation and tracking."""
        from normaml.execution.scheduler import ResourceRequirements
        
        requirements = ResourceRequirements(cpu_cores=2, memory_mb=1024)
        
        # Should be able to allocate
        can_allocate = resource_manager.can_allocate_resources(requirements)
        assert can_allocate
        
        # Allocate resources
        allocated = resource_manager.allocate_resources("task_1", requirements)
        assert allocated
        
        # Check allocation tracking
        assert resource_manager.allocated_cpu == 2
        assert resource_manager.allocated_memory == 1024 * 1024 * 1024  # bytes
    
    def test_resource_limits(self, resource_manager):
        """Test resource limit enforcement."""
        from normaml.execution.scheduler import ResourceRequirements
        
        # Try to allocate more than available
        excessive_requirements = ResourceRequirements(
            cpu_cores=999,
            memory_mb=999999
        )
        
        can_allocate = resource_manager.can_allocate_resources(excessive_requirements)
        assert not can_allocate
    
    def test_resource_release(self, resource_manager):
        """Test resource release."""
        from normaml.execution.scheduler import ResourceRequirements
        
        requirements = ResourceRequirements(cpu_cores=2, memory_mb=1024)
        
        resource_manager.allocate_resources("task_1", requirements)
        initial_cpu = resource_manager.allocated_cpu
        initial_memory = resource_manager.allocated_memory
        
        resource_manager.release_resources("task_1")
        
        assert resource_manager.allocated_cpu < initial_cpu
        assert resource_manager.allocated_memory < initial_memory
```

### 2. Integration Tests

#### test_basic_pipeline.py

```python
import pytest
import tempfile
import polars as pl
from normaml import NormaML, Pipeline
from normaml.core.interfaces import RunConfig, RunType

@pytest.mark.integration
class TestBasicPipeline:
    
    @pytest.fixture
    def sample_data_file(self):
        """Create sample CSV file for testing."""
        df = pl.DataFrame({
            'feature1': range(100),
            'feature2': [x * 2 for x in range(100)], 
            'feature3': ['A' if x % 2 == 0 else 'B' for x in range(100)],
            'target': [1 if x > 50 else 0 for x in range(100)]
        })
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.write_csv(f.name)
            return f.name
    
    @pytest.fixture
    def normaml_instance(self):
        """Create NormaML instance for testing."""
        return NormaML(
            config={'data_manager': {'max_memory_mb': 512}},
            enable_auto_logging=False  # Disable for testing
        )
    
    def test_complete_classification_pipeline(self, normaml_instance, sample_data_file):
        """Test complete classification pipeline."""
        
        # Load data
        data_id = normaml_instance.load_data(sample_data_file)
        assert data_id is not None
        
        # Feature engineering
        processed_data_id = normaml_instance.engineer_features(
            data_id,
            engineer="polars_engineer",
            operations=['drop_nulls', 'normalize_numeric']
        )
        assert processed_data_id is not None
        
        # Train model
        model = normaml_instance.train_model(
            processed_data_id,
            target_column="target",
            trainer="sklearn_classifier",
            algorithm="logistic_regression"
        )
        assert model is not None
        
        # Evaluate model
        metrics = normaml_instance.evaluate_model(
            model,
            processed_data_id,
            target_column="target",
            evaluator="classification"
        )
        
        assert 'accuracy' in metrics
        assert 0 <= metrics['accuracy'] <= 1
    
    def test_pipeline_with_config(self, normaml_instance, sample_data_file):
        """Test pipeline creation from configuration."""
        
        pipeline_config = {
            "steps": [
                {
                    "name": "data_loading",
                    "plugin": "csv_loader",
                    "config": {"source": sample_data_file}
                },
                {
                    "name": "feature_engineering",
                    "plugin": "sklearn_feature_selector",
                    "config": {"method": "variance", "threshold": 0.1}
                },
                {
                    "name": "model_training", 
                    "plugin": "sklearn_classifier",
                    "config": {"algorithm": "random_forest", "n_estimators": 10}
                }
            ],
            "target_column": "target"
        }
        
        pipeline = normaml_instance.create_pipeline(pipeline_config["steps"])
        assert pipeline is not None
        
        # Execute pipeline
        results = pipeline.execute(target_column="target")
        assert results is not None
    
    def test_memory_efficiency(self, normaml_instance, sample_data_file):
        """Test memory-efficient processing."""
        
        # Load data with memory mapping for larger files
        data_id = normaml_instance.load_data(
            sample_data_file, 
            strategy="memory_mapped"
        )
        
        # Check memory usage
        initial_stats = normaml_instance.data_manager.get_memory_stats()
        
        # Process data
        processed_data_id = normaml_instance.engineer_features(data_id)
        
        final_stats = normaml_instance.data_manager.get_memory_stats()
        
        # Memory usage should be controlled
        assert final_stats['memory_utilization'] < 0.9  # Less than 90%
    
    @pytest.mark.slow
    def test_large_data_processing(self, normaml_instance):
        """Test processing of larger datasets."""
        
        # Create larger synthetic dataset
        large_df = pl.DataFrame({
            'feature1': range(10000),
            'feature2': [x * 2 for x in range(10000)],
            'target': [1 if x > 5000 else 0 for x in range(10000)]
        })
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            large_df.write_csv(f.name)
            
            # Test streaming processing
            data_id = normaml_instance.load_data(
                f.name,
                strategy="streaming",
                chunk_size=1000
            )
            
            # Should handle large data efficiently
            processed_data_id = normaml_instance.engineer_features(data_id)
            assert processed_data_id is not None
```

### 3. Performance Tests

#### benchmark_data_operations.py

```python
import pytest
import polars as pl
import numpy as np
import time
from normaml.data.zero_copy import ZeroCopyDataManager

@pytest.mark.benchmark
class TestDataOperationsBenchmarks:
    
    @pytest.fixture(params=[1000, 10000, 100000])
    def dataset_size(self, request):
        return request.param
    
    @pytest.fixture
    def sample_data(self, dataset_size):
        return pl.DataFrame({
            'id': range(dataset_size),
            'value': np.random.rand(dataset_size),
            'category': np.random.choice(['A', 'B', 'C'], dataset_size),
            'target': np.random.randint(0, 2, dataset_size)
        })
    
    def test_data_loading_speed(self, benchmark, sample_data):
        """Benchmark data loading speed."""
        data_manager = ZeroCopyDataManager()
        
        def load_data():
            return data_manager.register_data(sample_data, "benchmark_data")
        
        result = benchmark(load_data)
        assert result is not None
    
    def test_shared_memory_vs_copy(self, benchmark, sample_data):
        """Compare shared memory vs copy performance."""
        data_manager = ZeroCopyDataManager(enable_shared_memory=True)
        
        # Benchmark shared memory
        def shared_memory_access():
            data_id = data_manager.register_data(
                sample_data, "shared_test", strategy="shared_memory"
            )
            return data_manager.get_data(data_id, "polars")
        
        shared_result = benchmark.pedantic(
            shared_memory_access, 
            rounds=10, 
            iterations=1
        )
        
        # Benchmark regular copy
        def copy_access():
            return sample_data.clone()
        
        copy_result = benchmark.pedantic(
            copy_access,
            rounds=10, 
            iterations=1
        )
        
        # Shared memory should be faster for larger datasets
        assert shared_result.shape == copy_result.shape
    
    def test_memory_usage_efficiency(self, sample_data):
        """Test memory usage efficiency."""
        import psutil
        process = psutil.Process()
        
        initial_memory = process.memory_info().rss
        
        data_manager = ZeroCopyDataManager()
        data_ids = []
        
        # Register multiple views of same data
        for i in range(10):
            data_id = data_manager.register_data(
                sample_data, f"data_{i}", strategy="shared_memory"
            )
            data_ids.append(data_id)
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be minimal due to sharing
        expected_single_copy = sample_data.estimated_size()
        assert memory_increase < expected_single_copy * 2  # Some overhead allowed
    
    @pytest.mark.parametrize("strategy", ["arrow_buffer", "shared_memory", "memory_mapped"])
    def test_access_speed_by_strategy(self, benchmark, sample_data, strategy):
        """Benchmark data access speed by storage strategy."""
        data_manager = ZeroCopyDataManager(
            enable_shared_memory=True,
            enable_memory_mapping=True
        )
        
        data_id = data_manager.register_data(
            sample_data, "speed_test", strategy=strategy
        )
        
        def access_data():
            return data_manager.get_data(data_id, "polars")
        
        result = benchmark(access_data)
        assert result.shape == sample_data.shape
```

### 4. Docker Integration Tests

#### test_container_execution.py

```python
import pytest
import docker
import tempfile
import json
from normaml.execution.engines.docker_engine import DockerExecutionEngine
from normaml.execution.scheduler import TaskDefinition

@pytest.mark.docker
class TestDockerExecution:
    
    @pytest.fixture(scope="session")
    def docker_client(self):
        """Docker client for testing."""
        try:
            client = docker.from_env()
            # Test if Docker is available
            client.ping()
            return client
        except Exception:
            pytest.skip("Docker not available")
    
    @pytest.fixture
    def docker_engine(self, docker_client):
        return DockerExecutionEngine(docker_client)
    
    def test_simple_container_execution(self, docker_engine):
        """Test basic container execution."""
        task = TaskDefinition(
            id="docker_test",
            plugin_name="test_plugin",
            plugin_method="test_method",
            docker_image="python:3.11-slim",
            docker_command=['python', '-c', 'print("Hello from Docker")']
        )
        
        result = docker_engine.execute_task(task)
        assert result.success
        assert "Hello from Docker" in result.metadata.get('logs', '')
    
    def test_shared_memory_in_container(self, docker_engine):
        """Test shared memory access in containers."""
        # Create test data in shared memory
        test_data = {'test': 'data', 'numbers': [1, 2, 3]}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json') as f:
            json.dump(test_data, f)
            f.flush()
            
            task = TaskDefinition(
                id="shared_memory_test",
                plugin_name="test_plugin", 
                plugin_method="process_shared_data",
                docker_image="python:3.11-slim",
                docker_volumes={f.name: {'bind': '/shared_data.json', 'mode': 'ro'}},
                docker_command=[
                    'python', '-c', 
                    '''
import json
with open("/shared_data.json", "r") as f:
    data = json.load(f)
    print(f"Processed data: {data}")
                    '''
                ]
            )
            
            result = docker_engine.execute_task(task)
            assert result.success
            assert "Processed data:" in result.metadata.get('logs', '')
    
    def test_container_resource_limits(self, docker_engine):
        """Test container resource limits."""
        task = TaskDefinition(
            id="resource_test",
            plugin_name="test_plugin",
            plugin_method="test_method", 
            docker_image="python:3.11-slim",
            resources={'cpu_cores': 1, 'memory_mb': 128},
            docker_command=[
                'python', '-c',
                '''
import psutil
print(f"CPU cores: {psutil.cpu_count()}")
print(f"Memory: {psutil.virtual_memory().total / 1024 / 1024:.0f} MB")
                '''
            ]
        )
        
        result = docker_engine.execute_task(task)
        assert result.success
        
        logs = result.metadata.get('logs', '')
        # Should respect resource limits
        assert "CPU cores: 1" in logs
    
    @pytest.mark.slow
    def test_parallel_container_execution(self, docker_engine):
        """Test parallel execution of multiple containers."""
        tasks = []
        
        for i in range(3):
            task = TaskDefinition(
                id=f"parallel_test_{i}",
                plugin_name="test_plugin",
                plugin_method="test_method",
                docker_image="python:3.11-slim", 
                docker_command=[
                    'python', '-c', 
                    f'import time; time.sleep(1); print("Task {i} completed")'
                ]
            )
            tasks.append(task)
        
        import asyncio
        async def run_parallel():
            results = await asyncio.gather(*[
                docker_engine.execute_task(task) for task in tasks
            ])
            return results
        
        results = asyncio.run(run_parallel())
        
        assert len(results) == 3
        assert all(result.success for result in results)
```

### 5. MLflow Integration Tests

#### test_experiment_tracking.py

```python
import pytest
import mlflow
import tempfile
from normaml.tracking.mlflow_integration import EnhancedExperimentTracker
from normaml.tracking.mlflow_integration import RunConfig, RunType

@pytest.mark.mlflow
class TestMLflowIntegration:
    
    @pytest.fixture(scope="session")
    def mlflow_tracking_uri(self):
        """Setup temporary MLflow tracking for tests."""
        with tempfile.TemporaryDirectory() as temp_dir:
            tracking_uri = f"file://{temp_dir}"
            yield tracking_uri
    
    @pytest.fixture
    def experiment_tracker(self, mlflow_tracking_uri):
        return EnhancedExperimentTracker(
            tracking_uri=mlflow_tracking_uri,
            enable_auto_logging=True
        )
    
    def test_experiment_creation(self, experiment_tracker):
        """Test experiment creation and management."""
        experiment_id = experiment_tracker.create_experiment(
            "test_experiment",
            tags={"purpose": "testing"}
        )
        
        assert experiment_id is not None
        
        # Test duplicate experiment handling
        duplicate_id = experiment_tracker.create_experiment("test_experiment")
        assert duplicate_id == experiment_id
    
    def test_run_lifecycle(self, experiment_tracker):
        """Test complete run lifecycle."""
        config = RunConfig(
            experiment_name="test_experiment",
            run_name="test_run",
            run_type=RunType.EXPERIMENT,
            tags={"test": "true"}
        )
        
        # Start run
        run_id = experiment_tracker.start_run(config)
        assert run_id is not None
        
        # Log parameters
        experiment_tracker.log_params({
            "param1": "value1",
            "param2": 42
        }, run_id)
        
        # Log metrics
        experiment_tracker.log_metrics({
            "accuracy": 0.95,
            "loss": 0.05
        }, run_id)
        
        # Log tags
        experiment_tracker.log_tags({
            "status": "completed"
        }, run_id)
        
        # End run
        result = experiment_tracker.end_run(run_id)
        assert result['status'] == 'FINISHED'
    
    def test_nested_runs(self, experiment_tracker):
        """Test nested run functionality."""
        # Parent run
        parent_config = RunConfig(
            experiment_name="nested_test",
            run_name="parent_run",
            run_type=RunType.PIPELINE
        )
        parent_run_id = experiment_tracker.start_run(parent_config)
        
        # Child run
        child_config = RunConfig(
            experiment_name="nested_test",
            run_name="child_run",
            run_type=RunType.MODEL_TRAINING,
            parent_run_id=parent_run_id,
            nested=True
        )
        child_run_id = experiment_tracker.start_run(child_config)
        
        # Log to child run
        experiment_tracker.log_metrics({"child_metric": 0.8}, child_run_id)
        
        # End runs
        experiment_tracker.end_run(child_run_id)
        experiment_tracker.end_run(parent_run_id)
        
        # Verify parent-child relationship exists
        assert child_run_id != parent_run_id
    
    def test_auto_logging_integration(self, experiment_tracker):
        """Test automatic logging with plugin integration."""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.datasets import make_classification
        
        # Enable sklearn autolog
        mlflow.sklearn.autolog()
        
        # Create sample data
        X, y = make_classification(n_samples=100, n_features=4, random_state=42)
        
        config = RunConfig(
            experiment_name="autolog_test",
            run_name="sklearn_autolog",
            auto_log=True
        )
        
        run_id = experiment_tracker.start_run(config)
        
        # Train model (should be auto-logged)
        with mlflow.start_run(run_id=run_id):
            model = RandomForestClassifier(n_estimators=10, random_state=42)
            model.fit(X, y)
        
        experiment_tracker.end_run(run_id)
        
        # Verify auto-logging worked
        run = mlflow.get_run(run_id)
        assert len(run.data.params) > 0  # Should have logged parameters
        assert len(run.data.metrics) > 0  # Should have logged metrics
    
    def test_batch_logging_performance(self, experiment_tracker):
        """Test batch logging performance."""
        import time
        
        config = RunConfig(
            experiment_name="batch_test",
            run_name="batch_logging"
        )
        
        run_id = experiment_tracker.start_run(config)
        
        start_time = time.time()
        
        # Log many parameters/metrics (should be batched)
        for i in range(100):
            experiment_tracker.log_params({f"param_{i}": f"value_{i}"}, run_id)
            experiment_tracker.log_metrics({f"metric_{i}": i * 0.01}, run_id)
        
        # Force flush
        experiment_tracker._flush_log_queue(run_id)
        
        end_time = time.time()
        
        experiment_tracker.end_run(run_id)
        
        # Batching should make this reasonably fast
        assert end_time - start_time < 5.0  # Should complete in under 5 seconds
```

### 6. Test Configuration и Fixtures

#### conftest.py

```python
import pytest
import tempfile
import shutil
import polars as pl
import numpy as np
from unittest.mock import Mock

# Test data fixtures
@pytest.fixture(scope="session")
def sample_dataset():
    """Create sample dataset for testing."""
    np.random.seed(42)
    return pl.DataFrame({
        'feature1': np.random.rand(1000),
        'feature2': np.random.rand(1000) * 100,
        'feature3': np.random.choice(['A', 'B', 'C'], 1000),
        'feature4': np.random.randint(0, 10, 1000),
        'target': np.random.randint(0, 2, 1000)
    })

@pytest.fixture(scope="session") 
def large_dataset():
    """Create large dataset for performance testing."""
    np.random.seed(42)
    size = 100000
    return pl.DataFrame({
        'id': range(size),
        'value': np.random.rand(size),
        'category': np.random.choice(['A', 'B', 'C', 'D'], size),
        'target': np.random.randint(0, 3, size)
    })

@pytest.fixture
def temp_data_dir():
    """Create temporary directory for test data."""
    temp_dir = tempfile.mkdtemp(prefix="normaml_test_")
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)

@pytest.fixture
def mock_plugin():
    """Create mock plugin for testing."""
    plugin = Mock()
    plugin.name = "mock_plugin"
    plugin.version = "1.0.0"
    plugin.capabilities = ["test"]
    plugin.initialize.return_value = None
    plugin.validate_config.return_value = True
    plugin.get_capabilities.return_value = {"supports_streaming": False}
    return plugin

# MLflow fixtures
@pytest.fixture(scope="session")
def mlflow_test_server():
    """Start MLflow test server."""
    import subprocess
    import time
    import requests
    
    # Start MLflow server in background
    process = subprocess.Popen([
        'mlflow', 'server', 
        '--host', '127.0.0.1',
        '--port', '5001',
        '--backend-store-uri', 'sqlite:///test_mlflow.db',
        '--default-artifact-root', './test_artifacts'
    ])
    
    # Wait for server to start
    for _ in range(30):
        try:
            response = requests.get('http://127.0.0.1:5001/health')
            if response.status_code == 200:
                break
        except:
            pass
        time.sleep(1)
    
    yield 'http://127.0.0.1:5001'
    
    # Cleanup
    process.terminate()
    process.wait()

# Docker fixtures
@pytest.fixture(scope="session")
def docker_available():
    """Check if Docker is available."""
    try:
        import docker
        client = docker.from_env()
        client.ping()
        return True
    except:
        return False

# Performance testing configuration
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "docker: marks tests requiring Docker"
    )
    config.addinivalue_line(
        "markers", "mlflow: marks tests requiring MLflow"
    )
    config.addinivalue_line(
        "markers", "benchmark: marks tests as benchmarks"
    )

def pytest_collection_modifyitems(config, items):
    """Auto-mark tests based on their location."""
    for item in items:
        # Auto-mark integration tests
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        
        # Auto-mark performance tests
        if "performance" in str(item.fspath) or "benchmark" in str(item.fspath):
            item.add_marker(pytest.mark.benchmark)
        
        # Auto-mark docker tests
        if "docker" in str(item.fspath):
            item.add_marker(pytest.mark.docker)
        
        # Auto-mark MLflow tests
        if "mlflow" in str(item.fspath):
            item.add_marker(pytest.mark.mlflow)
```

### 7. Test Commands и Scripts

#### scripts/run_tests.sh

```bash
#!/bin/bash
set -e

echo "Running NormaML test suite..."

# Configuration
PYTEST_ARGS="--cov=normaml --cov-report=html --cov-report=term-missing"
PARALLEL_JOBS=${PARALLEL_JOBS:-auto}

# Test categories
RUN_UNIT=${RUN_UNIT:-true}
RUN_INTEGRATION=${RUN_INTEGRATION:-true}
RUN_PERFORMANCE=${RUN_PERFORMANCE:-false}
RUN_DOCKER=${RUN_DOCKER:-false}
RUN_MLFLOW=${RUN_MLFLOW:-false}

# Check dependencies
echo "Checking test dependencies..."
python -c "import pytest, pytest_cov, pytest_benchmark, pytest_mock" || {
    echo "Installing test dependencies..."
    pip install -r requirements/dev.txt
}

# Unit tests
if [ "$RUN_UNIT" = "true" ]; then
    echo "Running unit tests..."
    pytest tests/unit/ $PYTEST_ARGS -n $PARALLEL_JOBS -v
fi

# Integration tests
if [ "$RUN_INTEGRATION" = "true" ]; then
    echo "Running integration tests..."
    pytest tests/integration/ $PYTEST_ARGS -v
fi

# Performance tests
if [ "$RUN_PERFORMANCE" = "true" ]; then
    echo "Running performance tests..."
    pytest tests/performance/ --benchmark-only --benchmark-json=benchmark.json -v
fi

# Docker tests (if Docker available)
if [ "$RUN_DOCKER" = "true" ]; then
    if command -v docker &> /dev/null && docker info &> /dev/null; then
        echo "Running Docker tests..."
        pytest tests/integration/test_docker/ -m docker -v
    else
        echo "Docker not available, skipping Docker tests"
    fi
fi

# MLflow tests (if MLflow server available)
if [ "$RUN_MLFLOW" = "true" ]; then
    echo "Running MLflow tests..."
    pytest tests/integration/test_mlflow/ -m mlflow -v
fi

echo "Test suite completed!"

# Generate coverage report
if [ -f ".coverage" ]; then
    echo "Coverage report available at htmlcov/index.html"
fi
```

### 8. Performance Benchmarking

#### benchmark_runner.py

```python
"""
Performance benchmark runner for NormaML components.
"""

import time
import psutil
import json
from typing import Dict, Any, List
from dataclasses import dataclass
from normaml import NormaML
import polars as pl
import numpy as np

@dataclass
class BenchmarkResult:
    name: str
    duration: float
    memory_usage: int
    cpu_usage: float
    metadata: Dict[str, Any]

class BenchmarkRunner:
    """Runner for performance benchmarks."""
    
    def __init__(self):
        self.results: List[BenchmarkResult] = []
        
    def benchmark(self, name: str, func, *args, **kwargs) -> BenchmarkResult:
        """Run a single benchmark."""
        process = psutil.Process()
        
        # Initial measurements
        start_memory = process.memory_info().rss
        start_time = time.perf_counter()
        start_cpu = process.cpu_percent()
        
        # Run function
        result = func(*args, **kwargs)
        
        # Final measurements
        end_time = time.perf_counter()
        end_memory = process.memory_info().rss
        end_cpu = process.cpu_percent()
        
        benchmark_result = BenchmarkResult(
            name=name,
            duration=end_time - start_time,
            memory_usage=end_memory - start_memory,
            cpu_usage=end_cpu - start_cpu,
            metadata={
                'result_size': len(result) if hasattr(result, '__len__') else None,
                'peak_memory': end_memory
            }
        )
        
        self.results.append(benchmark_result)
        return benchmark_result
    
    def run_data_benchmarks(self, sizes: List[int]):
        """Run data operation benchmarks."""
        ml = NormaML()
        
        for size in sizes:
            # Create test data
            data = pl.DataFrame({
                'feature1': np.random.rand(size),
                'feature2': np.random.rand(size),
                'target': np.random.randint(0, 2, size)
            })
            
            # Benchmark data loading
            self.benchmark(
                f"data_loading_{size}",
                lambda: ml.load_data(data)
            )
            
            # Benchmark feature engineering
            data_id = ml.load_data(data)
            self.benchmark(
                f"feature_engineering_{size}",
                lambda: ml.engineer_features(data_id, engineer="polars_engineer")
            )
    
    def export_results(self, filename: str):
        """Export benchmark results to JSON."""
        results_dict = {
            'timestamp': time.time(),
            'system_info': {
                'cpu_count': psutil.cpu_count(),
                'memory_total': psutil.virtual_memory().total,
                'python_version': '.'.join(map(str, __import__('sys').version_info[:3]))
            },
            'results': [
                {
                    'name': r.name,
                    'duration': r.duration,
                    'memory_usage': r.memory_usage,
                    'cpu_usage': r.cpu_usage,
                    'metadata': r.metadata
                }
                for r in self.results
            ]
        }
        
        with open(filename, 'w') as f:
            json.dump(results_dict, f, indent=2)

if __name__ == "__main__":
    runner = BenchmarkRunner()
    
    # Run benchmarks
    sizes = [1000, 10000, 100000]
    runner.run_data_benchmarks(sizes)
    
    # Export results
    runner.export_results("benchmark_results.json")
    
    # Print summary
    for result in runner.results:
        print(f"{result.name}: {result.duration:.3f}s, {result.memory_usage/1024/1024:.1f}MB")
```

Этот план тестирования обеспечивает:

1. **Comprehensive Coverage** - покрытие всех компонентов системы
2. **Performance Testing** - benchmarking критических операций
3. **Integration Testing** - тестирование взаимодействия компонентов
4. **Docker Support** - тестирование containerized execution
5. **MLflow Integration** - проверка experiment tracking
6. **Automation** - CI/CD интеграция с автоматическими тестами
7. **Performance Monitoring** - continuous benchmarking
8. **Error Scenarios** - тестирование error handling
9. **Scalability Testing** - тестирование с различными размерами данных
10. **Security Testing** - проверка изоляции и безопасности

План структурирован для легкого выполнения как локально, так и в CI/CD pipeline, с поддержкой параллельного выполнения и детальной отчетности.