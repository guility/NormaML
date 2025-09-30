"""
Execution engine for parallel processing with Docker and shared memory support.

This module implements the ExecutionEngine that manages parallel execution of tasks,
plugins in Docker containers, and coordinates shared memory access across processes.
"""

import asyncio
import logging
import os
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, Future, as_completed
from typing import Callable, Any, List, Dict, Optional, Union, Awaitable
from pathlib import Path

try:
    import docker
    from docker.models.containers import Container
    from docker.client import DockerClient
    DOCKER_AVAILABLE = True
except ImportError:
    DOCKER_AVAILABLE = False
    # Type placeholders when docker is not available
    Container = Any
    DockerClient = Any
    docker = None  # type: ignore

from .data_manager import SharedMemoryDataManager
from .interfaces import SharedMemoryHandle


logger = logging.getLogger(__name__)


class ExecutionResult:
    """Result of an execution operation."""
    
    def __init__(self, task_id: str, success: bool, result: Any = None, 
                 error: Optional[Exception] = None, execution_time: float = 0.0):
        self.task_id = task_id
        self.success = success
        self.result = result
        self.error = error
        self.execution_time = execution_time
        self.timestamp = time.time()
    
    def __repr__(self) -> str:
        status = "SUCCESS" if self.success else "FAILED"
        return f"ExecutionResult(task_id='{self.task_id}', status={status}, time={self.execution_time:.2f}s)"


class DockerExecutionConfig:
    """Configuration for Docker-based execution."""
    
    def __init__(self, image: str, command: str, environment: Optional[Dict[str, str]] = None,
                 volumes: Optional[Dict[str, Dict[str, str]]] = None, 
                 working_dir: Optional[str] = None, shm_size: str = "2g",
                 timeout: Optional[int] = None, remove: bool = True):
        self.image = image
        self.command = command
        self.environment = environment or {}
        self.volumes = volumes or {}
        self.working_dir = working_dir
        self.shm_size = shm_size
        self.timeout = timeout
        self.remove = remove


class ExecutionEngine:
    """
    Engine for managing parallel execution with Docker and shared memory support.
    
    The ExecutionEngine provides:
    - Parallel task execution using threads or processes
    - Docker container execution for isolated plugin runs
    - Shared memory integration for zero-copy data transfer
    - Resource cleanup and lifecycle management
    """
    
    def __init__(self, max_workers: Optional[int] = None, 
                 enable_docker: bool = True, docker_timeout: int = 3600):
        """
        Initialize the ExecutionEngine.
        
        Args:
            max_workers: Maximum number of worker threads/processes
            enable_docker: Whether to enable Docker support
            docker_timeout: Default timeout for Docker operations in seconds
        """
        self.max_workers = max_workers or min(32, (os.cpu_count() or 1) + 4)
        self.enable_docker = enable_docker and DOCKER_AVAILABLE
        self.docker_timeout = docker_timeout
        
        # Execution pools
        self.process_pool = ProcessPoolExecutor(max_workers=self.max_workers)
        self.thread_pool = ThreadPoolExecutor(max_workers=self.max_workers)
        
        # Docker client
        self.docker_client: Optional[Any] = None
        if self.enable_docker and DOCKER_AVAILABLE and docker is not None:
            try:
                self.docker_client = docker.from_env()
                logger.info("Docker client initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize Docker client: {e}")
                self.enable_docker = False
        
        # Data manager integration
        self.data_manager: Optional[SharedMemoryDataManager] = None
        
        # Container management
        self.active_containers: Dict[str, Any] = {}
        self.container_lock = threading.RLock()
        
        # Task tracking
        self.running_tasks: Dict[str, Future] = {}
        self.task_counter = 0
        self.task_lock = threading.Lock()
        
        logger.info(f"ExecutionEngine initialized with max_workers={self.max_workers}, "
                   f"docker_enabled={self.enable_docker}")
    
    def set_data_manager(self, data_manager: SharedMemoryDataManager) -> None:
        """
        Set the data manager for shared memory operations.
        
        Args:
            data_manager: SharedMemoryDataManager instance
        """
        self.data_manager = data_manager
        logger.info("Data manager set for ExecutionEngine")
    
    async def execute_parallel(self, tasks: List[Callable], 
                              execution_type: str = "thread",
                              task_names: Optional[List[str]] = None) -> List[ExecutionResult]:
        """
        Execute tasks in parallel using threads or processes.
        
        Args:
            tasks: List of callable tasks to execute
            execution_type: Either "thread" or "process"
            task_names: Optional names for tasks (for tracking)
            
        Returns:
            List of ExecutionResult objects
            
        Raises:
            ValueError: If execution_type is not supported
        """
        if execution_type not in ["thread", "process"]:
            raise ValueError(f"Unsupported execution type: {execution_type}")
        
        if not tasks:
            return []
        
        task_names = task_names or [f"task_{i}" for i in range(len(tasks))]
        
        if len(task_names) != len(tasks):
            raise ValueError("Number of task names must match number of tasks")
        
        logger.info(f"Starting parallel execution of {len(tasks)} tasks using {execution_type}")
        
        # Choose executor
        executor = self.thread_pool if execution_type == "thread" else self.process_pool
        
        # Submit tasks
        loop = asyncio.get_event_loop()
        futures_to_names = {}
        
        for task, name in zip(tasks, task_names):
            future = loop.run_in_executor(executor, self._execute_task_with_timing, task, name)
            futures_to_names[future] = name
        
        # Collect results
        results = []
        for future in asyncio.as_completed(futures_to_names.keys()):
            try:
                result = await future
                results.append(result)
            except Exception as e:
                task_name = futures_to_names[future]
                error_result = ExecutionResult(
                    task_id=task_name,
                    success=False,
                    error=e,
                    execution_time=0.0
                )
                results.append(error_result)
                logger.error(f"Task '{task_name}' failed: {e}")
        
        logger.info(f"Parallel execution completed. Success: {sum(1 for r in results if r.success)}/{len(results)}")
        return results
    
    def _execute_task_with_timing(self, task: Callable, task_name: str) -> ExecutionResult:
        """
        Execute a single task with timing information.
        
        Args:
            task: Callable task to execute
            task_name: Name of the task
            
        Returns:
            ExecutionResult with timing information
        """
        start_time = time.time()
        try:
            result = task()
            execution_time = time.time() - start_time
            return ExecutionResult(
                task_id=task_name,
                success=True,
                result=result,
                execution_time=execution_time
            )
        except Exception as e:
            execution_time = time.time() - start_time
            return ExecutionResult(
                task_id=task_name,
                success=False,
                error=e,
                execution_time=execution_time
            )
    
    def execute_plugin_in_docker(self, plugin_name: str, config: DockerExecutionConfig,
                                data_id: Optional[str] = None,
                                container_id: Optional[str] = None) -> ExecutionResult:
        """
        Execute a plugin in a Docker container with shared memory support.
        
        Args:
            plugin_name: Name of the plugin
            config: Docker execution configuration
            data_id: Optional data identifier for shared memory
            container_id: Optional custom container identifier
            
        Returns:
            ExecutionResult with container execution details
            
        Raises:
            RuntimeError: If Docker is not available
        """
        if not self.enable_docker:
            raise RuntimeError("Docker support is not available")
        
        if container_id is None:
            with self.task_lock:
                self.task_counter += 1
                container_id = f"{plugin_name}_{self.task_counter}_{int(time.time())}"
        
        logger.info(f"Executing plugin '{plugin_name}' in Docker container '{container_id}'")
        
        start_time = time.time()
        
        try:
            # Prepare volumes and environment
            volumes = config.volumes.copy()
            environment = config.environment.copy()
            shared_data_info = None
            
            # Setup shared memory if data_id is provided
            if data_id and self.data_manager:
                shared_data_info = self._setup_shared_data_for_container(data_id, container_id)
                
                # Add shared data volume mount
                volumes[shared_data_info['path']] = {
                    'bind': f'/shared_data/{data_id}.mmap',
                    'mode': 'ro'
                }
                
                # Add environment variables for shared data access
                environment.update(shared_data_info['environment_vars'])
            
            # Add /dev/shm volume for shared memory support
            volumes['/dev/shm'] = {'bind': '/dev/shm', 'mode': 'rw'}
            
            # Prepare container configuration
            container_config = {
                'image': config.image,
                'command': config.command,
                'environment': environment,
                'volumes': volumes,
                'detach': True,
                'remove': config.remove,
                'shm_size': config.shm_size,
                'name': container_id
            }
            
            if config.working_dir:
                container_config['working_dir'] = config.working_dir
            
            # Run container
            if self.docker_client is None:
                raise RuntimeError("Docker client is not initialized")
            
            container = self.docker_client.containers.run(**container_config)
            
            with self.container_lock:
                self.active_containers[container_id] = container
            
            # Wait for completion
            try:
                result = container.wait(timeout=config.timeout or self.docker_timeout)
                logs = container.logs().decode('utf-8')
                
                execution_time = time.time() - start_time
                success = result['StatusCode'] == 0
                
                if success:
                    logger.info(f"Container '{container_id}' completed successfully in {execution_time:.2f}s")
                else:
                    logger.error(f"Container '{container_id}' failed with exit code {result['StatusCode']}")
                
                return ExecutionResult(
                    task_id=container_id,
                    success=success,
                    result={
                        'exit_code': result['StatusCode'],
                        'logs': logs,
                        'shared_data_info': shared_data_info,
                        'container_id': container_id
                    },
                    execution_time=execution_time
                )
                
            except Exception as e:
                logger.error(f"Error waiting for container '{container_id}': {e}")
                # Try to stop the container
                try:
                    container.stop(timeout=10)
                except:
                    pass
                
                execution_time = time.time() - start_time
                return ExecutionResult(
                    task_id=container_id,
                    success=False,
                    error=e,
                    execution_time=execution_time
                )
            
            finally:
                # Clean up container reference
                with self.container_lock:
                    self.active_containers.pop(container_id, None)
        
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Failed to execute plugin '{plugin_name}' in Docker: {e}")
            return ExecutionResult(
                task_id=container_id,
                success=False,
                error=e,
                execution_time=execution_time
            )
    
    def _setup_shared_data_for_container(self, data_id: str, container_id: str) -> Dict[str, Any]:
        """
        Setup shared data access for a container.
        
        Args:
            data_id: Data identifier
            container_id: Container identifier
            
        Returns:
            Shared data configuration
        """
        if not self.data_manager:
            raise RuntimeError("Data manager not set")
        
        # Prepare data for container sharing
        shared_data_info = self.data_manager.share_across_containers(data_id, [container_id])
        
        logger.debug(f"Setup shared data '{data_id}' for container '{container_id}'")
        return shared_data_info
    
    def stop_container(self, container_id: str, timeout: int = 10) -> bool:
        """
        Stop a running container.
        
        Args:
            container_id: Container identifier
            timeout: Timeout for graceful stop
            
        Returns:
            True if container was stopped successfully
        """
        with self.container_lock:
            container = self.active_containers.get(container_id)
            if not container:
                logger.warning(f"Container '{container_id}' not found in active containers")
                return False
        
        try:
            container.stop(timeout=timeout)
            logger.info(f"Container '{container_id}' stopped successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to stop container '{container_id}': {e}")
            return False
    
    def get_container_logs(self, container_id: str) -> Optional[str]:
        """
        Get logs from a container.
        
        Args:
            container_id: Container identifier
            
        Returns:
            Container logs or None if not found
        """
        with self.container_lock:
            container = self.active_containers.get(container_id)
            if not container:
                return None
        
        try:
            return container.logs().decode('utf-8')
        except Exception as e:
            logger.error(f"Failed to get logs from container '{container_id}': {e}")
            return None
    
    def list_active_containers(self) -> List[str]:
        """
        List all active container IDs.
        
        Returns:
            List of active container identifiers
        """
        with self.container_lock:
            return list(self.active_containers.keys())
    
    def cleanup_containers(self) -> None:
        """Stop and cleanup all active containers."""
        logger.info("Cleaning up all active containers")
        
        with self.container_lock:
            containers_to_cleanup = list(self.active_containers.items())
            
        for container_id, container in containers_to_cleanup:
            try:
                logger.debug(f"Stopping container '{container_id}'")
                container.stop(timeout=10)
                container.remove()
                logger.debug(f"Container '{container_id}' cleaned up")
            except Exception as e:
                logger.error(f"Error cleaning up container '{container_id}': {e}")
        
        with self.container_lock:
            self.active_containers.clear()
        
        logger.info("Container cleanup completed")
    
    def shutdown(self) -> None:
        """Shutdown the execution engine and cleanup resources."""
        logger.info("Shutting down ExecutionEngine")
        
        # Stop all containers
        self.cleanup_containers()
        
        # Shutdown executor pools
        logger.info("Shutting down thread pool")
        self.thread_pool.shutdown(wait=True)
        
        logger.info("Shutting down process pool")
        self.process_pool.shutdown(wait=True)
        
        # Close Docker client
        if self.docker_client:
            try:
                self.docker_client.close()
            except Exception as e:
                logger.error(f"Error closing Docker client: {e}")
        
        logger.info("ExecutionEngine shutdown completed")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.shutdown()
    
    def __del__(self):
        """Destructor with cleanup."""
        try:
            self.shutdown()
        except Exception:
            pass  # Ignore cleanup errors during destruction