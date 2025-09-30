from __future__ import annotations
"""
Canonical execution scheduler for NormaML (canonical-modules strategy).

Provides:
- TaskDefinition dataclass
- ResourceRequirements dataclass
- ResourceManager minimal implementation
- TaskScheduler with a minimal synchronous/async structure suitable for tests

This module reuses ExecutionEngine where appropriate (delegation) rather than copying
large amounts of runtime logic.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any, List, Union
import asyncio
import threading
import time
import heapq
from collections import deque, defaultdict

# Try to reuse existing ExecutionEngine implementation if available
try:
    from normaml.core.execution_engine import ExecutionEngine  # type: ignore
except Exception:
    ExecutionEngine = None  # type: ignore


@dataclass
class ResourceRequirements:
    """Lightweight resource descriptor for a task."""
    cpu_cores: int = 1
    memory_mb: int = 128


@dataclass(order=True)
class _PrioritizedItem:
    """Internal helper to order tasks by priority and submission time."""
    priority_value: int
    submitted_at: float
    task: "TaskDefinition" = field(compare=False)


@dataclass
class TaskDefinition:
    """
    Definition of a scheduled task.

    Fields:
        id: Unique task identifier.
        plugin_name: Name of plugin to execute.
        plugin_method: Method on plugin to invoke.
        priority: One of "low", "normal", "high" (default: "normal").
        dependencies: List of task ids that must complete before this task becomes ready.
        resources: Dict describing resources, e.g. {"cpu_cores": 2, "memory_mb": 512}
        docker_image: Optional docker image for containerized execution.
        docker_command: Optional list command for docker execution.
        metadata: Additional arbitrary metadata.
    """
    id: str
    plugin_name: str
    plugin_method: str
    priority: str = "normal"
    dependencies: Optional[List[str]] = None
    resources: Optional[Dict[str, Any]] = None
    docker_image: Optional[str] = None
    docker_command: Optional[List[str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ResourceManager:
    """
    Minimal resource manager.

    Tracks common limits and current allocations in a very small API surface.
    Intended to be sufficient for unit/integration tests that validate simple allocation logic.
    """

    def __init__(self, cpu_count: Optional[int] = None, total_memory_mb: Optional[int] = None):
        """
        Initialize ResourceManager.

        Args:
            cpu_count: Total CPU cores available (defaults to os.cpu_count() or 1)
            total_memory_mb: Total memory available in MB (defaults to 1024)
        """
        import os as _os

        self.cpu_count = cpu_count or max(1, (_os.cpu_count() or 1))
        self.total_memory_mb = total_memory_mb or 1024
        # tracking allocated resources
        self._allocations: Dict[str, ResourceRequirements] = {}
        self._lock = threading.RLock()

    @property
    def allocated_cpu(self) -> int:
        with self._lock:
            return sum(req.cpu_cores for req in self._allocations.values())

    @property
    def allocated_memory(self) -> int:
        """Allocated memory in bytes (approx)."""
        with self._lock:
            return sum(req.memory_mb for req in self._allocations.values()) * 1024 * 1024

    def can_allocate_resources(self, req: Union[ResourceRequirements, Dict[str, int]]) -> bool:
        """Check if requested resources can be allocated."""
        with self._lock:
            if isinstance(req, dict):
                cpu = int(req.get("cpu_cores", 1))
                mem = int(req.get("memory_mb", 128))
            else:
                cpu = req.cpu_cores
                mem = req.memory_mb

            if cpu < 0 or mem < 0:
                return False

            available_cpu = self.cpu_count - self.allocated_cpu
            available_mem_mb = self.total_memory_mb - (self.allocated_memory // (1024 * 1024))

            return (cpu <= available_cpu) and (mem <= available_mem_mb)

    def allocate_resources(self, task_id: str, req: Union[ResourceRequirements, Dict[str, int]]) -> bool:
        """Attempt to allocate resources for a task. Returns True on success."""
        with self._lock:
            if isinstance(req, dict):
                requirements = ResourceRequirements(cpu_cores=int(req.get("cpu_cores", 1)),
                                                    memory_mb=int(req.get("memory_mb", 128)))
            else:
                requirements = req

            if not self.can_allocate_resources(requirements):
                return False

            self._allocations[task_id] = requirements
            return True

    def release_resources(self, task_id: str) -> None:
        """Release previously allocated resources for a task."""
        with self._lock:
            if task_id in self._allocations:
                del self._allocations[task_id]


class TaskScheduler:
    """
    Minimal TaskScheduler.

    - Accepts TaskDefinition submissions via submit_task.
    - Maintains an internal queue and dependency graph.
    - start_scheduler launches a background async loop that dispatches ready tasks
      to registered execution engines.
    - Scheduler is intentionally minimal (synchronous semantics where practical)
      but provides the public API expected by tests.
    """

    _PRIORITY_MAP = {"low": 2, "normal": 1, "high": 0}

    def __init__(self, max_concurrent_tasks: int = 4):
        """
        Args:
            max_concurrent_tasks: Maximum number of concurrently running tasks.
        """
        self.max_concurrent_tasks = max_concurrent_tasks
        self.task_queue: List[_PrioritizedItem] = []
        self._task_map: Dict[str, TaskDefinition] = {}
        self._dependencies: Dict[str, List[str]] = {}
        self._dependents: Dict[str, List[str]] = defaultdict(list)
        self.active_tasks: Dict[str, asyncio.Task] = {}
        self.completed_tasks: List[TaskDefinition] = []
        self._queue_lock = threading.RLock()
        self._stop_event = asyncio.Event()
        self._scheduler_task: Optional[asyncio.Task] = None
        self.resource_manager: ResourceManager = ResourceManager()
        # Execution engines: keys may include 'async', 'process', 'docker'
        self.engines: Dict[str, Any] = {}
        # bookkeeping for run loop
        self._loop = asyncio.get_event_loop()

    def submit_task(self, task: TaskDefinition) -> str:
        """
        Submit a task to the scheduler.

        Returns the task id.
        """
        with self._queue_lock:
            self._task_map[task.id] = task
            deps = list(task.dependencies or [])
            self._dependencies[task.id] = deps
            for d in deps:
                self._dependents[d].append(task.id)

            # create prioritized item
            priority_value = self._PRIORITY_MAP.get(task.priority, 1)
            item = _PrioritizedItem(priority_value=priority_value, submitted_at=time.time(), task=task)
            heapq.heappush(self.task_queue, item)
            return task.id

    def _get_ready_task(self) -> Optional[TaskDefinition]:
        """
        Pick the next ready task (dependencies satisfied and resources available).

        Respects priority ordering. Returns None if no ready task is available.
        """
        with self._queue_lock:
            if not self.task_queue:
                return None

            # We need to scan the heap for first ready task (heap doesn't support easy removal).
            # For minimal implementation, pop items until we find one ready or re-push non-ready.
            temp = []
            ready_task = None
            while self.task_queue:
                item = heapq.heappop(self.task_queue)
                tid = item.task.id
                deps = self._dependencies.get(tid, [])
                # all dependencies should be completed
                all_done = all(d not in self._task_map or any(c.id == d for c in self.completed_tasks) for d in deps)
                # check resource requirements
                resources_ok = True
                if item.task.resources:
                    resources_ok = self.resource_manager.can_allocate_resources(item.task.resources)
                if all_done and resources_ok:
                    ready_task = item.task
                    break
                else:
                    temp.append(item)

            # push back unready items
            for it in temp:
                heapq.heappush(self.task_queue, it)

            return ready_task

    async def _dispatch_task(self, task: TaskDefinition) -> None:
        """
        Dispatch a task to an appropriate execution engine.

        This method chooses an engine based on available registered engines and task metadata.
        """
        # allocate resources optimistically
        allocated = False
        try:
            if task.resources:
                allocated = self.resource_manager.allocate_resources(task.id, task.resources)
                if not allocated:
                    # Requeue task for later
                    self.submit_task(task)
                    return

            # select engine
            engine = None
            if task.docker_image and "docker" in self.engines:
                engine = self.engines.get("docker")
            elif "async" in self.engines:
                engine = self.engines.get("async")
            elif "process" in self.engines:
                engine = self.engines.get("process")

            # Fallback: if we have ExecutionEngine available, wrap it
            if engine is None and ExecutionEngine is not None:
                engine = ExecutionEngine()

            # If engine provides coroutine execute_task, await it; otherwise run in executor
            result = None
            if asyncio.iscoroutinefunction(getattr(engine, "execute_task", None)):
                result = await engine.execute_task(task)  # type: ignore
            else:
                # run sync method in threadpool to avoid blocking loop
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, getattr(engine, "execute_task", lambda t: None), task)  # type: ignore

            # mark completion
            with self._queue_lock:
                self.completed_tasks.append(task)
                self._task_map.pop(task.id, None)
                # release resources
                if allocated:
                    self.resource_manager.release_resources(task.id)
                # notify dependents (they are already in queue; they'll be picked up by _get_ready_task)
        except Exception:
            # Ensure release resources on failure
            if allocated:
                self.resource_manager.release_resources(task.id)
            with self._queue_lock:
                # move task to completed to avoid blocking dependents in this minimal implementation
                self.completed_tasks.append(task)
                self._task_map.pop(task.id, None)

    async def _run(self) -> None:
        """Main scheduler loop that continuously dispatches tasks while not stopped."""
        try:
            while not self._stop_event.is_set():
                # Do not oversubscribe
                if len(self.active_tasks) >= self.max_concurrent_tasks:
                    await asyncio.sleep(0.05)
                    continue

                task = self._get_ready_task()
                if task is None:
                    # nothing ready — small sleep
                    if not self.task_queue and not self.active_tasks:
                        # nothing left to do; break loop
                        break
                    await asyncio.sleep(0.05)
                    continue

                # create asyncio.Task for dispatch
                coro = self._dispatch_task(task)
                t = asyncio.create_task(coro)
                self.active_tasks[task.id] = t

                # attach callback to remove from active_tasks when done
                def _on_done(fut: asyncio.Future, tid: str = task.id):
                    try:
                        fut.result()
                    except Exception:
                        pass
                    # remove from active
                    self.active_tasks.pop(tid, None)

                t.add_done_callback(_on_done)

            # Wait for remaining active tasks to complete before exiting run
            if self.active_tasks:
                await asyncio.wait(self.active_tasks.values())
        finally:
            # ensure state is clean
            self._scheduler_task = None
            self._stop_event.clear()

    async def start_scheduler(self) -> None:
        """
        Start the scheduler background loop.

        This method returns quickly; the scheduler runs in the background. To wait for completion,
        monitor get_queue_status() or await stop_scheduler().
        """
        if self._scheduler_task is not None and not self._scheduler_task.done():
            return  # already running
        self._stop_event.clear()
        self._scheduler_task = asyncio.create_task(self._run())

    async def stop_scheduler(self) -> None:
        """
        Stop the scheduler and wait for currently running tasks to finish.
        """
        self._stop_event.set()
        if self._scheduler_task:
            try:
                await self._scheduler_task
            except Exception:
                pass
        # wait for active tasks
        if self.active_tasks:
            await asyncio.wait(self.active_tasks.values())

    def get_queue_status(self) -> Dict[str, int]:
        """
        Return a summary of queue status.
        """
        with self._queue_lock:
            return {
                "queued": len(self.task_queue),
                "active_tasks": len(self.active_tasks),
                "completed": len(self.completed_tasks)
            }

    def register_execution_engines(self, **engines: Any) -> None:
        """
        Register execution engine adapters.

        Example:
            register_execution_engines(async=AsyncExecutionEngine(), process=ProcessExecutionEngine(), docker=DockerEngine())
        """
        self.engines.update(engines)