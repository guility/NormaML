"""
Canonical execution engines package.

Exports:
- AsyncExecutionEngine: async adapter that uses a plugin manager to initialize plugins and call async methods.
- ProcessExecutionEngine: sync adapter that runs plugin calls in a thread/process executor.

These are thin adapters that delegate to plugin manager where appropriate.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Optional, Dict, List
import asyncio
import time
import concurrent.futures

# Attempt to reuse TaskDefinition type for typing clarity (optional)
try:
    from normaml.execution.scheduler import TaskDefinition  # type: ignore
except Exception:
    TaskDefinition = Any  # type: ignore

# Reuse ExecutionResult from core if available for interoperability
try:
    from normaml.core.execution_engine import ExecutionResult as CoreExecutionResult  # type: ignore
except Exception:
    CoreExecutionResult = None  # type: ignore


@dataclass
class ExecutionResult:
    """
    Adapter execution result returned by engine adapters.

    Fields:
        success: whether execution succeeded
        result_data: returned data from plugin method (if any)
        error: exception instance in case of failure
        execution_time: float seconds
        metadata: optional dict for engine-specific metadata
    """
    success: bool
    result_data: Any = None
    error: Optional[BaseException] = None
    execution_time: float = 0.0
    metadata: Dict[str, Any] = None


class AsyncExecutionEngine:
    """
    Thin async adapter engine.

    Usage:
        engine = AsyncExecutionEngine(max_concurrent=10)
        engine.set_dependencies(plugin_manager, data_manager)
        result = await engine.execute_task(task)

    The adapter expects a plugin manager with an async `initialize_plugin(name, config?)` method
    that returns an object where the actual callable is available (e.g., plugin.instance.method).
    """

    def __init__(self, max_concurrent: int = 4):
        self.max_concurrent = max_concurrent
        self._plugin_manager = None
        self._data_manager = None
        self._semaphore = asyncio.Semaphore(max_concurrent)

    def set_dependencies(self, plugin_manager: Any, data_manager: Any = None) -> None:
        """
        Provide required external dependencies.

        Args:
            plugin_manager: object exposing initialize_plugin (async) to obtain plugin instances
            data_manager: optional data manager passed to plugins
        """
        self._plugin_manager = plugin_manager
        self._data_manager = data_manager

    async def execute_task(self, task: TaskDefinition) -> ExecutionResult:
        """
        Execute a single task asynchronously.

        Returns:
            ExecutionResult
        """
        start = time.time()
        async with self._semaphore:
            try:
                if self._plugin_manager is None:
                    raise RuntimeError("Plugin manager not set on AsyncExecutionEngine")

                # plugin initialization may be async
                init = getattr(self._plugin_manager, "initialize_plugin", None)
                if asyncio.iscoroutinefunction(init):
                    plugin_wrapper = await init(task.plugin_name, {})  # type: ignore
                else:
                    # support sync initialize_plugin
                    loop = asyncio.get_running_loop()
                    plugin_wrapper = await loop.run_in_executor(None, init, task.plugin_name, {})  # type: ignore

                # plugin_wrapper may expose the callable as .instance or be the callable itself
                impl = getattr(plugin_wrapper, "instance", plugin_wrapper)

                method = getattr(impl, task.plugin_method, None)
                if method is None:
                    raise AttributeError(f"Plugin '{task.plugin_name}' has no method '{task.plugin_method}'")

                if asyncio.iscoroutinefunction(method):
                    result = await method()
                else:
                    loop = asyncio.get_running_loop()
                    result = await loop.run_in_executor(None, method)

                elapsed = time.time() - start
                return ExecutionResult(success=True, result_data=result, execution_time=elapsed, metadata={})
            except Exception as e:
                elapsed = time.time() - start
                return ExecutionResult(success=False, error=e, execution_time=elapsed, metadata={})

    async def execute_parallel(self, tasks: List[Any], execution_type: str = "async") -> List[Any]:
        """
        Minimal parallel executor for async engine.

        If tasks are coroutine objects or awaitables, gathers them limiting concurrency.
        If tasks are callables, runs them in an executor.

        Returns a list of results (raw return values).
        """
        # If tasks are coroutine objects, schedule them with semaphore-limited concurrency
        async def _wrap_coro(coro):
            async with self._semaphore:
                return await coro

        coros = []
        for t in tasks:
            if asyncio.iscoroutine(t) or asyncio.isfuture(t):
                coros.append(_wrap_coro(t))
            elif callable(t):
                # run callable in threadpool
                loop = asyncio.get_running_loop()
                coros.append(loop.run_in_executor(None, t))
            else:
                # literal value
                coros.append(asyncio.sleep(0, result=t))

        results = await asyncio.gather(*coros, return_exceptions=False)
        # Normalize results: if they are CoreExecutionResult wrappers, return .result where appropriate
        normalized = []
        for r in results:
            if CoreExecutionResult is not None and isinstance(r, CoreExecutionResult):
                normalized.append(r.result)
            else:
                normalized.append(r)
        return normalized


class ProcessExecutionEngine:
    """
    Synchronous execution adapter that runs plugin calls in a thread/process executor.

    Similar API to AsyncExecutionEngine but execute_task is synchronous (blocking).
    """

    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self._plugin_manager = None
        self._data_manager = None
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)

    def set_dependencies(self, plugin_manager: Any, data_manager: Any = None) -> None:
        """Provide plugin manager and optional data manager."""
        self._plugin_manager = plugin_manager
        self._data_manager = data_manager

    def execute_task(self, task: TaskDefinition) -> ExecutionResult:
        """
        Execute a task synchronously.

        Returns:
            ExecutionResult
        """
        start = time.time()
        try:
            if self._plugin_manager is None:
                raise RuntimeError("Plugin manager not set on ProcessExecutionEngine")

            init = getattr(self._plugin_manager, "initialize_plugin", None)
            if callable(init):
                plugin_wrapper = init(task.plugin_name, {})
            else:
                plugin_wrapper = None

            impl = getattr(plugin_wrapper, "instance", plugin_wrapper)
            method = getattr(impl, task.plugin_method, None)
            if method is None:
                raise AttributeError(f"Plugin '{task.plugin_name}' has no method '{task.plugin_method}'")

            # Run method in threadpool if it's potentially blocking
            if asyncio.iscoroutinefunction(method):
                # Run coroutine in new event loop inside thread
                loop = asyncio.new_event_loop()

                def _run_coro():
                    try:
                        asyncio.set_event_loop(loop)
                        return loop.run_until_complete(method())
                    finally:
                        loop.close()

                future = self._executor.submit(_run_coro)
                result = future.result()
            else:
                result = self._executor.submit(method).result()

            elapsed = time.time() - start
            return ExecutionResult(success=True, result_data=result, execution_time=elapsed, metadata={})
        except Exception as e:
            elapsed = time.time() - start
            return ExecutionResult(success=False, error=e, execution_time=elapsed, metadata={})