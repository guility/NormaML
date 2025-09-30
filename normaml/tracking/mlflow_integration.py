from __future__ import annotations
"""
Canonical MLflow integration wrapper.

Provides:
- RunConfig dataclass
- RunType enum
- EnhancedExperimentTracker that wraps/delegates to core.ExperimentTracker and MlflowClient

This module deliberately delegates experiment lifecycle management to the existing
normaml.core.experiment_tracker.ExperimentTracker while using MlflowClient for direct
log operations that accept an explicit run_id.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Any
import logging

# Try to import mlflow client and core ExperimentTracker for delegation
try:
    from mlflow.tracking import MlflowClient
    import mlflow
    MLFLOW_AVAILABLE = True
except Exception:
    MlflowClient = None  # type: ignore
    mlflow = None  # type: ignore
    MLFLOW_AVAILABLE = False

try:
    from normaml.core.experiment_tracker import ExperimentTracker
except Exception:
    ExperimentTracker = None  # type: ignore

logger = logging.getLogger(__name__)


class RunType(str, Enum):
    """Enumeration of run types for RunConfig."""
    EXPERIMENT = "EXPERIMENT"
    PIPELINE = "PIPELINE"
    MODEL_TRAINING = "MODEL_TRAINING"


@dataclass
class RunConfig:
    """
    Configuration for starting an MLflow run.

    Fields:
        experiment_name: Name of experiment to create/use
        run_name: Optional human-readable run name
        run_type: RunType enum value
        tags: Optional tags dictionary
        parent_run_id: Optional parent run id (for nested runs)
        nested: Whether the run should be a nested run (default False)
        auto_log: Whether to enable framework autologging for this run (default False)
        extra: Additional metadata
    """
    experiment_name: str
    run_name: Optional[str] = None
    run_type: Optional[RunType] = RunType.EXPERIMENT
    tags: Optional[Dict[str, str]] = None
    parent_run_id: Optional[str] = None
    nested: bool = False
    auto_log: bool = False
    extra: Optional[Dict[str, Any]] = None


class EnhancedExperimentTracker:
    """
    Enhanced experiment tracker that provides a canonical MLflow integration surface.

    This class:
    - Manages MlflowClient for explicit run_id-based logging
    - Delegates experiment/run lifecycle to normaml.core.experiment_tracker.ExperimentTracker
    - Provides convenience methods used in the test-plan
    """

    def __init__(self, tracking_uri: Optional[str] = None, enable_auto_logging: bool = True):
        """
        Initialize the enhanced tracker.

        Args:
            tracking_uri: MLflow tracking URI (file://, http://, etc.)
            enable_auto_logging: Whether to enable automatic framework autologging when requested
        """
        self.tracking_uri = tracking_uri
        self.enable_auto_logging = enable_auto_logging
        self._client = None
        self._trackers: Dict[str, "ExperimentTracker"] = {}
        self._run_to_tracker: Dict[str, "ExperimentTracker"] = {}

        # Setup Mlflow client if available
        if MLFLOW_AVAILABLE and MlflowClient is not None:
            try:
                if tracking_uri and mlflow is not None:
                    mlflow.set_tracking_uri(tracking_uri)
                self._client = MlflowClient()  # type: ignore
            except Exception as e:
                logger.warning(f"Failed to initialize MlflowClient: {e}")
                self._client = None

    def create_experiment(self, name: str, tags: Optional[Dict[str, str]] = None) -> str:
        """
        Create or get an experiment by name.

        Returns:
            experiment_id: str
        """
        # Prefer using MlflowClient directly for deterministic behaviour
        if self._client is not None:
            exp = self._client.get_experiment_by_name(name)
            if exp is not None:
                return exp.experiment_id
            exp_id = self._client.create_experiment(name)
            # Optionally set tags via client (MlflowClient has set_experiment_tag in newer versions)
            if tags:
                try:
                    for k, v in (tags or {}).items():
                        # set tag on experiment via client if available; fallback to setting on first run
                        try:
                            self._client.set_experiment_tag(exp_id, k, v)  # type: ignore
                        except Exception:
                            pass
                except Exception:
                    pass
            return exp_id

        # Fallback: use core ExperimentTracker to create experiment by instantiating it
        if ExperimentTracker is None:
            raise RuntimeError("No MLflow support available to create experiment")
        tracker = ExperimentTracker(name, tracking_uri=self.tracking_uri, auto_create=True)
        # Store tracker for later reuse
        self._trackers[name] = tracker
        return tracker.experiment_id  # type: ignore

    def _get_or_create_tracker(self, experiment_name: str) -> "ExperimentTracker":
        """
        Return an ExperimentTracker instance for the experiment_name, creating it if necessary.
        """
        if experiment_name in self._trackers:
            return self._trackers[experiment_name]
        if ExperimentTracker is None:
            raise RuntimeError("Core ExperimentTracker not available")
        tracker = ExperimentTracker(experiment_name, tracking_uri=self.tracking_uri, auto_create=True)
        self._trackers[experiment_name] = tracker
        return tracker

    def start_run(self, config: RunConfig) -> str:
        """
        Start a run according to RunConfig.

        Returns:
            run_id: str
        """
        # Ensure experiment exists and get its tracker
        tracker = self._get_or_create_tracker(config.experiment_name)

        # Optionally enable autologging for supported frameworks
        if config.auto_log and self.enable_auto_logging and mlflow is not None:
            try:
                # Best-effort: enable sklearn autolog if available and tests expect it
                try:
                    mlflow.sklearn.autolog()
                except Exception:
                    pass
                try:
                    mlflow.pytorch.autolog()
                except Exception:
                    pass
                try:
                    mlflow.tensorflow.autolog()
                except Exception:
                    pass
            except Exception:
                # Ignore autolog failures; not critical for lifecycle
                pass

        # Start run using core tracker (handles nested runs)
        run_id = tracker.start_run(run_name=config.run_name, nested=config.nested, tags=config.tags)
        # Map run to tracker for later end/log operations
        self._run_to_tracker[run_id] = tracker
        # If parent_run_id supplied, attempt to set relationship using tags
        if config.parent_run_id:
            try:
                if self._client is not None:
                    # Add tag pointing to parent
                    self._client.set_tag(run_id, "parent_run_id", config.parent_run_id)  # type: ignore
                else:
                    tracker.set_tag("parent_run_id", config.parent_run_id)
            except Exception:
                pass
        return run_id

    def log_params(self, params: Dict[str, Any], run_id: str) -> None:
        """
        Log parameters for a run identified by run_id.

        Args:
            params: dict of parameters
            run_id: target run id
        """
        if self._client is not None:
            try:
                for k, v in params.items():
                    self._client.log_param(run_id, k, str(v))  # type: ignore
                return
            except Exception as e:
                logger.debug(f"Client-based log_param failed: {e}")
        # Fallback: attempt to use the tracker associated with run_id by temporarily activating it
        tracker = self._run_to_tracker.get(run_id)
        if tracker is not None:
            # Use mlflow.set_tag/params by starting a run context if necessary
            try:
                # Use mlflow.start_run to set active run and call tracker.log_params which uses active run
                if mlflow is not None:
                    mlflow.start_run(run_id=run_id)
                    tracker.log_params(params)
                    mlflow.end_run()
                    return
            except Exception:
                pass
        # Last-resort: no-op but log warning
        logger.warning(f"Could not log params for run {run_id}: client and tracker methods failed")

    def log_metrics(self, metrics: Dict[str, float], run_id: str) -> None:
        """
        Log multiple metrics to a run.

        Args:
            metrics: mapping of metric name to float value
            run_id: target run id
        """
        if self._client is not None:
            try:
                for k, v in metrics.items():
                    # MlflowClient.log_metric signature: log_metric(run_id, key, value, timestamp=None, step=None)
                    self._client.log_metric(run_id, k, float(v))  # type: ignore
                return
            except Exception as e:
                logger.debug(f"Client-based log_metric failed: {e}")
        tracker = self._run_to_tracker.get(run_id)
        if tracker is not None:
            try:
                if mlflow is not None:
                    mlflow.start_run(run_id=run_id)
                    tracker.log_metrics(metrics)
                    mlflow.end_run()
                    return
            except Exception:
                pass
        logger.warning(f"Could not log metrics for run {run_id}: client and tracker methods failed")

    def log_tags(self, tags: Dict[str, str], run_id: str) -> None:
        """
        Log tags for a run.

        Args:
            tags: mapping of tag name to value
            run_id: target run id
        """
        if self._client is not None:
            try:
                for k, v in tags.items():
                    try:
                        self._client.set_tag(run_id, k, v)  # type: ignore
                    except Exception:
                        # set_tag may not be available in some client versions; fallback to mlflow.set_tag
                        if mlflow is not None:
                            mlflow.set_tag(k, v)
                return
            except Exception as e:
                logger.debug(f"Client-based set_tag failed: {e}")
        tracker = self._run_to_tracker.get(run_id)
        if tracker is not None:
            try:
                if mlflow is not None:
                    mlflow.start_run(run_id=run_id)
                    tracker.set_tags(tags)
                    mlflow.end_run()
                    return
            except Exception:
                pass
        logger.warning(f"Could not log tags for run {run_id}: client and tracker methods failed")

    def end_run(self, run_id: str) -> Dict[str, Any]:
        """
        End a run and return a small summary dict (status, run_id).

        Returns:
            dict with at least 'status' key.
        """
        tracker = self._run_to_tracker.get(run_id)
        try:
            if tracker is not None:
                # Use tracker to end run (handles nested runs)
                tracker.end_run()
            else:
                # Fallback: call mlflow.end_run if available
                if mlflow is not None:
                    mlflow.end_run()
        except Exception as e:
            logger.debug(f"Error ending run via tracker: {e}")

        # Attempt to retrieve run status via client
        if self._client is not None:
            try:
                run = self._client.get_run(run_id)  # type: ignore
                status = getattr(run.info, "status", "FINISHED")
                return {"run_id": run_id, "status": status}
            except Exception:
                pass

        # Fallback: return best-effort status
        return {"run_id": run_id, "status": "FINISHED"}

    def _flush_log_queue(self, run_id: str) -> None:
        """
        Flush any internal buffering. For this minimal implementation this is a no-op,
        but is provided to satisfy the test-plan interface for batching.
        """
        # If client had a batching API we would call it here. For now, no-op.
        return