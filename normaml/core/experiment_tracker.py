"""
Experiment tracking with MLflow integration.

This module implements the ExperimentTracker that provides comprehensive
experiment tracking including parameters, metrics, artifacts, and shared memory usage.
"""

import logging
import os
import time
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import threading

try:
    import mlflow
    import mlflow.tracking
    from mlflow.tracking import MlflowClient
    from mlflow.entities import Run, Experiment
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    # Type placeholders
    MlflowClient = Any
    Run = Any
    Experiment = Any
    mlflow = None  # type: ignore


logger = logging.getLogger(__name__)


class ExperimentTracker:
    """
    MLflow-based experiment tracking with shared memory metrics support.
    
    The ExperimentTracker provides:
    - MLflow integration for experiment tracking
    - Parameter, metric, and artifact logging
    - Shared memory usage tracking
    - Model versioning and registration
    - Run lifecycle management
    """
    
    def __init__(self, experiment_name: str, tracking_uri: Optional[str] = None,
                 artifact_location: Optional[str] = None, auto_create: bool = True):
        """
        Initialize the ExperimentTracker.
        
        Args:
            experiment_name: Name of the MLflow experiment
            tracking_uri: MLflow tracking server URI (defaults to local)
            artifact_location: Location to store artifacts
            auto_create: Whether to auto-create experiment if it doesn't exist
        """
        if not MLFLOW_AVAILABLE:
            raise ImportError("MLflow is required for ExperimentTracker. Install with: pip install mlflow")
        
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self.artifact_location = artifact_location
        self.auto_create = auto_create
        
        # MLflow client and experiment
        self.client: Optional[Any] = None
        self.experiment: Optional[Any] = None
        self.experiment_id: Optional[str] = None
        
        # Current run tracking
        self.current_run: Optional[Any] = None
        self.current_run_id: Optional[str] = None
        self.run_stack: List[str] = []  # Support for nested runs
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Initialize MLflow
        self._initialize_mlflow()
        
        logger.info(f"ExperimentTracker initialized for experiment '{experiment_name}'")
    
    def _initialize_mlflow(self) -> None:
        """Initialize MLflow client and experiment."""
        try:
            # Set tracking URI if provided
            if self.tracking_uri and mlflow is not None:
                mlflow.set_tracking_uri(self.tracking_uri)
                logger.info(f"MLflow tracking URI set to: {self.tracking_uri}")
            
            # Initialize client
            if mlflow is not None:
                self.client = MlflowClient()
            
            # Get or create experiment
            if self.client is not None:
                try:
                    self.experiment = self.client.get_experiment_by_name(self.experiment_name)
                    if self.experiment is None and self.auto_create:
                        self.experiment_id = self.client.create_experiment(
                            name=self.experiment_name,
                            artifact_location=self.artifact_location
                        )
                        self.experiment = self.client.get_experiment(self.experiment_id)
                        logger.info(f"Created new experiment: {self.experiment_name}")
                    elif self.experiment is None:
                        raise ValueError(f"Experiment '{self.experiment_name}' does not exist")
                    else:
                        self.experiment_id = self.experiment.experiment_id
                        logger.info(f"Using existing experiment: {self.experiment_name}")
                
                except Exception as e:
                    logger.error(f"Failed to initialize experiment: {e}")
                    raise
                
                # Set default experiment
                if mlflow is not None:
                    mlflow.set_experiment(self.experiment_name)
            
        except Exception as e:
            logger.error(f"Failed to initialize MLflow: {e}")
            raise
    
    def start_run(self, run_name: Optional[str] = None, nested: bool = False,
                  tags: Optional[Dict[str, str]] = None) -> str:
        """
        Start a new MLflow run.
        
        Args:
            run_name: Optional name for the run
            nested: Whether this is a nested run
            tags: Optional tags for the run
            
        Returns:
            Run ID of the started run
        """
        with self._lock:
            try:
                # Handle nested runs
                if nested and self.current_run_id:
                    self.run_stack.append(self.current_run_id)
                elif not nested and self.current_run_id:
                    # End current run if starting a new non-nested run
                    self.end_run()
                
                # Start new run
                run = self.client.create_run(
                    experiment_id=self.experiment_id,
                    tags=tags or {}
                )
                
                self.current_run = run
                self.current_run_id = run.info.run_id
                
                # Set active run in MLflow
                mlflow.start_run(run_id=self.current_run_id, nested=nested)
                
                if run_name:
                    self.log_params({"run_name": run_name})
                
                # Log system information
                self._log_system_info()
                
                logger.info(f"Started MLflow run: {self.current_run_id}")
                return self.current_run_id
                
            except Exception as e:
                logger.error(f"Failed to start MLflow run: {e}")
                raise
    
    def end_run(self, status: str = "FINISHED") -> None:
        """
        End the current MLflow run.
        
        Args:
            status: Run status (FINISHED, FAILED, KILLED)
        """
        with self._lock:
            if not self.current_run_id:
                logger.warning("No active run to end")
                return
            
            try:
                # End run in MLflow
                mlflow.end_run(status=status)
                
                # Update run status
                self.client.set_terminated(
                    run_id=self.current_run_id,
                    status=status,
                    end_time=int(time.time() * 1000)
                )
                
                logger.info(f"Ended MLflow run: {self.current_run_id} with status: {status}")
                
                # Handle nested runs
                if self.run_stack:
                    # Return to parent run
                    parent_run_id = self.run_stack.pop()
                    self.current_run_id = parent_run_id
                    self.current_run = self.client.get_run(parent_run_id)
                else:
                    self.current_run = None
                    self.current_run_id = None
                
            except Exception as e:
                logger.error(f"Failed to end MLflow run: {e}")
                raise
    
    def log_params(self, params: Dict[str, Any]) -> None:
        """
        Log parameters to the current run.
        
        Args:
            params: Dictionary of parameters to log
        """
        if not self.current_run_id:
            logger.warning("No active run to log parameters")
            return
        
        try:
            # Convert all values to strings (MLflow requirement)
            str_params = {k: str(v) for k, v in params.items()}
            mlflow.log_params(str_params)
            logger.debug(f"Logged {len(params)} parameters")
            
        except Exception as e:
            logger.error(f"Failed to log parameters: {e}")
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """
        Log metrics to the current run.
        
        Args:
            metrics: Dictionary of metrics to log
            step: Optional step number for metric
        """
        if not self.current_run_id:
            logger.warning("No active run to log metrics")
            return
        
        try:
            mlflow.log_metrics(metrics, step=step)
            logger.debug(f"Logged {len(metrics)} metrics" + (f" at step {step}" if step else ""))
            
        except Exception as e:
            logger.error(f"Failed to log metrics: {e}")
    
    def log_metric(self, key: str, value: float, step: Optional[int] = None) -> None:
        """
        Log a single metric to the current run.
        
        Args:
            key: Metric name
            value: Metric value
            step: Optional step number
        """
        self.log_metrics({key: value}, step=step)
    
    def log_shared_memory_usage(self, memory_info: Dict[str, int]) -> None:
        """
        Log shared memory usage metrics.
        
        Args:
            memory_info: Memory usage information from DataManager
        """
        if not self.current_run_id:
            logger.warning("No active run to log shared memory usage")
            return
        
        try:
            # Convert to float metrics with memory prefix
            memory_metrics = {
                f"memory.{key}": float(value) for key, value in memory_info.items()
            }
            
            self.log_metrics(memory_metrics)
            logger.debug("Logged shared memory usage metrics")
            
        except Exception as e:
            logger.error(f"Failed to log shared memory usage: {e}")
    
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        """
        Log an artifact (file or directory) to the current run.
        
        Args:
            local_path: Local path to the artifact
            artifact_path: Path where to store the artifact in MLflow
        """
        if not self.current_run_id:
            logger.warning("No active run to log artifact")
            return
        
        try:
            if os.path.isdir(local_path):
                mlflow.log_artifacts(local_path, artifact_path)
            else:
                mlflow.log_artifact(local_path, artifact_path)
            
            logger.debug(f"Logged artifact: {local_path}")
            
        except Exception as e:
            logger.error(f"Failed to log artifact {local_path}: {e}")
    
    def log_model(self, model: Any, artifact_path: str, 
                  registered_model_name: Optional[str] = None,
                  model_framework: str = "sklearn") -> None:
        """
        Log a model to the current run.
        
        Args:
            model: Model object to log
            artifact_path: Path where to store the model
            registered_model_name: Name for model registry
            model_framework: Framework used for the model
        """
        if not self.current_run_id:
            logger.warning("No active run to log model")
            return
        
        try:
            if model_framework.lower() == "sklearn":
                import mlflow.sklearn
                mlflow.sklearn.log_model(
                    model, 
                    artifact_path,
                    registered_model_name=registered_model_name
                )
            elif model_framework.lower() == "pytorch":
                import mlflow.pytorch
                mlflow.pytorch.log_model(
                    model,
                    artifact_path,
                    registered_model_name=registered_model_name
                )
            elif model_framework.lower() == "tensorflow":
                import mlflow.tensorflow
                mlflow.tensorflow.log_model(
                    model,
                    artifact_path,
                    registered_model_name=registered_model_name
                )
            else:
                # Generic model logging
                mlflow.log_model(
                    model,
                    artifact_path,
                    registered_model_name=registered_model_name
                )
            
            logger.info(f"Logged {model_framework} model: {artifact_path}")
            
        except Exception as e:
            logger.error(f"Failed to log model: {e}")
    
    def log_text(self, text: str, artifact_file: str) -> None:
        """
        Log text content as an artifact.
        
        Args:
            text: Text content to log
            artifact_file: Filename for the artifact
        """
        if not self.current_run_id:
            logger.warning("No active run to log text")
            return
        
        try:
            mlflow.log_text(text, artifact_file)
            logger.debug(f"Logged text artifact: {artifact_file}")
            
        except Exception as e:
            logger.error(f"Failed to log text artifact: {e}")
    
    def set_tags(self, tags: Dict[str, str]) -> None:
        """
        Set tags for the current run.
        
        Args:
            tags: Dictionary of tags to set
        """
        if not self.current_run_id:
            logger.warning("No active run to set tags")
            return
        
        try:
            mlflow.set_tags(tags)
            logger.debug(f"Set {len(tags)} tags")
            
        except Exception as e:
            logger.error(f"Failed to set tags: {e}")
    
    def set_tag(self, key: str, value: str) -> None:
        """
        Set a single tag for the current run.
        
        Args:
            key: Tag key
            value: Tag value
        """
        self.set_tags({key: value})
    
    def get_run_info(self, run_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get information about a run.
        
        Args:
            run_id: Run ID (defaults to current run)
            
        Returns:
            Dictionary with run information or None if not found
        """
        target_run_id = run_id or self.current_run_id
        if not target_run_id:
            return None
        
        try:
            run = self.client.get_run(target_run_id)
            return {
                'run_id': run.info.run_id,
                'experiment_id': run.info.experiment_id,
                'status': run.info.status,
                'start_time': run.info.start_time,
                'end_time': run.info.end_time,
                'artifact_uri': run.info.artifact_uri,
                'params': dict(run.data.params),
                'metrics': dict(run.data.metrics),
                'tags': dict(run.data.tags)
            }
            
        except Exception as e:
            logger.error(f"Failed to get run info for {target_run_id}: {e}")
            return None
    
    def list_experiments(self) -> List[Dict[str, Any]]:
        """
        List all experiments.
        
        Returns:
            List of experiment information dictionaries
        """
        try:
            experiments = self.client.search_experiments()
            return [
                {
                    'experiment_id': exp.experiment_id,
                    'name': exp.name,
                    'artifact_location': exp.artifact_location,
                    'lifecycle_stage': exp.lifecycle_stage,
                    'creation_time': exp.creation_time,
                    'last_update_time': exp.last_update_time
                }
                for exp in experiments
            ]
        except Exception as e:
            logger.error(f"Failed to list experiments: {e}")
            return []
    
    def search_runs(self, filter_string: str = "", max_results: int = 1000) -> List[Dict[str, Any]]:
        """
        Search runs in the current experiment.
        
        Args:
            filter_string: MLflow filter string
            max_results: Maximum number of results
            
        Returns:
            List of run information dictionaries
        """
        if not self.experiment_id:
            return []
        
        try:
            runs = self.client.search_runs(
                experiment_ids=[self.experiment_id],
                filter_string=filter_string,
                max_results=max_results
            )
            
            return [
                {
                    'run_id': run.info.run_id,
                    'status': run.info.status,
                    'start_time': run.info.start_time,
                    'end_time': run.info.end_time,
                    'params': dict(run.data.params),
                    'metrics': dict(run.data.metrics),
                    'tags': dict(run.data.tags)
                }
                for run in runs
            ]
            
        except Exception as e:
            logger.error(f"Failed to search runs: {e}")
            return []
    
    def _log_system_info(self) -> None:
        """Log basic system information."""
        try:
            import platform
            import psutil
            
            system_info = {
                "platform.system": platform.system(),
                "platform.release": platform.release(),
                "platform.machine": platform.machine(),
                "platform.processor": platform.processor(),
                "cpu_count": str(psutil.cpu_count()),
                "memory_total_gb": str(round(psutil.virtual_memory().total / (1024**3), 2))
            }
            
            self.log_params(system_info)
            
        except ImportError:
            # psutil not available, log basic info
            import platform
            basic_info = {
                "platform.system": platform.system(),
                "platform.release": platform.release()
            }
            self.log_params(basic_info)
        except Exception as e:
            logger.debug(f"Could not log system info: {e}")
    
    def cleanup(self) -> None:
        """Clean up tracker resources."""
        with self._lock:
            try:
                # End any active runs
                while self.current_run_id:
                    self.end_run("FINISHED")
                
                logger.info("ExperimentTracker cleanup completed")
                
            except Exception as e:
                logger.error(f"Error during cleanup: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        if exc_type is not None:
            # Exception occurred, mark run as failed
            self.end_run("FAILED")
        else:
            self.end_run("FINISHED")
    
    def __del__(self):
        """Destructor with cleanup."""
        try:
            self.cleanup()
        except Exception:
            pass  # Ignore cleanup errors during destruction