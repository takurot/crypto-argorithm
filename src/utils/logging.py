"""Structured logging setup with MLflow integration."""

import logging
import sys
from pathlib import Path
from typing import Optional

import structlog


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[Path] = None,
    json_logs: bool = False,
) -> None:
    """Configure structured logging.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file
        json_logs: Whether to output logs in JSON format
    """
    # Configure standard logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level.upper()),
    )

    # Configure structlog processors
    processors = [
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if json_logs:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Add file handler if specified
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        logging.getLogger().addHandler(file_handler)


def get_logger(name: str) -> structlog.BoundLogger:
    """Get a structured logger instance.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Structured logger instance
    """
    return structlog.get_logger(name)


class MLflowLogger:
    """Logger that integrates with MLflow for experiment tracking."""

    def __init__(self, experiment_name: str = "crypto_price_prediction"):
        """Initialize MLflow logger.
        
        Args:
            experiment_name: Name of MLflow experiment
        """
        self.experiment_name = experiment_name
        self.logger = get_logger(__name__)
        self._mlflow_available = False
        
        try:
            import mlflow
            self._mlflow = mlflow
            self._mlflow_available = True
            
            # Set experiment
            self._mlflow.set_experiment(experiment_name)
            self.logger.info("mlflow_initialized", experiment=experiment_name)
        except ImportError:
            self.logger.warning("mlflow_not_available", 
                              message="MLflow not installed, logging disabled")

    def log_params(self, params: dict) -> None:
        """Log parameters to MLflow.
        
        Args:
            params: Dictionary of parameters
        """
        if not self._mlflow_available:
            return
            
        try:
            self._mlflow.log_params(params)
            self.logger.debug("params_logged", count=len(params))
        except Exception as e:
            self.logger.error("params_log_failed", error=str(e))

    def log_metrics(self, metrics: dict, step: Optional[int] = None) -> None:
        """Log metrics to MLflow.
        
        Args:
            metrics: Dictionary of metrics
            step: Optional step number
        """
        if not self._mlflow_available:
            return
            
        try:
            self._mlflow.log_metrics(metrics, step=step)
            self.logger.debug("metrics_logged", count=len(metrics), step=step)
        except Exception as e:
            self.logger.error("metrics_log_failed", error=str(e))

    def log_artifact(self, file_path: Path, artifact_path: Optional[str] = None) -> None:
        """Log an artifact file to MLflow.
        
        Args:
            file_path: Path to artifact file
            artifact_path: Optional subdirectory in artifact store
        """
        if not self._mlflow_available:
            return
            
        try:
            self._mlflow.log_artifact(str(file_path), artifact_path)
            self.logger.debug("artifact_logged", file=str(file_path))
        except Exception as e:
            self.logger.error("artifact_log_failed", error=str(e), file=str(file_path))

    def start_run(self, run_name: Optional[str] = None, tags: Optional[dict] = None):
        """Start an MLflow run.
        
        Args:
            run_name: Optional name for the run
            tags: Optional tags dictionary
            
        Returns:
            MLflow run context manager
        """
        if not self._mlflow_available:
            # Return a dummy context manager
            from contextlib import nullcontext
            return nullcontext()
            
        return self._mlflow.start_run(run_name=run_name, tags=tags)

    def end_run(self) -> None:
        """End the current MLflow run."""
        if not self._mlflow_available:
            return
            
        try:
            self._mlflow.end_run()
            self.logger.debug("run_ended")
        except Exception as e:
            self.logger.error("run_end_failed", error=str(e))

