"""
================================================================================
REPOSITORY 1: Amharic_Language_Model
MODULE: utils/logger.py
Paper: "ANLI: A Multi-Domain Benchmark Corpus, Automated Quality Filtering,
        and Subword Methodologies for Amharic Natural Language Inference"
Authors: Tesfaye Bekele Kassa, Dr. Tesfa Tegegne Asfaw, Beyene Feleke Mekonnen
================================================================================

Centralized Training & Experiment Logger for Amharic Language Model & FastText
Pre-training. Tracks hyperparameters, model artifacts, training duration, and
corpus statistics with JSON summary exports.
================================================================================
"""

import os
import sys
import time
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List


class TrainingLogger:
    """
    Centralized training configuration and execution logger for unsupervised
    language model pre-training and vector evaluation.
    """

    def __init__(
        self,
        experiment_name: str = "amharic_lm_pretraining",
        log_dir: str = "logs",
        hyperparameters: Optional[Dict[str, Any]] = None
    ):
        self.experiment_name = experiment_name
        self.log_dir = log_dir
        self.hyperparameters = hyperparameters or {}
        self.artifacts: List[Dict[str, Any]] = []
        self.metrics_history: List[Dict[str, Any]] = []
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.duration_seconds: float = 0.0

        os.makedirs(self.log_dir, exist_ok=True)
        self._setup_logging()

    def _setup_logging(self):
        """Configures file and stream handlers for Python logging."""
        self.logger = logging.getLogger(f"{self.experiment_name}_{id(self)}")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False

        if not self.logger.handlers:
            # Console formatter
            ch = logging.StreamHandler(sys.stdout)
            ch.setLevel(logging.INFO)
            formatter = logging.Formatter(
                "[%(asctime)s][%(levelname)s] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)

            # File log handler
            log_file = os.path.join(self.log_dir, f"{self.experiment_name}.log")
            fh = logging.FileHandler(log_file, encoding="utf-8")
            fh.setLevel(logging.INFO)
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)

    def log_hyperparameters(self, params: Dict[str, Any]):
        """Logs and records hyperparameter configuration."""
        self.hyperparameters.update(params)
        self.logger.info("=" * 60)
        self.logger.info(f"EXPERIMENT CONFIGURATION: {self.experiment_name}")
        self.logger.info("=" * 60)
        for k, v in self.hyperparameters.items():
            self.logger.info(f"  - {k}: {v}")
        self.logger.info("=" * 60)

    def start_timer(self):
        """Starts timing the training run."""
        self.start_time = time.time()
        self.logger.info(f"Training started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    def stop_timer(self) -> float:
        """Stops the timer and calculates elapsed seconds."""
        if self.start_time is not None:
            self.end_time = time.time()
            self.duration_seconds = self.end_time - self.start_time
            mins, secs = divmod(self.duration_seconds, 60)
            hrs, mins = divmod(mins, 60)
            self.logger.info(
                f"Training completed. Duration: {int(hrs):02d}h {int(mins):02d}m {secs:05.2f}s "
                f"({self.duration_seconds:.2f} seconds total)"
            )
        return self.duration_seconds

    def log_metric(self, step_or_epoch: int, metrics: Dict[str, Any]):
        """Records evaluation or step metrics."""
        record = {
            "epoch_or_step": step_or_epoch,
            "timestamp": datetime.now().isoformat(),
            **metrics
        }
        self.metrics_history.append(record)
        metric_str = ", ".join([f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in metrics.items()])
        self.logger.info(f"[Epoch {step_or_epoch}] {metric_str}")

    def log_artifact(self, artifact_type: str, path: str, metadata: Optional[Dict[str, Any]] = None):
        """Tracks generated model checkpoints, vector files, or vocabularies."""
        size_bytes = os.path.getsize(path) if os.path.exists(path) else 0
        size_mb = round(size_bytes / (1024 * 1024), 2)
        artifact_entry = {
            "type": artifact_type,
            "path": path,
            "size_mb": size_mb,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        self.artifacts.append(artifact_entry)
        self.logger.info(f"[ARTIFACT SAVED] {artifact_type} -> {path} ({size_mb} MB)")

    def export_summary_json(self, output_filename: Optional[str] = None) -> str:
        """Exports full training run summary to a machine-readable JSON file."""
        if not output_filename:
            output_filename = os.path.join(self.log_dir, f"{self.experiment_name}_summary.json")

        summary = {
            "experiment_name": self.experiment_name,
            "start_time": datetime.fromtimestamp(self.start_time).isoformat() if self.start_time else None,
            "end_time": datetime.fromtimestamp(self.end_time).isoformat() if self.end_time else None,
            "duration_seconds": round(self.duration_seconds, 2),
            "hyperparameters": self.hyperparameters,
            "artifacts": self.artifacts,
            "metrics_history": self.metrics_history
        }

        with open(output_filename, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Exported training run summary JSON to: {output_filename}")
        return output_filename


if __name__ == "__main__":
    logger = TrainingLogger(experiment_name="fasttext_d300_subword_test", log_dir="logs")
    logger.log_hyperparameters({
        "model": "cbow",
        "dim": 300,
        "ws": 5,
        "minn": 3,
        "maxn": 6,
        "bucket": 2000000,
        "epochs": 10,
        "corpus": "training.txt"
    })
    logger.start_timer()
    time.sleep(0.1)  # Simulate brief work
    logger.log_metric(1, {"loss": 0.521, "svd_effective_rank": 118.6, "isotropy": 0.3953})
    logger.stop_timer()
    logger.export_summary_json()
