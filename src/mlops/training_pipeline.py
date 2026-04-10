"""
MLOps Training Pipeline with MLFlow Integration

Orchestrates the complete ML workflow:
1. Generate synthetic data (reproducible)
2. Track experiments with MLFlow
3. Train models
4. Evaluate on multiple metrics
5. Log artifacts and models
6. Register best models
7. Cross-validation strategy

Run with: python -m src.mlops.training_pipeline
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib
import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
import xgboost as xgb
import yaml
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)
from sklearn.model_selection import cross_val_score

from src.core.smoothness_ml_engine import generate_synthetic_labels
from src.mlops.mlflow_common import log_serving_artifacts, log_xgboost_model
from src.mlops.mlflow_settings import ensure_mlflow_experiment, resolve_tracking_uri
from src.utils.data_generation_strategy import (
    SyntheticDataPipeline,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class MLOpsTrainingPipeline:
    """Complete MLOps training pipeline with MLFlow."""

    def __init__(self, config_path: str = "mlops_config.yaml"):
        """Initialize with MLOps configuration."""
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.mlflow_config = self.config["mlflow"]
        self.model_config = self.config["model"]
        self.exp_config = self.config["experiment"]
        self.cv_config = self.config["cross_validation"]
        self.eval_config = self.config["evaluation"]
        self.repro_config = self.config["reproducibility"]
        self.env_config = self.config["environment"]["dev"]
        self.output_config = self.config["output"]

        # Setup directories
        self._setup_directories()

        # Setup MLFlow
        self._setup_mlflow()

        # Set reproducibility
        self._set_reproducibility()

        logger.info("✅ MLOps pipeline initialized")

    def _setup_directories(self):
        """Create necessary output directories."""
        for dir_key in ["model_dir", "reports_dir", "data_dir"]:
            dir_path = Path(self.output_config[dir_key])
            dir_path.mkdir(parents=True, exist_ok=True)

    def _setup_mlflow(self):
        """Configure MLflow (env overrides YAML for CI / shared servers)."""
        uri = resolve_tracking_uri(self.mlflow_config["tracking_uri"])
        experiment = os.environ.get(
            "MLFLOW_EXPERIMENT_SYNTHETIC_18", self.mlflow_config["experiment_name"]
        )
        mlflow.set_tracking_uri(uri)
        ensure_mlflow_experiment(experiment)
        logger.info("📊 MLflow tracking URI: %s", uri)
        logger.info("📊 MLflow experiment: %s", experiment)

    def _set_reproducibility(self):
        """Set random seeds for reproducibility."""
        seed = self.repro_config.get("numpy_seed", 42)
        np.random.seed(seed)
        logger.info(f"🔒 Reproducibility seed set: {seed}")

    def generate_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Generate synthetic training data.

        Returns:
            (train_df, val_df, test_df)
        """
        logger.info("🔄 Generating synthetic data...")

        pipeline = SyntheticDataPipeline(config_path="mlops_config.yaml")
        full_df = pipeline.generate_dataset()

        train_df, val_df, test_df = pipeline.split_data(full_df)

        # Save datasets for reproducibility
        data_dir = Path(self.output_config["data_dir"])
        train_df.to_csv(data_dir / "train.csv", index=False)
        val_df.to_csv(data_dir / "val.csv", index=False)
        test_df.to_csv(data_dir / "test.csv", index=False)

        logger.info(f"💾 Saved datasets to {data_dir}")

        return train_df, val_df, test_df

    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, list]:
        """
        Prepare features and labels.

        Returns:
            (X, y, feature_names)
        """
        # Feature columns (all 18 features)
        feature_columns = [
            # Longitudinal
            "avg_accel_g",
            "avg_accel_std",
            "max_decel_g",
            "total_harsh_brakes",
            "total_harsh_accels",
            # Lateral
            "avg_lateral_g",
            "max_lateral_g",
            "total_harsh_corners",
            # Speed
            "avg_speed_kmh",
            "avg_speed_std",
            "max_speed_kmh",
            # Jerk
            "avg_jerk",
            "avg_jerk_std",
            "max_jerk",
            # Engine
            "avg_rpm",
            "max_rpm",
            "total_idle_seconds",
            "total_over_revs",
        ]

        # Use available features
        available_features = [f for f in feature_columns if f in df.columns]

        X = df[available_features].values
        y = generate_synthetic_labels(df[available_features])

        logger.info(f"📊 Features: {len(available_features)}")
        logger.info(f"   {', '.join(available_features[:5])}...")
        logger.info(f"🎯 Labels: generated {len(y)} synthetic labels")

        return X, y, available_features

    def train_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> xgb.XGBRegressor:
        """
        Train XGBoost model.

        Returns:
            Trained model
        """
        logger.info("\n🚀 Training XGBoost model...")

        params = self.model_config["xgboost"]

        model = xgb.XGBRegressor(
            n_estimators=params["n_estimators"],
            learning_rate=params["learning_rate"],
            max_depth=params["max_depth"],
            subsample=params["subsample"],
            colsample_bytree=params["colsample_bytree"],
            random_state=params["random_state"],
            objective=params["objective"],
        )

        # Train with early stopping if validation set provided
        if X_val is not None and y_val is not None:
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                verbose=50,
            )
        else:
            model.fit(X_train, y_train)

        logger.info("✅ Training complete")
        return model

    def evaluate_model(
        self,
        model: xgb.XGBRegressor,
        X_test: np.ndarray,
        y_test: np.ndarray,
        split_name: str = "test",
    ) -> Dict:
        """
        Evaluate model and compute metrics.

        Returns:
            Dict of metrics
        """
        predictions = model.predict(X_test)

        metrics = {
            "mae": mean_absolute_error(y_test, predictions),
            "mse": mean_squared_error(y_test, predictions),
            "rmse": np.sqrt(mean_squared_error(y_test, predictions)),
            "r2_score": r2_score(y_test, predictions),
            "median_ae": median_absolute_error(y_test, predictions),
        }

        logger.info(f"\n📈 {split_name.capitalize()} Metrics:")
        for metric, value in metrics.items():
            logger.info(f"   {metric}: {value:.4f}")

        return metrics

    def cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Dict:
        """
        Perform k-fold cross-validation.

        Returns:
            CV metrics
        """
        logger.info(f"\n🔄 Cross-validation ({self.cv_config['n_splits']} folds)...")

        model_template = xgb.XGBRegressor(**self.model_config["xgboost"])

        # CV on R² score
        cv_scores = cross_val_score(
            model_template,
            X,
            y,
            cv=self.cv_config["n_splits"],
            scoring="r2",
        )

        cv_metrics = {
            "cv_r2_mean": cv_scores.mean(),
            "cv_r2_std": cv_scores.std(),
            "cv_r2_scores": cv_scores.tolist(),
        }

        logger.info(
            f"   Mean R²: {cv_metrics['cv_r2_mean']:.4f} " f"(±{cv_metrics['cv_r2_std']:.4f})"
        )

        return cv_metrics

    def log_to_mlflow(
        self,
        model: xgb.XGBRegressor,
        metrics: Dict,
        cv_metrics: Dict,
        feature_names: list,
        train_df: pd.DataFrame,
    ):
        """Log model, metrics, and artifacts to MLFlow."""

        logger.info("\n📝 Logging to MLFlow...")

        # Log parameters
        params = self.model_config["xgboost"]
        for param, value in params.items():
            mlflow.log_param(param, value)

        mlflow.log_param("num_features", len(feature_names))
        mlflow.log_param("train_size", len(train_df))
        mlflow.log_param("data_seed", self.config["data"]["random_seed"])

        # Log metrics
        for metric_name, value in metrics.items():
            mlflow.log_metric(metric_name, value)

        for cv_metric_name, value in cv_metrics.items():
            if cv_metric_name != "cv_r2_scores":
                mlflow.log_metric(cv_metric_name, value)

        run_id = mlflow.active_run().info.run_id
        X_frame = train_df[feature_names].astype(float)
        log_xgboost_model(model, feature_names, X_frame)
        root = Path(__file__).resolve().parents[2]
        cfg_snap = root / "mlops_config.yaml"
        log_serving_artifacts(
            run_id,
            X_frame,
            feature_names,
            background_rows=min(96, len(X_frame)),
            random_seed=int(self.config["data"]["random_seed"]),
            contract_extra={
                "pipeline": "synthetic_18f",
                "model_name": "smoothness_synthetic_18f",
            },
            yaml_snapshots=[cfg_snap] if cfg_snap.is_file() else None,
        )

        # Log feature importance
        importance = model.get_booster().get_score(importance_type="weight")
        importance_file = Path(self.output_config["reports_dir"]) / "feature_importance.json"
        with open(importance_file, "w") as f:
            json.dump(importance, f, indent=2)
        mlflow.log_artifact(str(importance_file))

        # Log training metadata
        metadata = {
            "model_type": "xgboost",
            "data_generator": "synthetic",
            "features_used": feature_names,
            "timestamp": datetime.now().isoformat(),
            "cv_scores": cv_metrics.get("cv_r2_scores", []),
        }
        metadata_file = Path(self.output_config["reports_dir"]) / "metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)
        mlflow.log_artifact(str(metadata_file))

        logger.info("✅ Logged to MLFlow")

    def run_pipeline(self):
        """Execute complete MLOps pipeline."""

        logger.info("=" * 70)
        logger.info("🚀 STARTING MLOps TRAINING PIPELINE")
        logger.info("=" * 70)

        with mlflow.start_run() as run:
            logger.info(f"📊 MLFlow Run ID: {run.info.run_id}")

            # Step 1: Generate data
            train_df, val_df, test_df = self.generate_data()

            # Step 2: Prepare features
            X_train, y_train, feature_names = self.prepare_features(train_df)
            X_val, y_val, _ = self.prepare_features(val_df)
            X_test, y_test, _ = self.prepare_features(test_df)

            # Step 3: Train model
            model = self.train_model(X_train, y_train, X_val, y_val)

            # Step 4: Evaluate on train
            train_metrics = self.evaluate_model(model, X_train, y_train, "train")

            # Step 5: Evaluate on validation
            val_metrics = self.evaluate_model(model, X_val, y_val, "validation")

            # Step 6: Evaluate on test
            test_metrics = self.evaluate_model(model, X_test, y_test, "test")

            # Step 7: Cross-validation
            cv_metrics = self.cross_validate(X_train, y_train)

            # Step 8: Check quality gate
            quality_gate = self.model_config["quality_gate"]
            if test_metrics["r2_score"] < quality_gate["min_r2_score"]:
                logger.warning(
                    f"⚠️  Quality gate FAILED: R²={test_metrics['r2_score']:.4f} "
                    f"< {quality_gate['min_r2_score']}"
                )
                mlflow.log_param("quality_gate_status", "FAILED")
            else:
                logger.info(f"✅ Quality gate PASSED: R²={test_metrics['r2_score']:.4f}")
                mlflow.log_param("quality_gate_status", "PASSED")

                # Save model
                model_path = Path(self.output_config["model_dir"]) / "smoothness_model.joblib"
                joblib.dump(model, model_path)
                logger.info(f"💾 Model saved to {model_path}")

            # Step 9: Log to MLFlow
            all_metrics = {
                **{f"train_{k}": v for k, v in train_metrics.items()},
                **{f"val_{k}": v for k, v in val_metrics.items()},
                **{f"test_{k}": v for k, v in test_metrics.items()},
            }

            self.log_to_mlflow(model, all_metrics, cv_metrics, feature_names, train_df)

            logger.info("\n" + "=" * 70)
            logger.info("✅ MLOps PIPELINE COMPLETE")
            logger.info("=" * 70)

            return {
                "model": model,
                "metrics": {
                    "train": train_metrics,
                    "val": val_metrics,
                    "test": test_metrics,
                    "cv": cv_metrics,
                },
                "run_id": run.info.run_id,
            }


def main():
    """Entry point for training pipeline."""
    pipeline = MLOpsTrainingPipeline("mlops_config.yaml")
    result = pipeline.run_pipeline()

    logger.info("\n📊 FINAL RESULTS:")
    logger.info(f"   Test R²: {result['metrics']['test']['r2_score']:.4f}")
    logger.info(f"   Test RMSE: {result['metrics']['test']['rmse']:.4f}")
    logger.info(f"   CV R² Mean: {result['metrics']['cv']['cv_r2_mean']:.4f}")
    logger.info(f"   MLFlow Run: {result['run_id']}")


if __name__ == "__main__":
    main()
