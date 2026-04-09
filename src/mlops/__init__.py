"""MLOps: experiment tracking, model management, and training pipelines."""

__all__ = ["MLOpsTrainingPipeline", "ProductionWindowTrainingPipeline"]


def __getattr__(name: str):
    if name == "MLOpsTrainingPipeline":
        from .training_pipeline import MLOpsTrainingPipeline

        return MLOpsTrainingPipeline
    if name == "ProductionWindowTrainingPipeline":
        from .production_window_training import ProductionWindowTrainingPipeline

        return ProductionWindowTrainingPipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
