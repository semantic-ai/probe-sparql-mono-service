from __future__ import annotations
from .constant import CONFIG_PREFIX
from ...base import Settings


class EvaluationConfig(Settings):
    type: str = "multi"

    with_threshold: bool = True
    overview_plot: bool = True
    confusion_matrix: bool = True
    precision_recall_plot: bool = True
    classification_report: bool = True

    # general flag to logg artifacts to mlflow
    log_artifacts: bool = True

    class Config():
        env_prefix = f"{CONFIG_PREFIX}evaluation_"
