from __future__ import annotations

# typing imports
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..config.runs.benchmark import BenchmarkConfig
    from logging import Logger

import numpy as np
import pandas as pd
import os

import mlflow
import math

from sklearn.metrics import f1_score, recall_score, precision_score
from sklearn.metrics import multilabel_confusion_matrix, ConfusionMatrixDisplay, PrecisionRecallDisplay, \
    classification_report

import matplotlib.pyplot as plt


class Metrics:
    """
    This class is used to compute metrics for a benchmarking run.
    It is capable of generating multiple different artifacts that can be used for further model performance analysis.

    """

    def __init__(
            self,
            config: BenchmarkConfig,
            logger: Logger,
            model_id: str,
            base_folder: str,
            classes: list[str],
            average: str = "weighted"
    ):
        self.config = config
        self.logger = logger

        self.model_id = model_id
        self.classes = classes
        self.base_folder = base_folder
        os.makedirs(self.base_folder, exist_ok=True)
        self.average = average
        self.zero_devision = self.config.metrics.zero_division_default


    @staticmethod
    def hamming_score(
            y_true: np.array,
            y_pred: np.array
    ) -> np.array:
        """
        This function contains the code to compute the hamming_Score.

        Hamming score formula:
            >>>  Hamming score = (Σ (y_true_i == y_pred_i)) / (Σ 1)

        The Hamming score is a useful metric for evaluating the performance of multi-label classification models,
        which are models that predict multiple labels for each instance. In multi-label classification,
        the Hamming score is more sensitive to errors than accuracy, as it considers both false positives and false negatives.

        :param y_true: matrix what the labels should be
        :param y_pred: matrix with the predicted labels
        :return: the hamming scores
        """

        acc_list = []
        for i in range(y_true.shape[0]):
            set_true = set(np.where(y_true[i])[0])
            set_pred = set(np.where(y_pred[i])[0])
            if len(set_true) == 0 and len(set_pred) == 0:
                tmp_a = 1
            else:
                tmp_a = len(set_true.intersection(set_pred)) / float(len(set_true.union(set_pred)))
            acc_list.append(tmp_a)
        return np.mean(acc_list)

    def f1_score(
            self,
            y_true: np.array,
            y_pred: np.array
    ) -> np.array:
        """
        This function contains the code to compute the hamming_Score.

        F1 score formula:
            >>> F1 = 2 * (precision * recall) / (precision + recall)

        The F1 score is a statistical measure of a test's accuracy, particularly in the context of binary classification.
        It is the harmonic mean of precision and recall, which are two important metrics for evaluating the performance of binary classifiers.

        :param y_true: matrix what the labels should be
        :param y_pred: matrix with the predicted labels
        :return: the f1 scores
        """

        return f1_score(
            y_true,
            y_pred,
            average=self.average,
            zero_division=self.zero_devision
        )

    def recall_score(
            self,
            y_true: np.array,
            y_pred: np.array
    ) -> np.array:
        """
        This function computes the recall for the given input

        The recall formula:
            >>> Recall = TP / (TP + FN)
            >>> where:
            >>>     TP is the number of true positives
            >>>     FN is the number of false negatives

        In machine learning, recall, also known as sensitivity, true positive rate (TPR), or completeness, is the proportion of actual positives that are correctly identified as such by the model.
        It is calculated as the number of true positives divided by the total number of actual positives. Recall is a binary classification metric, meaning it is only relevant when a model is predicting one of two classes.

        :param y_true: matrix what the labels should be
        :param y_pred: matrix with the predicted labels
        :return: the recall scores
        """

        return recall_score(
            y_true,
            y_pred,
            average=self.average,
            zero_division=self.zero_devision
        )

    def precision_score(
            self,
            y_true: np.array,
            y_pred: np.array
    ) -> np.array:
        """
        This function enables the computation of precision score

        The precision formula:
            >>> Precision = TP / (TP + FP)
            >>> where:
            >>>    TP is the number of true positives
            >>>    FP is the number of false positives



        In machine learning, precision, also known as positive predictive value (PPV), is the proportion of predicted positives that are actually positive.
        It is calculated as the number of true positives divided by the total number of predicted positives.
        Precision is a binary classification metric, meaning it is only relevant when a model is predicting one of two classes

        :param y_true: matrix what the labels should be
        :param y_pred: matrix with the predicted labels
        :return: the precision scores
        """

        return precision_score(
            y_true,
            y_pred,
            average=self.average,
            zero_division=self.zero_devision
        )

    def compute(self, y_true: np.array, logits: np.array, suffix: str = None, save: bool = True) -> pd.DataFrame:
        """
        This function brings all previous calculations together.
        Paired with the config, you can enable and disable certain artifacts and metrics

        Metrics that can be used:
            1. Hamming score
            2. f1 score
            3. precision
            4. recall

        Artifacts that can be generated:
            1. Confusion matrix
            2. classification report
            3. precision-recall plot
            4. overview plot

        :param y_true: the values that are expected to be predicted
        :param logits: the logits for the predicted values
        :param suffix: a suffix that can be used for custom naming of the artifacts
        :param save: flag to save (currently not used -> config handles this)
        :return: pandas dataframe containing all the metric values
        """
        y_true = np.nan_to_num(y_true)
        logits = np.nan_to_num(logits)

        experiment = {}
        for thresh in range(0, 100, 1):
            calc_thresh = thresh / 100.0
            y_pred = np.where(logits > calc_thresh, 1, 0)

            metrics = dict()

            if self.config.metrics.hamming:
                metrics["hamming"] = self.hamming_score(y_true, y_pred)

            if self.config.metrics.precision:
                metrics["precision_score"] = self.precision_score(y_true, y_pred)

            if self.config.metrics.recall:
                metrics["recall_score"] = self.recall_score(y_true, y_pred)

            if self.config.metrics.f1:
                metrics["f1_score"] = self.f1_score(y_true, y_pred)

            if self.config.evaluation.with_threshold:
                metrics["threshold"] = calc_thresh

            experiment[thresh] = metrics
            mlflow.log_metrics(metrics, step=thresh)

            if self.config.evaluation.confusion_matrix:

                confusion_matrix_folder = os.path.join(
                    self.base_folder,
                    "confusion_matrix"
                )
                os.makedirs(confusion_matrix_folder, exist_ok=True)

                cms = multilabel_confusion_matrix(
                    y_true,
                    y_pred
                )

                row = math.ceil(math.sqrt(len(self.classes)))

                confusion_matrix_save_path = os.path.join(
                    confusion_matrix_folder,
                    f"{thresh:02}_confusion_matrix.png"
                )

                f, axes = plt.subplots(row, row, figsize=(row * 5, row * 5))
                axes = axes.ravel()
                disp = None

                for i, (cm, label) in enumerate(zip(cms, self.classes)):
                    disp = ConfusionMatrixDisplay(cm)
                    disp.plot(ax=axes[i], values_format='.4g')
                    disp.ax_.set_title(f'class {label}')

                    disp.im_.colorbar.remove()

                plt.subplots_adjust(wspace=0.10, hspace=0.1)
                f.colorbar(disp.im_, ax=axes)
                f.savefig(confusion_matrix_save_path)

                plt.close(f)

            if self.config.evaluation.classification_report:
                classification_report_folder = os.path.join(
                    self.base_folder,
                    "classification_report"
                )
                os.makedirs(classification_report_folder, exist_ok=True)

                classification_report_file = os.path.join(
                    classification_report_folder,
                    f"{thresh:02}_classification_report.txt"
                )

                classification_result = classification_report(
                    y_true,
                    y_pred,
                    output_dict=False,
                    target_names=self.classes,
                    zero_division=self.zero_devision
                )

                with open(classification_report_file, 'w+') as f:
                    f.write(classification_result)

        stat_df = pd.DataFrame(experiment).T

        if self.config.evaluation.precision_recall_plot:
            precision_recall_save_path = os.path.join(
                self.base_folder,
                "precision_recall_plot.png"
            )
            disp = PrecisionRecallDisplay(
                recall=stat_df["recall_score"],
                precision=stat_df["precision_score"]
            )
            fig = disp.plot().figure_
            fig.savefig(precision_recall_save_path)
            plt.close(fig)

        if self.config.evaluation.overview_plot:
            figure = stat_df.plot(
                figsize=(18, 8),
                title=f"Score related to threshold - {self.model_id}",
                xlabel="threshold steps 0->100",
                ylabel="score"
            ).get_figure()

            save_location = os.path.join(
                self.base_folder,
                f"{self.model_id.replace('/', '_')}_scores{('_' + suffix) if suffix is not None else ''}.png"
            )
            figure.savefig(save_location)
            plt.close(figure)

        mlflow.log_artifacts(self.base_folder)

        return stat_df.T
