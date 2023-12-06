from __future__ import annotations

from abc import ABC
# typing imports
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..config import Config
    from logging import Logger

    from ..dataset import DatasetBuilder

from ..enums import SetfitClassifierHeads

from setfit import SetFitModel
from uuid import uuid4
from .trainers import CustomSetFitTrainer
from .base import Training
# from ..models import SetfitSupervisedModel

from ..enums import TrainerTypes


import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, jaccard_score, hamming_loss, \
    classification_report, multilabel_confusion_matrix

import mlflow, os


class SetfitTraining(Training, ABC):
    """
    Setfit implementation for training

    Compared to the distilbert or regular bert this had inferior performance (model not fully integrated into retraining stack)
    """

    def __init__(
            self,
            config: Config,
            logger: Logger,
            base_model_id: str,
            dataset_builder: DatasetBuilder,
            setfit_head: SetfitClassifierHeads,
            sub_node: str = None,
            nested_mlflow_run: bool = False,
            trainer_flavour: TrainerTypes = TrainerTypes.SETFIT
    ) -> None:
        super().__init__(
            config=config,
            logger=logger,
            base_model_id=base_model_id,
            dataset_builder=dataset_builder
        )
        self.sub_node = sub_node
        self.nested_mlflow_run = nested_mlflow_run

        self.setfit_head = setfit_head or SetfitClassifierHeads.SKLEARN_MULTI_OUTPUT

        mlflow.autolog()

        self.train_folder = f"/tmp/training_{uuid4().hex}"
        self.count_flag = 0

        os.makedirs(self.train_folder, exist_ok=True)

        self._create_dataset()
        self._create_model()

        self.trainer_flavour = trainer_flavour

    def compute_metrics(self, pred, labels):
        self.count_flag += 1

        # print(f"pred: ", pred)
        # print(f"label: ", labels)

        # labels = pred.label_ids.reshape(-1, len(self.target_names))
        probs = torch.sigmoid(torch.tensor(pred)).cpu()
        preds = torch.where(probs < 0.5, 0, 1).int().numpy() # this is somewhat of a duplicate line of code?
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='micro')
        acc = accuracy_score(labels, preds)
        jaccard = jaccard_score(labels, preds, average='micro')
        hamming = hamming_loss(labels, preds)

        clsf_report = classification_report(
            labels,
            preds,
            target_names=self.target_names
        )
        multilabel_conf_matrix = multilabel_confusion_matrix(labels, preds)

        classification_report_file = 'Classification_report.txt'
        with open(os.path.join(self.train_folder, classification_report_file), 'w+') as f:
            f.write(clsf_report)

        confusion_matrix_file = 'Confusion_matrix.txt'
        with open(os.path.join(self.train_folder, confusion_matrix_file), 'w+') as f:
            f.write(str(multilabel_conf_matrix))

        metrics = {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'jaccard_score': jaccard,
            'hamming_loss': hamming
        }

        mlflow.log_metrics({k: v for k, v in metrics.items()}, step=self.count_flag)

        return metrics

    def _create_dataset(self):
        from ..dataset import create_dataset
        from datasets import Dataset

        train_dataset = create_dataset(
            config=self.config,
            logger=self.logger,
            dataset=self.dataset_builder.train_dataset,
            taxonomy=self.dataset_builder.taxonomy,
            sub_node=self.sub_node
        )

        self.target_names = list(train_dataset.binarized_label_dictionary.keys())

        self.train_ds = Dataset.from_list(
            train_dataset
        )

        eval_dataset = create_dataset(
            config=self.config,
            logger=self.logger,
            dataset=self.dataset_builder.test_dataset,
            taxonomy=self.dataset_builder.taxonomy,
            sub_node=self.sub_node
        )

        self.eval_ds = Dataset.from_list(
            eval_dataset
        )

    def _create_model(self):
        self.model = SetFitModel.from_pretrained(
            self.base_model_id,
            **SetfitClassifierHeads.match(self.config, self.setfit_head)
        )

    def train(self):
        with mlflow.start_run(nested=self.nested_mlflow_run):
            trainer = CustomSetFitTrainer(
                model=self.model,
                train_dataset=self.train_ds,
                eval_dataset=self.eval_ds,
                batch_size=self.config.run.training.arguments.per_device_train_batch_size,
                num_iterations=self.config.run.training.arguments.num_train_epochs,
                metric=self.compute_metrics,
                column_mapping={"text": "text", "labels": "label"},
            )

            trainer.train()

            model_checkpoint_dir = f"{self.train_folder}/ModelCheckpoint"

            os.makedirs(model_checkpoint_dir, exist_ok=True)
            trainer.model.save_pretrained(model_checkpoint_dir)
            mlflow.log_artifacts(self.train_folder)

            metrics = trainer.evaluate()
            mlflow.log_metrics(metrics)

    def __call__(self):
        self.train()
