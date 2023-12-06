from __future__ import annotations

from abc import ABC
# typing imports
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..config import Config
    from logging import Logger

    from ..dataset import DatasetBuilder

from .base import Training
from .trainers import MultilabelTrainer

from ..dataset import create_dataset

from datasets import Dataset
from transformers import TrainingArguments, AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, jaccard_score, hamming_loss, \
    classification_report, multilabel_confusion_matrix, roc_auc_score

from .trainers import get_trainer
from ..enums import TrainerTypes

import mlflow
import os
import torch
from uuid import uuid4
import numpy as np


class BertTraining(Training, ABC):
    """
    Training implementation for the bert class
    """

    def __init__(
            self,
            config: Config,
            logger: Logger,
            base_model_id: str,
            dataset_builder: DatasetBuilder,
            sub_node: str = None,
            nested_mlflow_run: bool = False,
            trainer_flavour: TrainerTypes = TrainerTypes.CUSTOM
    ):
        super().__init__(
            config=config,
            logger=logger,
            base_model_id=base_model_id,
            dataset_builder=dataset_builder
        )

        self.sub_node = sub_node
        self.nested_mlflow_run=nested_mlflow_run

        self.config.run.dataset.tokenize = True
        mlflow.transformers.autolog()

        self.train_folder = f"/tmp/training_{uuid4().hex}"
        os.makedirs(self.train_folder, exist_ok=True)

        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_id, do_lower_case=True)
        self._create_dataset()
        self._create_model()

        self.trainer_flavour = trainer_flavour

        self.count_flag = 0

    def compute_metrics(self, pred):
        self.logger.info(f"pred: {pred}")
        labels = pred.label_ids.reshape(-1, len(self.target_names))
        probs = torch.sigmoid(torch.tensor(pred.predictions)).cpu()
        preds = torch.where(probs < 0.5, 0, 1).int().numpy()

        self.logger.debug("___probabilities___")
        self.logger.debug(probs.shape)
        self.logger.debug(probs)

        self.logger.debug("___labels___")
        self.logger.debug(labels.shape)
        self.logger.debug(labels)

        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='micro')
        acc = accuracy_score(labels, preds)
        jaccard = jaccard_score(labels, preds, average='micro')
        hamming = hamming_loss(labels, preds)
        roc_auc = roc_auc_score(labels, preds, average='micro')

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
            'hamming_loss': hamming,
            'roc_auc': roc_auc
        }

        mlflow.log_metrics(metrics, step=self.count_flag)
        self.count_flag += 1

        return metrics

    def _create_model(self):
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.base_model_id,
            problem_type="multi_label_classification",
            num_labels=len(self.target_names),
            id2label=self.id2label,
            label2id=self.label2id
        )

    def _create_dataset(self):
        self.logger.info(f"Creating training dataset")
        self.train_dataset = create_dataset(
            config=self.config,
            logger=self.logger,
            dataset=self.dataset_builder.train_dataset,
            taxonomy=self.dataset_builder.taxonomy,
            tokenizer=self.tokenizer,
            sub_node=self.sub_node
        )

        self.target_names = list(self.train_dataset.binarized_label_dictionary.keys())
        self.logger.info(f"target names: len({len(self.target_names)}) {self.target_names}")

        self.id2label = {i: label for i, label in enumerate(self.target_names)}
        self.label2id = {v: k for k, v in self.id2label.items()}

        self.train_ds = Dataset.from_list(
            self.train_dataset
        )

        self.train_dist = self.train_dataset.label_distribution
        mlflow.log_dict(self.train_dist, "train_distribution.json")
        self.logger.info(f"label distribution train: {self.train_dist}")

        eval_dataset = create_dataset(
            config=self.config,
            logger=self.logger,
            dataset=self.dataset_builder.test_dataset,
            taxonomy=self.dataset_builder.taxonomy,
            tokenizer=self.tokenizer,
            sub_node=self.sub_node
        )

        self.logger.info(f"Creating eval dataset")
        self.eval_ds = Dataset.from_list(
            eval_dataset
        )

        if len(eval_dataset) > 0:
            self.logger.debug(f"Example dataset record: {eval_dataset[0]}")
            self.logger.debug("information")
            self.logger.debug(np.asarray(self.eval_ds[0:80]['labels']).shape)
            self.logger.debug(np.sum(np.array(self.eval_ds[0:80]['labels']), axis=0))

        mlflow.log_dict(eval_dataset.label_distribution, "eval_distribution.json")
        self.logger.info(f"label distribution test: {eval_dataset.label_distribution}")

    def train(self):
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=self.config.run.training.arguments.num_train_epochs,
            per_device_train_batch_size=self.config.run.training.arguments.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.run.training.arguments.per_device_eval_batch_size,
            warmup_steps=self.config.run.training.arguments.warmup_steps,
            weight_decay=self.config.run.training.arguments.weight_decay,
            logging_dir=os.path.join(self.train_folder, "logs"),
            load_best_model_at_end=self.config.run.training.arguments.load_best_model_at_end,
            evaluation_strategy=self.config.run.training.arguments.evaluation_strategy,
            save_strategy=self.config.run.training.arguments.save_strategy,
            dataloader_pin_memory=self.config.run.training.arguments.dataloader_pin_memory,
        )

        trainer = get_trainer(
            trainer_flavour=self.trainer_flavour,
            model=self.model,
            args=training_args,
            train_dataset=self.train_ds,
            eval_dataset=self.eval_ds,
            compute_metrics=self.compute_metrics,
            label_dist=self.train_dist
        )

        trainer.train()
        best_model_results = trainer.evaluate()

        mlflow.log_metrics(best_model_results, step=self.count_flag)
        mlflow.log_artifacts(self.train_folder)

        components = dict(
            model=trainer.model,
            tokenizer=self.tokenizer
        )

        model_flavour = "bert"
        taxonomy_name = self.dataset_builder.taxonomy.uri.split('/')[-1]

        if hasattr(self.train_dataset, "sub_node_taxo"):
            sub_taxonomy_name = "_".join(self.train_dataset.sub_node_taxo.label.split())
        else:
            sub_taxonomy_name = "parent_node"

        model_name = "__".join([model_flavour, taxonomy_name, sub_taxonomy_name]).lower()
        mlflow.transformers.log_model(
            transformers_model=components,
            registered_model_name=model_name,
            artifact_path="model"
        )
