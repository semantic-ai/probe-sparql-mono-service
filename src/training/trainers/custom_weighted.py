from transformers import Trainer
import numpy as np

import torch

class WeightedMultilabelTrainer(Trainer):
    __doc__ = Trainer.__doc__

    def __init__(self, **kwargs):
        self.label_distribution = kwargs.pop("label_dist")
        super().__init__(**kwargs)


        self.label_sum = sum(self.label_distribution.values())

        weight = torch.tensor(np.asarray(list(self.label_distribution.values())) / float(self.label_sum)).to(dtype=torch.float).cpu()
        self.criterion = torch.nn.BCEWithLogitsLoss(weight=weight)

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels").view(-1, self.model.config.num_labels).cpu()
        outputs = model(**inputs)
        logits = outputs["logits"].cpu()
        loss = self.criterion(logits, labels)

        return (loss, outputs) if return_outputs else loss
