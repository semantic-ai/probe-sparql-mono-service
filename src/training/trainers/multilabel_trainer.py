from transformers import Trainer

from ..losses import AsymmetricLossOptimized


class MultilabelTrainer(Trainer):
    __doc__ = Trainer.__doc__

    def __init__(self, **kwargs):
        self.label_distribution = kwargs.pop("label_dist")
        super().__init__(**kwargs)

        self.loss_class = AsymmetricLossOptimized()

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        loss = self.loss_class.forward(
            x=logits.view(-1, self.model.config.num_labels),
            y=labels.float().view(-1, self.model.config.num_labels)
        )
        return (loss, outputs) if return_outputs else loss
