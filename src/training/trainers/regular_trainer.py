from transformers import Trainer

class DefaultTrainer(Trainer):
    __doc__ = Trainer.__doc__

    def __init__(self, **kwargs):
        self.label_distribution = kwargs.pop("label_dist")
        super().__init__(**kwargs)

