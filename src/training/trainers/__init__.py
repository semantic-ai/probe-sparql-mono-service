from .multilabel_trainer import MultilabelTrainer
from .setfit import CustomSetFitTrainer
from .custom_weighted import WeightedMultilabelTrainer
from .regular_trainer import DefaultTrainer

from ...enums import TrainerTypes


def get_trainer(
        trainer_flavour: TrainerTypes,
        model,
        args,
        train_dataset,
        eval_dataset,
        compute_metrics,
        **kwargs
):
    match trainer_flavour:

        case TrainerTypes.CUSTOM | TrainerTypes.CUSTOM.value:
            return MultilabelTrainer(
                model=model,
                args=args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                compute_metrics=compute_metrics,
                **kwargs
            )

        # case TrainerTypes.SETFIT | TrainerTypes.SETFIT.value:
        #     return CustomSetFitTrainer(
        #         model=model,
        #         args=args,
        #         train_dataset=train_dataset,
        #         eval_dataset=eval_dataset,
        #         compute_metrics=compute_metrics,
        #     )

        case TrainerTypes.DEFAULT | TrainerTypes.DEFAULT.value:
            return DefaultTrainer(
                model=model,
                args=args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                compute_metrics=compute_metrics,
                **kwargs
            )

        case TrainerTypes.WEIGHTED | TrainerTypes.WEIGHTED.value:
            return WeightedMultilabelTrainer(
                model=model,
                args=args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                compute_metrics=compute_metrics,
                **kwargs
            )

        case _:
            raise Exception("Provided trainer does not exists")
