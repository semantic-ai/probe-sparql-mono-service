from .constant import CONFIG_PREFIX
from ...base import Settings


class TrainingConfig(Settings):
    # training parameters
    num_train_epochs: int = 10
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 20
    warmup_steps: int = 200
    weight_decay: float = 0.01
    load_best_model_at_end: bool = True
    evaluation_strategy: str = "epoch"
    save_strategy: str = "epoch"
    dataloader_pin_memory: bool = True

    class Config():
        env_prefix = f"{CONFIG_PREFIX}args_"
