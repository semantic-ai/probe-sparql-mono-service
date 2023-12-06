import enum


class TrainerTypes(str, enum.Enum):
    """
    This enum is used to identify what type of taxonomy find method you want to use.
    """

    DEFAULT: str = "default"
    WEIGHTED: str = "weighted"
    CUSTOM: str = "custom"
    SETFIT: str = "setfit"
