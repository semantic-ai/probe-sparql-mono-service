from .base import Settings


class LoggingConfig(Settings):
    level: str = "DEBUG"

    class Config:
        env_prefix = "logging_"
