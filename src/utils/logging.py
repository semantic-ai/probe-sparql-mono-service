import logging
import os

from ..config import LoggingConfig


class LoggingBase:
    """
    Logging class with default setup for logging
    """

    def __init__(self, config: LoggingConfig, logger_prefix: str = "Main"):
        self.logger = logging.getLogger(logger_prefix)

        h1 = logging.StreamHandler()
        h1.setFormatter(logging.Formatter("[%(code)s] %(message)s (%(filename)s:%(lineno)d)"))
        self.logger.addHandler(h1)
        self.logger.setLevel(config.level)
        self.logger.propagate = False

        self.logger = logging.LoggerAdapter(self.logger, extra={"code": logger_prefix})
