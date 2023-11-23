from ..config import Config

from typing import Callable
from .logging import LoggingBase

config = Config()
logger = LoggingBase(config.logging, logger_prefix="Wrapper").logger


def wrap(pre: Callable, post: Callable):
    """ Wrapper """

    def decorate(func):
        """ Decorator """

        def call(*args, **kwargs):
            """ Actual wrapping """
            pre(func)
            result = func(*args, **kwargs)
            post(func)
            return result

        call.__doc__ = func.__doc__

        return call

    return decorate


# pre function call logging
def entering(func):
    """ Pre function logging """
    logger.debug("Entered %s", func.__name__)


# post function call debugging
def exiting(func):
    """ Post function logging """
    logger.debug("Exited  %s", func.__name__)
