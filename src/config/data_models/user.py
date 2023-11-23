from ..base import Settings
from .constant import CONFIG_PREFIX


class UserConfig(Settings):
    uri_base: str = "https://classifications.ghent.com/ml2grow/user/"

    class Config():
        env_prefix = f"{CONFIG_PREFIX}user_"
