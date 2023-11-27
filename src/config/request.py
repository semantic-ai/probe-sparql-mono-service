from .base import Settings
from ..enums import AuthType


class RequestConfig(Settings):
    endpoint_decision: str = ""
    endpoint_taxonomy: str = ""
    auth_type: AuthType = AuthType.NONE
    username: str = ""
    password: str = ""
    max_retries: int = 3
    header: dict[str, str] = {"accept": "application/sparql-results+json,*/*;q=0.9"}

    class Config():
        env_prefix = "request_"
