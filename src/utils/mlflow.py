import uuid


def get_unique_run_name(run_name: str) -> str:
    """
    Mflow unique helper
    :param run_name:
    :return:
    """
    return f"{run_name}__{uuid.uuid4().hex}"
