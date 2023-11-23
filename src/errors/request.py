class StatuscodeError(Exception):
    """
    Custom error caused when the service returned a non 200 statuscode.
    """

    def __init__(self, status_code: int):
        self.status_code = status_code

    @property
    def message(self):
        return f"Error caused by non 200 response, received {self.status_code}"
