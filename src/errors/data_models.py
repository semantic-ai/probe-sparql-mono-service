class CustomValueError(ValueError):
    """
    Custom value error that is more verbose than the default implementation

    Default docs:
    ---

    """

    __doc__ += str(ValueError.__doc__)

    def __init__(self, property: str, expected_type: object, received_type: object):
        super().__init__()

        self.exp_type = expected_type
        self.rec_type = received_type
        self.property = property

    @property
    def message(self):
        return f"Property {self.property} expected value to be of type <{self.exp_type}> but received {self.rec_type}"

    def __repr__(self):
        return self.message
