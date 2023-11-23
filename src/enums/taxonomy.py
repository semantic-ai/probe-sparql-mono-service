import enum


class TaxonomyFindTypes(str, enum.Enum):
    """
    This enum is used to identify what type of taxonomy find method you want to use.
    """

    LABEL: str = "label"
    URI: str = "uri"
