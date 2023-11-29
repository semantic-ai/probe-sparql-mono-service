from ..base import Settings


class SparqlConfig(Settings):
    testing_graph: str = "<http://mu.semte.ch/application/probe/testing>"
    user_annotations_graph: str = "<http://mu.semte.ch/application/probe/user-annotations>"
    probe_model_annotations_graph: str = "<http://mu.semte.ch/application/probe/model-annotations>"

    class Config:
        env_prefix = "sparql_"
