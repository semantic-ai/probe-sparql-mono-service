from ..base import Settings
from .constant import CONFIG_PREFIX


class TaxonomyConfig(Settings):
    """
    The relation here is not defined yet, this is something that District09 has to define
    """
    master_node_uri: str = "http://stad.gent/id/datasets/probe_taxonomies"
    master_child_relation: str = "void:vocabulary"
    ghent_base_uri: str = "http://stad.gent/id/concepts/gent_words"
    ghent_replace_uri: str = "http://stad.gent/id/concepts/gent_words/328"

    inschema_relation: str = "skos:inScheme"
    pref_label_relation: str = "skos:prefLabel"
    broader_relation: str = "skos:broader"

    query_master_nodes: str = f"""\
    PREFIX void: <http://rdfs.org/ns/void#>

    SELECT ?uri
    WHERE {{{{
      <{{master_node_uri}}> {master_child_relation} ?uri
    }}}}
    """

    query_all_children: str = f"""\
        PREFIX skos: <http://www.w3.org/2004/02/skos/core#>

        SELECT DISTINCT ?concept ?label ?broaderConcept ?broaderConceptLabel
        WHERE {{{{
        ?concept a skos:Concept ;
            {pref_label_relation} ?label ;
            {inschema_relation} <{{taxonomy_schema}}> .
        OPTIONAL {{{{
            ?concept {broader_relation} ?broaderConcept .
            ?broaderConcept {pref_label_relation} ?broaderConceptLabel .
        }}}}
        }}}}
        ORDER BY ?broaderConceptLabel ?concept
        """

    class Config():
        env_prefix = f"{CONFIG_PREFIX}taxonomy_"
