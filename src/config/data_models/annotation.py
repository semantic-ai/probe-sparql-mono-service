from __future__ import annotations
from ..base import Settings
from .constant import CONFIG_PREFIX


class AnnotationConfig(Settings):
    uri_base: str = "https://lblod.data.gift/concepts/ml2grow/annotations/"

    insert_graph: str = "<http://mu.semte.ch/application>"

    probe_model_relation: str = "ext:withModel"
    taxonomy_relation: str = "ext:withTaxonomy"
    label_relation: str = "ext:hasLabel"
    date_relation: str = "ext:creationDate"
    user_relation: str = "ext:withUser"

    query_annotation_info: str = f"""\
    PREFIX  ext:  <http://mu.semte.ch/vocabularies/ext/>

    SELECT ?date ?taxonomy_uri ?model_uri ?user_uri ?label_uris 
    WHERE {{{{
      VALUES ?annotation {{{{ <{{annotation_uri}}> }}}}
        ?annotation {date_relation} ?date ;
        {taxonomy_relation} ?taxonomy_uri .
        OPTIONAL {{{{ ?annotation {probe_model_relation} ?model_uri }}}}
        OPTIONAL {{{{ ?annotation {user_relation} ?user_uri }}}}
        }}}}
    """

    query_linked_labels: str = f"""\
        PREFIX  ext:  <http://mu.semte.ch/vocabularies/ext/>

        SELECT * WHERE
          {{{{
            <{{annotation_uri}}>  {label_relation} ?label_uri
          }}}}
        """

    sub_query: str = f"""\
    {{uri}} {taxonomy_relation} {{taxonomy}} .
    {{uri}} {date_relation} {{timestamp}} .
    {{uri}} {label_relation} {{label_uris}} .
    {{user_reference}}
    {{model_reference}}
    {{labels}}
    """

    class Config():
        env_prefix = f"{CONFIG_PREFIX}annotation_"
