from ..base import Settings
from .constant import CONFIG_PREFIX


class LabelConfig(Settings):
    uri_base: str = "https://lblod.data.gift/concepts/ml2grow/label/"
    taxonomy_relation: str = "ext:isTaxonomy"
    score_relation: str = "ext:hasScore"

    query: str = f"""\
    PREFIX  ext:  <http://mu.semte.ch/vocabularies/ext/>

    SELECT * WHERE
      {{{{
      <{{uri}}> {score_relation} ?score ; {taxonomy_relation} ?taxonomy_node_uri .
      }}}}
    """

    sub_query: str = f"""
    {{uri}} {taxonomy_relation} {{taxonomy_node_uri}} .
    {{uri}} {score_relation} {{score}}.
        """

    class Config():
        env_prefix = f"{CONFIG_PREFIX}label_"
