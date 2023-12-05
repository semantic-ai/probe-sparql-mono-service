from ..base import Settings
from .constant import CONFIG_PREFIX


class ModelConfig(Settings):
    uri_base: str = "https://lblod.data.gift/concepts/ml2grow/model/"
    date_relation: str = "ext:creationDate"
    name_relation: str = "ext:modelName"
    category_relation: str = "ext:modelCategory"
    mlflow_url_relation: str = "ext:mlflowLink"
    mlflow_registered_model_relation: str = "ext:registeredMlflowModel"

    query: str = f"""\
    PREFIX ext: <http://mu.semte.ch/vocabularies/ext/>
    SELECT * {{{{
        <{{uri}}> {category_relation} ?category; 
            {mlflow_registered_model_relation} ?mlflow_model; 
            {name_relation} ?model_name;
            {mlflow_url_relation} ?mlflow_link;
            {date_relation} ?create_data . 
    }}}}
    """

    sub_query: str = f"""\
    {{uri}} {date_relation} {{date}} .
    {{uri}} {name_relation} "{{name}}".
    {{uri}} {category_relation} "{{category}}" .
    {{uri}} {mlflow_url_relation} \"\"\"{{mlflow_reference}}\"\"\" .
    {{uri}} {mlflow_registered_model_relation} "{{registered_model}}" .
    """

    class Config():
        env_prefix = f"{CONFIG_PREFIX}annotation_model_"
