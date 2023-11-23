from ..base import Settings
from .constant import CONFIG_PREFIX


class DecisionConfig(Settings):
    annotation_relation: str = "ext:hasAnnotation"
    besluit_relation: str = "besluit:Besluit"

    description_relation: str = "eli:description"
    short_title_relation: str = "eli:title_short"
    motivation_relation: str = "besluit:motivering"
    publication_date_relation: str = "eli:date_publication"
    langauge_relation: str = "eli:language"

    generated_relation: str = "prov:wasGeneratedBy"
    subject_relation: str = "dct:subject"

    haspart_relation: str = "eli:has_part"
    article_value_relation: str = "prov:value"

    _creation_date_relation: str = "ext:creationDate"
    _with_taxonomy_relation: str = "ext:withTaxonomy"
    _with_user_relation: str = "ext:withUser"

    query_all_decisions: str = f"""\
    PREFIX besluit: <http://data.vlaanderen.be/ns/besluit#>
    PREFIX ext: <http://mu.semte.ch/vocabularies/ext/>
    
    
    SELECT ?_besluit WHERE {{{{
      ?_besluit a {besluit_relation}. 
      FILTER (!STRSTARTS(STR(?_besluit), "http://srv"))
    }}}}
    """

    query_all_decisions_with_specified_taxonomy: str = f"""\
    PREFIX besluit: <http://data.vlaanderen.be/ns/besluit#>
    PREFIX ext: <http://mu.semte.ch/vocabularies/ext/>
     
    
    SELECT ?_besluit ?date ?anno WHERE {{{{
      ?_besluit a {besluit_relation}. 
       
      {{{{
        SELECT * WHERE {{{{
          ?_besluit {annotation_relation} ?anno.
          ?anno {_creation_date_relation} ?date ; {_with_user_relation} ?user ; {_with_taxonomy_relation} ?taxonomy .
          
          FILTER (?taxonomy = <{{taxonomy_uri}}>)
          
        }}}} ORDER BY DESC (?date)
      }}}} 
    }}}}
    """

    query_latest_annotations: str = f"""\
    PREFIX besluit: <http://data.vlaanderen.be/ns/besluit#>
    PREFIX ext: <http://mu.semte.ch/vocabularies/ext/>
    
    
    SELECT * WHERE {{{{
      VALUES ?_besluit ?date {{{{ <{{decision_uri}}> }}}}
        ?_besluit {annotation_relation} ?anno .
        ?anno {_creation_date_relation} ?date.
        ?anno {_with_user_relation} ?user .
        ?anno {_with_taxonomy_relation} ?taxonomy .
    
        FILTER (?taxonomy = <{{taxonomy_uri}}>)
    
    }}}} ORDER BY DESC (?date) LIMIT 1
    """

    query_all_annotations: str = f"""\
    PREFIX ext: <http://mu.semte.ch/vocabularies/ext/>
        SELECT * WHERE {{{{
            <{{decision_uri}}> {annotation_relation} ?annotation_uri .
        }}}}
    """

    query_decision_information: str = f"""\
    PREFIX eli: <http://data.europa.eu/eli/ontology#>
    PREFIX prov: <http://www.w3.org/ns/prov#>
    PREFIX dct: <http://purl.org/dc/terms/>
    PREFIX besluit: <http://data.vlaanderen.be/ns/besluit#>
    
    SELECT * WHERE {{{{
    VALUES ?_besluit {{{{ <{{decision_uri}}> }}}}
     ?_besluit {generated_relation} ?_behandeling.
     ?_behandeling {subject_relation} ?_agendapunt.
     OPTIONAL {{{{?_besluit {description_relation} ?description.}}}}
     OPTIONAL {{{{?_besluit {short_title_relation} ?short_title.}}}}
     OPTIONAL {{{{?_besluit {motivation_relation} ?motivation.}}}}
     OPTIONAL {{{{?_besluit {publication_date_relation} ?publication_date.}}}}
     OPTIONAL {{{{?_besluit {langauge_relation} ?language.}}}}
    }}}}
    """

    query_all_articles: str = f"""\
    PREFIX besluit: <http://data.vlaanderen.be/ns/besluit#>
    PREFIX prov: <http://www.w3.org/ns/prov#>
    PREFIX dct: <http://purl.org/dc/terms/>
    PREFIX eli: <http://data.europa.eu/eli/ontology#>
    
    SELECT * WHERE {{{{
    ?_besluit a {besluit_relation}.
    ?_besluit eli:has_part ?_artikel.
      OPTIONAL {{{{ ?_artikel {haspart_relation} ?number. }}}}
      OPTIONAL {{{{ ?_artikel {article_value_relation} ?content. }}}}
    FILTER (?_besluit = <{{decision_uri}}>)
    }}}}
    ORDER BY ?number
    """

    insert_query: str = f"""\
    PREFIX ext: <http://mu.semte.ch/vocabularies/ext/>

    INSERT DATA {{{{
        GRAPH {{graph_uri}} {{{{
            <{{uri}}> {annotation_relation} {{annotation_uri}} .
            {{annotation_subquery}}
        }}}}
    }}}}
    """

    create_decision_from_uri: str = f"""\
    PREFIX besluit: <http://data.vlaanderen.be/ns/besluit#>
    PREFIX prov: <http://www.w3.org/ns/prov#>
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX eli: <http://data.europa.eu/eli/ontology#>
    PREFIX dct: <http://purl.org/dc/terms/>
    PREFIX ext:  <http://mu.semte.ch/vocabularies/ext/>    
    
    SELECT * WHERE
    {{{{
      VALUES ?besluit {{{{ <{{decision_uri}}> }}}}
      ?besluit ext:hasAnnotation ?anno .
      ?anno  ext:creationDate ?date ; ext:withTaxonomy ?taxonomy_uri .
      OPTIONAL 
      {{{{
        ?anno ext:withModel ?model_uri .
        {{{{
          SELECT * WHERE 
          {{{{
            ?model_uri ext:modelCategory ?category;
                       ext:registeredMlflowModel ?mlflow_model;
                       ext:modelName ?model_name;
                       ext:mlflowLink ?mlflow_link;
                       ext:creationDate ?create_data .
          }}}}
        }}}}
      }}}}
      OPTIONAL 
      {{{{
        ?anno ext:withUser ?user_uri .
    #    {{{{
    #      SELECT * WHERE 
    #      {{{{
    #      }}}}
    #    }}}}
      }}}}
      OPTIONAL
      {{{{
        SELECT DISTINCT ?anno (GROUP_CONCAT(?label_uri;separator="|") AS ?label_uris) (GROUP_CONCAT(?taxonomy_node;separator="|") AS ?taxonomy_node_uris) (GROUP_CONCAT(?score;separator="|") AS ?scores) WHERE 
        {{{{
          ?anno ext:hasLabel ?label_uri .
          ?label_uri ext:hasScore ?score ; ext:isTaxonomy ?taxonomy_node .
                
        }}}} GROUP BY ?anno
      }}}}
    }}}}
    """

    query_all_content: str = f"""\
    PREFIX eli: <http://data.europa.eu/eli/ontology#>
    PREFIX prov: <http://www.w3.org/ns/prov#>
    PREFIX dct: <http://purl.org/dc/terms/>
    PREFIX besluit: <http://data.vlaanderen.be/ns/besluit#>
        
    
    SELECT * WHERE {{{{
      VALUES ?_besluit {{{{ <{{decision_uri}}> }}}}
      ?_besluit prov:wasGeneratedBy ?_behandeling.
      ?_behandeling dct:subject ?_agendapunt.
      OPTIONAL {{{{?_besluit eli:description ?description.}}}}
      OPTIONAL {{{{?_besluit eli:title_short ?short_title.}}}}
      OPTIONAL {{{{?_besluit besluit:motivering ?motivation.}}}}
      OPTIONAL {{{{?_besluit eli:date_publication ?publication_date.}}}}
      OPTIONAL {{{{?_besluit eli:language ?language.}}}}
      
      OPTIONAL {{{{
        SELECT DISTINCT ?_besluit (GROUP_CONCAT(?_artikel;separator="|") AS ?artikels) (GROUP_CONCAT(?number;separator="|") AS ?numbers) (GROUP_CONCAT(?waarde;separator="|") AS ?waardes) WHERE {{{{
          ?_besluit eli:has_part ?_artikel.
          OPTIONAL {{{{ ?_artikel eli:number ?number. }}}}
          OPTIONAL {{{{ ?_artikel prov:value ?waarde. }}}}
        }}}} GROUP BY ?_besluit
        }}}}
    }}}}
    """

    class Config():
        env_prefix = f"{CONFIG_PREFIX}decision_"
