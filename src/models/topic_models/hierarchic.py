from __future__ import annotations

from .regular import RegularTopicModel

from scipy.cluster import hierarchy as sch
from bertopic import BERTopic
import mlflow, os


class HierarchicTopicModel(RegularTopicModel):
    """
    Regular topic modeling implementation.

    This class creates the following artifacts:
        1. Topic report (keywords per topic and the found topics in excels sheet)
        2. Topic distribution plot (barchart that visualizes what amount of docs can be found under the topics)
        3. Topic heatmap (a plot that visually can represent how close certain topics are to each-other)
    """

    def analyse(self):
        with mlflow.start_run():
            topic_model = BERTopic()
            topics, probs = topic_model.fit_transform(self.docs)

            topic_model.visualize_documents(self.docs).write_html(
                open(
                    os.path.join(
                        self.base_path,
                        "visualized_documents.html"
                    ),
                    "w"
                )
            )

            linkage_function = lambda x: sch.linkage(x, 'single', optimal_ordering=True)
            hierarchical_topics = topic_model.hierarchical_topics(
                docs=self.docs,
                linkage_function=linkage_function
            )

            tree = topic_model.get_topic_tree(hierarchical_topics)

            with open(os.path.join(self.base_path, "topic_tree") , "w") as f:
                f.write(tree)

            topic_model.visualize_hierarchy(
                hierarchical_topics=hierarchical_topics
            ).write_html(
                open(
                    os.path.join(
                        self.base_path,
                        "hierarchical_topics.html"
                    ),
                    "w"
                )
            )

            mlflow.log_artifacts(
                local_dir=self.base_path,
                artifact_path="artifacts"
            )