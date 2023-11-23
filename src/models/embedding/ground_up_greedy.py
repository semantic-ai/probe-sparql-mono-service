import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from .ground_up import GroundUpRegularEmbeddingModel


class GroundUpGreedyEmbeddingModel(GroundUpRegularEmbeddingModel):
    """
    Ground up greedy approach only takes the values with the highest possible scores
    """

    def classify(self, text: str, multi_label, **kwargs) -> dict[str, float]:
        text_embedding = np.asarray(self._embed(text))

        if labels := kwargs.get("labels", None):
            self._prep_labels(labels)

        similarity = cosine_similarity(np.asarray([text_embedding]), self.embedding_matrix)[0]

        self.logger.debug(f"Indexes: {self.indexes}")

        # create dataframe for easy builtin functions
        index_index_mapper = pd.DataFrame(
            list(
                zip(
                    self.indexes,
                    similarity
                )
            ),
            columns=["indexes", "scores"]
        )

        self.logger.debug(f"Index_mapper: {index_index_mapper}")

        # remap average score per subset and compute
        level_score_dict = {
            record.get("indexes"): record.get("scores")
            for record in index_index_mapper.groupby("indexes").agg(
                {
                    "indexes": "first",
                    "scores": "mean"
                }
            ).to_dict(orient="records")
        }
        self.logger.debug(f"level_Score_dict: {level_score_dict}")

        master_keys = [k for k in level_score_dict.keys() if "." not in k]
        self.logger.debug(f"Master keys: {master_keys}")

        average = dict()
        for master_key in master_keys:
            average[int(master_key)] = np.max([
                v for k, v in level_score_dict.items()
                if k.startswith(master_key)
            ])

        return_scores = {k: v for k, v in zip(self.labels, list(average.values()))}
        self.logger.debug(f"Return scores: {return_scores}")

        return return_scores

    classify.__doc__ = GroundUpRegularEmbeddingModel.classify.__doc__
