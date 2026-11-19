
"""
RecoMind - Hybrid Recommendation Engine

Combines Content-Based and Collaborative Filtering
to generate personalized article recommendations.
"""

import numpy as np
import pandas as pd

from src.content_recommender import ContentRecommender
from src.collaborative_recommender import CollaborativeRecommender


class HybridRecommender:

    def __init__(self, content_weight=0.5):

        if not 0 <= content_weight <= 1:
            raise ValueError("content_weight must be between 0 and 1")

        self.content_weight = content_weight
        self.collaborative_weight = 1 - content_weight

        self.content_model = ContentRecommender()
        self.collaborative_model = CollaborativeRecommender()

        self.is_built = False

    def build_model(self):

        print("\nBuilding Content-Based Model...")

        self.content_model.load_data()
        self.content_model.build_model()

        print("\nBuilding Collaborative Model...")

        self.collaborative_model.load_data()
        self.collaborative_model.build_model()

        self.is_built = True

        print("\nHybrid Recommendation Engine Ready!")

    @staticmethod
    def normalize_scores(scores):

        scores = np.asarray(scores, dtype=float)

        if len(scores) == 0:
            return scores

        min_score = scores.min()
        max_score = scores.max()

        if max_score == min_score:
            return np.ones(len(scores))

        return (scores - min_score) / (max_score - min_score)

    def recommend(self, person_id, content_id=None, top_n=10):

        if not self.is_built:
            self.build_model()

        if person_id not in self.collaborative_model.user_map:
            raise ValueError("User ID not found.")

        if top_n <= 0:
            raise ValueError("top_n must be positive.")

        # ----------------------------------
        # 1. Collaborative Recommendations
        # ----------------------------------

        cf_results = self.collaborative_model.recommend(
            person_id,
            top_n=100
        )

        # ----------------------------------
        # 2. Select Reference Article
        # ----------------------------------

        if content_id is None:

            user_index = (
                self.collaborative_model.user_map[person_id]
            )

            user_vector = (
                self.collaborative_model.user_item_matrix
                .getrow(user_index)
            )

            if user_vector.nnz == 0:
                raise ValueError("User has no interaction history.")

            strongest_index = user_vector.indices[
                np.argmax(user_vector.data)
            ]

            content_id = (
                self.collaborative_model.reverse_item_map[
                    strongest_index
                ]
            )

        # ----------------------------------
        # 3. Content-Based Recommendations
        # ----------------------------------

        content_results = self.content_model.recommend(
            content_id,
            top_n=100
        ).copy()

        content_results = content_results[
            ["contentId", "similarity_score"]
        ]

        content_results["content_score"] = (
            self.normalize_scores(
                content_results["similarity_score"]
            )
        )

        content_results = content_results[
            ["contentId", "content_score"]
        ]

        # ----------------------------------
        # 4. Normalize Collaborative Scores
        # ----------------------------------

        if not cf_results.empty:

            cf_results = cf_results.copy()

            cf_results["collaborative_score"] = (
                self.normalize_scores(
                    cf_results["recommendation_score"]
                )
            )

            cf_results = cf_results[
                ["contentId", "collaborative_score"]
            ]

        else:

            cf_results = pd.DataFrame(
                columns=[
                    "contentId",
                    "collaborative_score"
                ]
            )

        # ----------------------------------
        # 5. Combine Recommendation Lists
        # ----------------------------------

        merged = pd.merge(
            content_results,
            cf_results,
            on="contentId",
            how="outer"
        )

        merged["content_score"] = (
            merged["content_score"].fillna(0)
        )

        merged["collaborative_score"] = (
            merged["collaborative_score"].fillna(0)
        )

        # ----------------------------------
        # 6. Calculate Hybrid Score
        # ----------------------------------

        merged["hybrid_score"] = (
            self.content_weight * merged["content_score"]
            +
            self.collaborative_weight
            * merged["collaborative_score"]
        )

        # ----------------------------------
        # 7. Attach Article Metadata
        # ----------------------------------

        article_metadata = (
            self.content_model.articles[
                ["contentId", "title", "url"]
            ]
            .drop_duplicates(subset=["contentId"])
        )

        merged = merged.merge(
            article_metadata,
            on="contentId",
            how="inner"
        )

        # ----------------------------------
        # 8. Remove Previously Interacted Articles
        # ----------------------------------

        user_index = (
            self.collaborative_model.user_map[person_id]
        )

        user_vector = (
            self.collaborative_model.user_item_matrix
            .getrow(user_index)
        )

        interacted_ids = {
            self.collaborative_model.reverse_item_map[i]
            for i in user_vector.indices
        }

        merged = merged[
            ~merged["contentId"].isin(interacted_ids)
        ]

        # ----------------------------------
        # 9. Sort and Return Recommendations
        # ----------------------------------

        merged = merged.sort_values(
            by="hybrid_score",
            ascending=False
        )

        merged = merged.drop_duplicates(
            subset=["contentId"]
        )

        return merged[
            [
                "contentId",
                "title",
                "url",
                "content_score",
                "collaborative_score",
                "hybrid_score"
            ]
        ].head(top_n).reset_index(drop=True)


if __name__ == "__main__":

    recommender = HybridRecommender(
        content_weight=0.5
    )

    recommender.build_model()

    person_id, _ = (
        recommender.collaborative_model
        .recommend_for_active_users(top_n=10)
    )

    print("\nSelected User:", person_id)

    print("\nGenerating Hybrid Recommendations...")

    recommendations = recommender.recommend(
        person_id=person_id,
        top_n=10
    )

    print("\n========== HYBRID RECOMMENDATIONS ==========\n")

    if recommendations.empty:
        print("No recommendations available.")

    else:
        print(
            recommendations.to_string(index=False)
        )

    print("\nRecommendation generation completed!")
