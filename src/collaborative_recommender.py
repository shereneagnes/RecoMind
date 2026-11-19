
"""
RecoMind - Collaborative Filtering Recommendation Engine

Uses user interaction patterns and item-item similarity
to recommend articles.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity


ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    ROOT_DIR / "data" / "processed"
    / "user_item_interactions.csv"
)

ARTICLES_PATH = (
    ROOT_DIR / "data" / "processed"
    / "cleaned_articles.csv"
)


class CollaborativeRecommender:

    def __init__(self):

        self.interactions = None
        self.articles = None

        self.user_item_matrix = None
        self.item_similarity = None

        self.user_map = {}
        self.item_map = {}
        self.reverse_item_map = {}

    def load_data(self):

        self.interactions = pd.read_csv(DATA_PATH)
        self.articles = pd.read_csv(ARTICLES_PATH)

        print("Interactions loaded:", len(self.interactions))
        print("Articles loaded:", len(self.articles))

    def build_model(self):

        if self.interactions is None:
            self.load_data()

        # Create user and article mappings
        users = self.interactions["personId"].unique()
        items = self.interactions["contentId"].unique()

        self.user_map = {
            user: index
            for index, user in enumerate(users)
        }

        self.item_map = {
            item: index
            for index, item in enumerate(items)
        }

        self.reverse_item_map = {
            index: item
            for item, index in self.item_map.items()
        }

        # Convert IDs into matrix indices
        row = self.interactions["personId"].map(
            self.user_map
        )

        col = self.interactions["contentId"].map(
            self.item_map
        )

        values = self.interactions[
            "interaction_strength"
        ].astype(float)

        # Construct sparse user-item matrix
        self.user_item_matrix = csr_matrix(
            (values, (row, col)),
            shape=(len(users), len(items))
        )

        # Item-item similarity
        item_user_matrix = self.user_item_matrix.T

        self.item_similarity = cosine_similarity(
            item_user_matrix,
            dense_output=False
        )

        print("User-item matrix:", self.user_item_matrix.shape)
        print("Item similarity matrix:", self.item_similarity.shape)

    def recommend(self, person_id, top_n=10):

        if self.user_item_matrix is None:
            self.build_model()

        if person_id not in self.user_map:
            raise ValueError("User ID not found.")

        user_index = self.user_map[person_id]

        user_vector = self.user_item_matrix.getrow(
            user_index
        )

        # Predict scores for unseen articles
        scores = user_vector @ self.item_similarity

        scores = np.asarray(scores.todense()).flatten()

        # Exclude articles already interacted with
        interacted_items = set(
            user_vector.indices
        )

        scores[list(interacted_items)] = -np.inf

        # Get top recommendations
        top_indices = np.argsort(scores)[::-1][:top_n]

        recommendations = []

        for index in top_indices:

            if not np.isfinite(scores[index]):
                continue

            content_id = self.reverse_item_map[index]

            recommendations.append({
                "contentId": content_id,
                "recommendation_score": scores[index]
            })

        result = pd.DataFrame(recommendations)

        if result.empty:
            return result

        result = result.merge(
            self.articles[
                ["contentId", "title", "url"]
            ],
            on="contentId",
            how="left"
        )

        return result[
            ["contentId", "title", "url", "recommendation_score"]
        ]

    def recommend_for_active_users(self, top_n=10):

        if self.user_item_matrix is None:
            self.build_model()

        # Select the most active user
        activity = self.user_item_matrix.getnnz(axis=1)

        user_index = int(np.argmax(activity))

        person_id = list(self.user_map.keys())[user_index]

        return person_id, self.recommend(person_id, top_n)


if __name__ == "__main__":

    recommender = CollaborativeRecommender()

    recommender.load_data()
    recommender.build_model()

    print("\nModel successfully built!")

    person_id, recommendations = (
        recommender.recommend_for_active_users(top_n=10)
    )

    print("\nSample user:", person_id)

    print("\nRecommended articles:")

    print(recommendations.to_string(index=False))
