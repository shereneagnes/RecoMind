
"""
RecoMind - Content-Based Recommendation Engine

Uses TF-IDF and cosine similarity to recommend
articles based on textual content.
"""

from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_DIR / "data" / "processed" / "cleaned_articles.csv"


class ContentRecommender:

    def __init__(self):
        self.articles = None
        self.vectorizer = None
        self.tfidf_matrix = None
        self.article_indices = None

    def load_data(self):
        """Load cleaned article data."""

        if not DATA_PATH.exists():
            raise FileNotFoundError(
                "Cleaned articles not found. Run preprocessing.py first."
            )

        self.articles = pd.read_csv(DATA_PATH)

        self.articles["content"] = (
            self.articles["content"].fillna("")
        )

        self.articles = self.articles.reset_index(drop=True)

        self.article_indices = pd.Series(
            self.articles.index,
            index=self.articles["contentId"]
        ).drop_duplicates()

        print("Articles loaded:", len(self.articles))

    def build_model(self):
        """Convert article text into TF-IDF vectors."""

        if self.articles is None:
            self.load_data()

        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=15000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True
        )

        self.tfidf_matrix = self.vectorizer.fit_transform(
            self.articles["content"]
        )

        print("TF-IDF model created!")
        print("Matrix shape:", self.tfidf_matrix.shape)

    def recommend(self, content_id, top_n=10):
        """Recommend articles similar to a given article."""

        if self.tfidf_matrix is None:
            self.build_model()

        if content_id not in self.article_indices.index:
            raise ValueError("Article ID not found.")

        idx = self.article_indices.loc[content_id]

        if isinstance(idx, pd.Series):
            idx = idx.iloc[0]

        scores = cosine_similarity(
            self.tfidf_matrix[idx],
            self.tfidf_matrix
        ).flatten()

        similar_indices = scores.argsort()[::-1]

        similar_indices = [
            i for i in similar_indices if i != idx
        ][:top_n]

        results = self.articles.iloc[similar_indices][
            ["contentId", "title", "url"]
        ].copy()

        results["similarity_score"] = scores[similar_indices]

        return results.reset_index(drop=True)

    def recommend_by_title(self, title, top_n=10):
        """Find an article by title and recommend similar articles."""

        if self.tfidf_matrix is None:
            self.build_model()

        matches = self.articles[
            self.articles["title"].str.contains(
                title, case=False, na=False, regex=False
            )
        ]

        if matches.empty:
            raise ValueError("No article found with that title.")

        content_id = matches.iloc[0]["contentId"]

        return self.recommend(content_id, top_n)


if __name__ == "__main__":

    recommender = ContentRecommender()

    recommender.load_data()
    recommender.build_model()

    print("\nModel ready!")

    sample_id = recommender.articles.iloc[0]["contentId"]

    print("\nSample article:")
    print(recommender.articles.iloc[0]["title"])

    print("\nRecommended articles:")

    recommendations = recommender.recommend(sample_id, top_n=10)

    print(recommendations.to_string(index=False))
