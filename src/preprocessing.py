
"""
RecoMind - Data Preprocessing Module

Cleans article metadata and user interaction data
for the recommendation engine.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd


# Project paths
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"

ARTICLES_PATH = DATA_DIR / "shared_articles.csv"
INTERACTIONS_PATH = DATA_DIR / "users_interactions.csv"

# Engagement weights
EVENT_WEIGHTS = {
    "VIEW": 1,
    "LIKE": 2,
    "FOLLOW": 3,
    "BOOKMARK": 3,
    "COMMENT CREATED": 4,
}


def clean_text(text):
    """Clean HTML tags, whitespace and unwanted characters."""

    if pd.isna(text):
        return ""

    text = str(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def load_data():
    """Load the original CSV datasets."""

    if not ARTICLES_PATH.exists():
        raise FileNotFoundError(f"Missing file: {ARTICLES_PATH}")

    if not INTERACTIONS_PATH.exists():
        raise FileNotFoundError(f"Missing file: {INTERACTIONS_PATH}")

    articles = pd.read_csv(ARTICLES_PATH)
    interactions = pd.read_csv(INTERACTIONS_PATH)

    return articles, interactions


def clean_articles(articles):
    """Clean article metadata and textual content."""

    df = articles.copy()

    # Retain only published/shared articles
    df = df[df["eventType"] == "CONTENT SHARED"].copy()

    # Remove articles without an identifier
    df = df.dropna(subset=["contentId"])

    # Remove duplicate article identifiers
    df = df.drop_duplicates(subset=["contentId"], keep="last")

    # Handle missing textual fields
    df["title"] = df["title"].fillna("")
    df["text"] = df["text"].fillna("")

    # Clean article content
    df["title"] = df["title"].apply(clean_text)
    df["text"] = df["text"].apply(clean_text)

    # Remove articles without usable text
    df["content"] = (
        df["title"] + " " + df["text"]
    ).str.strip()

    df = df[df["content"].str.len() > 0].copy()

    # Normalize language metadata
    df["lang"] = df["lang"].fillna("unknown")

    return df.reset_index(drop=True)


def clean_interactions(interactions, articles):
    """Clean interactions and calculate engagement strength."""

    df = interactions.copy()

    # Remove records with missing identifiers
    df = df.dropna(subset=["personId", "contentId"])

    # Keep supported interaction types
    df = df[df["eventType"].isin(EVENT_WEIGHTS)].copy()

    # Remove interactions for unavailable articles
    valid_ids = set(articles["contentId"])
    df = df[df["contentId"].isin(valid_ids)].copy()

    # Assign engagement weights
    df["interaction_weight"] = df["eventType"].map(
        EVENT_WEIGHTS
    )

    # Convert identifiers into consistent numeric types
    df["personId"] = pd.to_numeric(
        df["personId"], errors="coerce"
    )
    df["contentId"] = pd.to_numeric(
        df["contentId"], errors="coerce"
    )

    df = df.dropna(subset=["personId", "contentId"])

    return df.reset_index(drop=True)


def preprocess_data():
    """Run the complete preprocessing pipeline."""

    print("\n========== RecoMind Preprocessing ==========\n")

    articles, interactions = load_data()

    print("Original articles:", len(articles))
    print("Original interactions:", len(interactions))

    cleaned_articles = clean_articles(articles)

    cleaned_interactions = clean_interactions(
        interactions,
        cleaned_articles
    )

    # Aggregate repeated user-article interactions
    user_item = (
        cleaned_interactions
        .groupby(["personId", "contentId"], as_index=False)
        .agg(
            interaction_strength=("interaction_weight", "sum"),
            interaction_count=("eventType", "count"),
            last_interaction=("timestamp", "max")
        )
    )

    # Save processed datasets
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    cleaned_articles.to_csv(
        PROCESSED_DIR / "cleaned_articles.csv",
        index=False
    )

    cleaned_interactions.to_csv(
        PROCESSED_DIR / "cleaned_interactions.csv",
        index=False
    )

    user_item.to_csv(
        PROCESSED_DIR / "user_item_interactions.csv",
        index=False
    )

    print("\n========== Preprocessing Results ==========")

    print("Cleaned articles:", len(cleaned_articles))
    print("Cleaned interactions:", len(cleaned_interactions))
    print("Unique users:", cleaned_interactions["personId"].nunique())
    print("Unique articles:", cleaned_interactions["contentId"].nunique())
    print("Unique user-article pairs:", len(user_item))

    print("\nSaved files:")
    print("1. cleaned_articles.csv")
    print("2. cleaned_interactions.csv")
    print("3. user_item_interactions.csv")

    print("\nPreprocessing completed successfully!")


if __name__ == "__main__":
    preprocess_data()
