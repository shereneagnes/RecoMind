
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------------------------------
# PATH CONFIGURATION
# --------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT_DIR / "data" / "processed"
RESULTS_DIR = ROOT_DIR / "results"
EDA_DIR = RESULTS_DIR / "eda"

INTERACTIONS_PATH = DATA_DIR / "cleaned_interactions.csv"
ARTICLES_PATH = DATA_DIR / "cleaned_articles.csv"

EDA_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

def load_data():

    print("\nLoading datasets...")

    interactions = pd.read_csv(INTERACTIONS_PATH)
    articles = pd.read_csv(ARTICLES_PATH)

    interactions["timestamp"] = pd.to_datetime(
        interactions["timestamp"],
        errors="coerce",
        utc=True
    )

    print("Interactions loaded:", len(interactions))
    print("Articles loaded:", len(articles))

    return interactions, articles


# --------------------------------------------------
# 1. INTERACTION TYPE DISTRIBUTION
# --------------------------------------------------

def interaction_distribution(interactions):

    print("\n1. Analysing interaction types...")

    counts = interactions["eventType"].value_counts()

    plt.figure(figsize=(9, 5))
    sns.barplot(x=counts.index, y=counts.values)

    plt.title("Distribution of Interaction Types")
    plt.xlabel("Interaction Type")
    plt.ylabel("Number of Interactions")
    plt.xticks(rotation=30)

    plt.tight_layout()
    plt.savefig(EDA_DIR / "interaction_distribution.png", dpi=300)
    plt.close()

    counts.to_csv(EDA_DIR / "interaction_type_counts.csv")

    print(counts)


# --------------------------------------------------
# 2. MOST POPULAR ARTICLES
# --------------------------------------------------

def popular_articles(interactions, articles):

    print("\n2. Finding most popular articles...")

    popular = (
        interactions.groupby("contentId")
        .size()
        .reset_index(name="interaction_count")
        .sort_values("interaction_count", ascending=False)
        .head(10)
    )

    if "contentId" in articles.columns:

        title_column = next(
            (c for c in ["title", "headline", "name"] if c in articles.columns),
            None
        )

        if title_column:
            popular = popular.merge(
                articles[["contentId", title_column]].drop_duplicates("contentId"),
                on="contentId",
                how="left"
            )

    print(popular.to_string(index=False))

    plt.figure(figsize=(11, 6))

    labels = (
        popular[title_column].fillna("Unknown").astype(str)
        if "title_column" in locals() and title_column
        else popular["contentId"].astype(str)
    )

    sns.barplot(
        x=popular["interaction_count"],
        y=labels
    )

    plt.title("Top 10 Most Popular Articles")
    plt.xlabel("Number of Interactions")
    plt.ylabel("Article")

    plt.tight_layout()
    plt.savefig(EDA_DIR / "popular_articles.png", dpi=300)
    plt.close()

    popular.to_csv(EDA_DIR / "popular_articles.csv", index=False)


# --------------------------------------------------
# 3. MOST ACTIVE USERS
# --------------------------------------------------

def active_users(interactions):

    print("\n3. Finding most active users...")

    active = (
        interactions.groupby("personId")
        .size()
        .reset_index(name="interaction_count")
        .sort_values("interaction_count", ascending=False)
        .head(10)
    )

    print(active.to_string(index=False))

    plt.figure(figsize=(10, 5))

    sns.barplot(
        x="interaction_count",
        y="personId",
        data=active
    )

    plt.title("Top 10 Most Active Users")
    plt.xlabel("Number of Interactions")
    plt.ylabel("User ID")

    plt.tight_layout()
    plt.savefig(EDA_DIR / "active_users.png", dpi=300)
    plt.close()

    active.to_csv(EDA_DIR / "active_users.csv", index=False)


# --------------------------------------------------
# 4. ARTICLE POPULARITY DISTRIBUTION
# --------------------------------------------------

def popularity_distribution(interactions):

    print("\n4. Analysing article popularity...")

    popularity = interactions.groupby("contentId").size()

    plt.figure(figsize=(9, 5))

    sns.histplot(
        popularity,
        bins=40,
        kde=True
    )

    plt.title("Article Popularity Distribution")
    plt.xlabel("Number of Interactions per Article")
    plt.ylabel("Number of Articles")

    plt.tight_layout()
    plt.savefig(EDA_DIR / "article_popularity_distribution.png", dpi=300)
    plt.close()

    print("\nPopularity statistics:")
    print(popularity.describe().to_string())

    popularity.describe().to_csv(
        EDA_DIR / "article_popularity_statistics.csv"
    )


# --------------------------------------------------
# 5. INTERACTION TRENDS OVER TIME
# --------------------------------------------------

def interaction_trends(interactions):

    print("\n5. Analysing interaction trends over time...")

    dated = interactions.dropna(subset=["timestamp"]).copy()

    if dated.empty:
        print("No valid timestamps found. Skipping time analysis.")
        return

    dated["date"] = dated["timestamp"].dt.date

    daily = (
        dated.groupby("date")
        .size()
        .reset_index(name="interaction_count")
    )

    plt.figure(figsize=(12, 5))

    sns.lineplot(
        x="date",
        y="interaction_count",
        data=daily
    )

    plt.title("Daily Interaction Trends")
    plt.xlabel("Date")
    plt.ylabel("Number of Interactions")
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(EDA_DIR / "interaction_trends.png", dpi=300)
    plt.close()

    daily.to_csv(EDA_DIR / "daily_interaction_trends.csv", index=False)


# --------------------------------------------------
# 6. USER ENGAGEMENT PATTERNS
# --------------------------------------------------

def user_engagement(interactions):

    print("\n6. Analysing user engagement patterns...")

    engagement = (
        interactions.groupby("personId")
        .agg(
            total_interactions=("contentId", "count"),
            unique_articles=("contentId", "nunique"),
            interaction_types=("eventType", "nunique")
        )
        .reset_index()
    )

    print("\nUser engagement statistics:")
    print(engagement[
        ["total_interactions", "unique_articles", "interaction_types"]
    ].describe().to_string())

    plt.figure(figsize=(9, 5))

    sns.scatterplot(
        x="unique_articles",
        y="total_interactions",
        data=engagement,
        alpha=0.5
    )

    plt.title("User Engagement: Articles Explored vs Interactions")
    plt.xlabel("Number of Unique Articles")
    plt.ylabel("Total Interactions")

    plt.tight_layout()
    plt.savefig(EDA_DIR / "user_engagement.png", dpi=300)
    plt.close()

    engagement.to_csv(EDA_DIR / "user_engagement.csv", index=False)


# --------------------------------------------------
# MAIN EXECUTION
# --------------------------------------------------

def run_eda():

    print("\n" + "=" * 55)
    print("       RECOMIND - EXPLORATORY DATA ANALYSIS")
    print("=" * 55)

    interactions, articles = load_data()

    interaction_distribution(interactions)
    popular_articles(interactions, articles)
    active_users(interactions)
    popularity_distribution(interactions)
    interaction_trends(interactions)
    user_engagement(interactions)

    print("\n" + "=" * 55)
    print("EDA COMPLETED SUCCESSFULLY!")
    print("Charts and reports saved in:", EDA_DIR)
    print("=" * 55)


if __name__ == "__main__":
    run_eda()
