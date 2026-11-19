
"""
RecoMind - Recommendation Model Evaluation

Evaluates Content-Based, Collaborative Filtering,
and Hybrid Recommendation models.
"""

import os
import shutil
import numpy as np
import pandas as pd

from src.content_recommender import ContentRecommender
from src.collaborative_recommender import CollaborativeRecommender
from src.hybrid_recommender import HybridRecommender


# ==========================================
# 1. EVALUATION METRICS
# ==========================================

def precision_at_k(recommended, relevant, k=10):

    if k <= 0:
        return 0.0

    hits = len(set(recommended[:k]) & set(relevant))

    return hits / k


def recall_at_k(recommended, relevant, k=10):

    if not relevant:
        return 0.0

    hits = len(set(recommended[:k]) & set(relevant))

    return hits / len(relevant)


def f1_at_k(recommended, relevant, k=10):

    precision = precision_at_k(recommended, relevant, k)
    recall = recall_at_k(recommended, relevant, k)

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


def average_precision_at_k(recommended, relevant, k=10):

    if not relevant or k <= 0:
        return 0.0

    recommended = recommended[:k]

    hits = 0
    score = 0.0
    seen = set()

    for i, item in enumerate(recommended):

        if item in relevant and item not in seen:

            hits += 1
            score += hits / (i + 1)
            seen.add(item)

    return score / min(len(relevant), k)


def ndcg_at_k(recommended, relevant, k=10):

    if not relevant or k <= 0:
        return 0.0

    recommended = recommended[:k]

    dcg = sum(
        (1 if item in relevant else 0) / np.log2(i + 2)
        for i, item in enumerate(recommended)
    )

    ideal_hits = min(len(relevant), k)

    idcg = sum(
        1 / np.log2(i + 2)
        for i in range(ideal_hits)
    )

    return dcg / idcg if idcg else 0.0


# ==========================================
# 2. EVALUATE A RECOMMENDER
# ==========================================

def evaluate_recommender(recommendations, ground_truth, k=10):

    metrics = {
        "Precision@10": [],
        "Recall@10": [],
        "F1@10": [],
        "MAP@10": [],
        "NDCG@10": []
    }

    for user_id, relevant in ground_truth.items():

        recommended = recommendations.get(user_id, [])

        metrics["Precision@10"].append(
            precision_at_k(recommended, relevant, k)
        )

        metrics["Recall@10"].append(
            recall_at_k(recommended, relevant, k)
        )

        metrics["F1@10"].append(
            f1_at_k(recommended, relevant, k)
        )

        metrics["MAP@10"].append(
            average_precision_at_k(recommended, relevant, k)
        )

        metrics["NDCG@10"].append(
            ndcg_at_k(recommended, relevant, k)
        )

    return {
        name: float(np.mean(values)) if values else 0.0
        for name, values in metrics.items()
    }


# ==========================================
# 3. PREPARE TRAINING AND TEST DATA
# ==========================================

def prepare_test_data():

    interaction_path = (
        "data/processed/cleaned_interactions.csv"
    )

    matrix_path = (
        "data/processed/user_item_interactions.csv"
    )

    interactions = pd.read_csv(interaction_path)

    print("Original interactions:", len(interactions))

    # Use actual dataset column names
    interactions = interactions[
        ["personId", "contentId", "interaction_weight"]
    ].dropna()

    # Aggregate repeated interactions
    interactions = interactions.groupby(
        ["personId", "contentId"],
        as_index=False
    )["interaction_weight"].sum()

    # Keep users with at least two distinct articles
    user_counts = interactions.groupby(
        "personId"
    )["contentId"].nunique()

    valid_users = user_counts[
        user_counts >= 2
    ].index

    interactions = interactions[
        interactions["personId"].isin(valid_users)
    ].copy()

    train_parts = []
    test_parts = []

    # Split each user's interactions
    for user_id, group in interactions.groupby("personId"):

        group = group.sample(
            frac=1,
            random_state=42
        )

        test_size = max(1, int(len(group) * 0.2))

        test = group.iloc[:test_size]
        train = group.iloc[test_size:]

        train_parts.append(train)
        test_parts.append(test)

    if not train_parts:
        raise ValueError("Insufficient training data.")

    train_data = pd.concat(
        train_parts,
        ignore_index=True
    )

    test_data = pd.concat(
        test_parts,
        ignore_index=True
    )

    # IMPORTANT:
    # CollaborativeRecommender expects long-format data
    # with interaction_strength, not a wide matrix.

    training_interactions = train_data.rename(
        columns={
            "interaction_weight": "interaction_strength"
        }
    )

    training_interactions = training_interactions[
        ["personId", "contentId", "interaction_strength"]
    ]

    # Save training interactions in the expected format
    training_interactions.to_csv(
        matrix_path,
        index=False
    )

    print("Training interactions:", len(training_interactions))
    print("Testing interactions:", len(test_data))

    # Ground truth from held-out interactions
    ground_truth = (
        test_data.groupby("personId")["contentId"]
        .apply(lambda x: list(set(x.astype(str))))
        .to_dict()
    )

    # Select strongest training article for each user
    seed_articles = (
        train_data.sort_values(
            "interaction_weight",
            ascending=False
        )
        .drop_duplicates("personId")
        .set_index("personId")["contentId"]
        .to_dict()
    )

    return matrix_path, ground_truth, seed_articles


# ==========================================
# 4. EXTRACT RECOMMENDED ARTICLE IDS
# ==========================================

def extract_ids(result):

    if result is None:
        return []

    if isinstance(result, pd.DataFrame):

        if "contentId" not in result.columns:
            return []

        return result["contentId"].astype(str).tolist()

    if isinstance(result, pd.Series):
        return result.astype(str).tolist()

    if isinstance(result, list):

        output = []

        for item in result:

            if isinstance(item, dict):
                output.append(str(item["contentId"]))

            else:
                output.append(str(item))

        return output

    return []


# ==========================================
# 5. RUN EVALUATION
# ==========================================

def run_evaluation(k=10, max_users=200):

    matrix_path = (
        "data/processed/user_item_interactions.csv"
    )

    # Back up the original interaction file
    backup_path = "user_item_interactions_backup.csv"

    original_exists = os.path.exists(matrix_path)

    if original_exists:
        shutil.copy2(matrix_path, backup_path)

    try:

        print("\nPreparing train/test dataset...")

        matrix_path, ground_truth, seed_articles = (
            prepare_test_data()
        )

        print("Evaluation users:", len(ground_truth))

        # Select users for evaluation
        user_ids = list(ground_truth.keys())[:max_users]

        ground_truth = {
            user: ground_truth[user]
            for user in user_ids
        }

        print("\nBuilding Content-Based Model...")

        content_model = ContentRecommender()

        content_model.load_data()
        content_model.build_model()

        print("\nBuilding Collaborative Model...")

        collaborative_model = CollaborativeRecommender()

        collaborative_model.load_data()
        collaborative_model.build_model()

        print("\nBuilding Hybrid Model...")

        hybrid_model = HybridRecommender(
            content_weight=0.5
        )

        hybrid_model.build_model()

        print("\nGenerating recommendations...")

        content_results = {}
        collaborative_results = {}
        hybrid_results = {}

        for i, user_id in enumerate(user_ids):

            print(
                f"Evaluating user {i + 1}/{len(user_ids)}",
                end="\r"
            )

            # --------------------------------
            # CONTENT-BASED RECOMMENDATIONS
            # --------------------------------

            seed = seed_articles.get(user_id)

            if seed is not None:

                result = content_model.recommend(
                    seed,
                    top_n=k
                )

                content_results[user_id] = extract_ids(result)

            else:

                content_results[user_id] = []

            # --------------------------------
            # COLLABORATIVE RECOMMENDATIONS
            # --------------------------------

            try:

                result = collaborative_model.recommend(
                    user_id,
                    top_n=k
                )

                collaborative_results[user_id] = extract_ids(
                    result
                )

            except ValueError:

                collaborative_results[user_id] = []

            # --------------------------------
            # HYBRID RECOMMENDATIONS
            # --------------------------------

            try:

                result = hybrid_model.recommend(
                    person_id=user_id,
                    content_id=seed,
                    top_n=k
                )

                hybrid_results[user_id] = extract_ids(result)

            except ValueError:

                hybrid_results[user_id] = []

        print("\n\nCalculating evaluation metrics...")

        # --------------------------------
        # COMPARE ALL MODELS
        # --------------------------------

        model_recommendations = {
            "Content-Based": content_results,
            "Collaborative Filtering": collaborative_results,
            "Hybrid": hybrid_results
        }

        results = {}

        for model_name, recommendations in (
            model_recommendations.items()
        ):

            print("Evaluating:", model_name)

            results[model_name] = evaluate_recommender(
                recommendations,
                ground_truth,
                k
            )

        results_df = pd.DataFrame(results).T

        # --------------------------------
        # DISPLAY RESULTS
        # --------------------------------

        print("\n")
        print("=" * 65)
        print("       RECOMIND MODEL EVALUATION RESULTS")
        print("=" * 65)

        print(results_df.round(4).to_string())

        # --------------------------------
        # SAVE RESULTS
        # --------------------------------

        os.makedirs("results", exist_ok=True)

        results_df.to_csv(
            "results/model_evaluation.csv"
        )

        print("\nEvaluation completed successfully!")

        print(
            "Results saved: results/model_evaluation.csv"
        )

        return results_df

    finally:

        # Restore original interaction dataset
        if original_exists:

            shutil.copy2(
                backup_path,
                matrix_path
            )

            os.remove(backup_path)

            print("\nOriginal interaction dataset restored.")

        elif os.path.exists(matrix_path):

            os.remove(matrix_path)


# ==========================================
# 6. MAIN
# ==========================================

if __name__ == "__main__":

    print("\n========== RECOMIND MODEL EVALUATION ==========")

    run_evaluation(
        k=10,
        max_users=200
    )
