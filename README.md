# RecoMind – AI-Powered Hybrid Recommendation System

RecoMind is an AI-powered personalized content discovery system that recommends relevant articles by combining Content-Based Filtering and Collaborative Filtering.

Built using the CI&T Deskdrop dataset, RecoMind analyses article content and user interaction patterns to generate personalized recommendations through a hybrid recommendation engine.

The project includes data preprocessing, exploratory data analysis, recommendation model development, quantitative evaluation, and an interactive Streamlit dashboard.

## Key Features

- **Content-Based Filtering:** Recommends articles based on textual similarity.
- **TF-IDF Vectorization:** Converts article text into numerical feature representations.
- **Cosine Similarity:** Identifies articles with similar content.
- **Collaborative Filtering:** Uses user-item interaction patterns to generate recommendations.
- **Hybrid Recommendation Engine:** Combines content-based and collaborative scores using configurable weights.
- **Exploratory Data Analysis:** Examines interaction types, article popularity, user engagement, and activity patterns.
- **Model Evaluation:** Compares recommendation models using Precision@10, Recall@10, F1@10, MAP@10, and NDCG@10.
- **Interactive Dashboard:** Allows users to explore analytics, compare models, and generate personalized recommendations.

## Recommendation Architecture

RecoMind uses three recommendation approaches.

### 1. Content-Based Filtering

Article metadata and textual content are processed using TF-IDF vectorization.

Cosine similarity is calculated between article representations to identify content that is similar to a user's previously interacted articles.

### 2. Collaborative Filtering

User interaction data is used to construct a user-item interaction matrix.

Item similarity is calculated from interaction patterns to identify articles that may interest users based on collective behaviour.

### 3. Hybrid Recommendation

The hybrid engine combines both recommendation approaches using configurable weights.

The final recommendation score is calculated as:

`Hybrid Score = (Content Weight × Content Score) + (Collaborative Weight × Collaborative Score)`

This combines textual relevance with patterns in user behaviour.

## Dataset

**CI&T Deskdrop – Articles Sharing and Reading Dataset**

The dataset contains article metadata and user interaction records.

Source: Kaggle – CI&T Deskdrop Dataset.

### Processed Dataset Statistics

| Attribute | Count |
|---|---:|
| Articles | 3,047 |
| User interactions | 72,269 |
| Unique users | 1,895 |
| Unique interacted articles | 2,979 |

Interaction types include views, likes, bookmarks, comments, and follows.

The processed datasets are stored in `data/processed/`.

If the original dataset is not included, download it from Kaggle and follow the preprocessing instructions before running the models.

## Model Evaluation

The recommendation approaches were evaluated using a train-test split and ranking metrics.

| Metric | Content-Based | Collaborative | Hybrid |
|---|---:|---:|---:|
| Precision@10 | 0.0100 | 0.0160 | 0.0130 |
| Recall@10 | 0.0363 | 0.0511 | 0.0446 |
| F1@10 | 0.0125 | 0.0199 | 0.0169 |
| MAP@10 | 0.0173 | 0.0194 | 0.0229 |
| NDCG@10 | 0.0264 | 0.0342 | 0.0349 |

The collaborative model achieved the strongest Precision@10, Recall@10, and F1@10 among the evaluated approaches.

The hybrid model achieved the highest MAP@10 and NDCG@10, demonstrating stronger ranking performance in this evaluation.

Note: These are initial benchmark results from the implemented evaluation procedure, not universal performance guarantees.

## Exploratory Data Analysis

The project includes visualizations and reports covering:

- Interaction type distribution.
- Most popular articles.
- Most active users.
- Article popularity distribution.
- Interaction trends over time.
- User engagement patterns.

EDA outputs are stored in `results/eda/`.

## Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| NLP | TF-IDF |
| Similarity | Cosine Similarity |
| Recommendation | Content-Based and Collaborative Filtering |
| Visualization | Matplotlib, Seaborn, Plotly |
| Dashboard | Streamlit |
| Version Control | Git, GitHub |

## Project Structure

```text
RecoMind/
│
├── api/
│   └── app.py
│
├── data/
│   └── processed/
│       ├── cleaned_articles.csv
│       ├── cleaned_interactions.csv
│       └── user_item_interactions.csv
│
├── results/
│   ├── eda/
│   │   ├── interaction_distribution.png
│   │   ├── popular_articles.png
│   │   ├── active_users.png
│   │   ├── article_popularity_distribution.png
│   │   ├── interaction_trends.png
│   │   └── user_engagement.png
│   └── model_evaluation.csv
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── eda.py
│   ├── content_recommender.py
│   ├── collaborative_recommender.py
│   ├── hybrid_recommender.py
│   └── evaluation.py
│
├── .gitignore
├── requirements.txt
├── README.md
└── Project138,139,140,141.ipynb
```

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/shereneagnes/RecoMind.git
cd RecoMind
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the environment

Windows:

```bash
venv\Scripts\activate
```

Linux/macOS:

```bash
source venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Streamlit dashboard

```bash
python -m streamlit run api/app.py
```

The dashboard will open locally in your browser.

### 6. Run individual modules

Data preprocessing:

```bash
python -m src.preprocessing
```

Exploratory data analysis:

```bash
python -m src.eda
```

Model evaluation:

```bash
python -m src.evaluation
```

Ensure the required input datasets are available in `data/processed/` before running modules that depend on them.

## Future Enhancements

- Semantic embeddings using Sentence Transformers.
- Improved cold-start recommendation strategies.
- Recommendation explanations.
- Advanced ranking and reranking methods.
- Recommendation API integration.
- Docker-based deployment.
- Online recommendation feedback and updates.

## Learning Outcomes

- Recommendation system architecture and implementation.
- Natural Language Processing and TF-IDF vectorization.
- Content-Based and Collaborative Filtering.
- Similarity computation and ranking.
- Evaluation of recommendation systems.
- Exploratory data analysis and visualization.
- Building interactive machine learning applications.



---

**RecoMind – Discover content that matters.**
