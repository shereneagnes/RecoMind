# RecoMind – AI-Powered Personalized Content Discovery Engine

RecoMind is a hybrid article recommendation system that combines **Natural Language Processing (NLP)** and user interaction patterns to recommend relevant content.

Built using the CI&T Deskdrop dataset, the project explores content-based filtering, collaborative filtering, and hybrid recommendation techniques to improve personalized content discovery.

##  Key Features

- **Content-Based Filtering:** Recommends articles based on textual similarity.
- **TF-IDF Vectorization:** Extracts meaningful features from article content.
- **Cosine Similarity:** Measures similarity between article representations.
- **Collaborative Filtering:** Uses user interaction patterns to identify content preferences.
- **Hybrid Recommendation Engine:** Combines content relevance and user preferences.
- **Engagement Analysis:** Analyses views, likes, bookmarks, follows and comments.
- **Recommendation Evaluation:** Evaluates recommendation quality using Precision@K and Recall@K.
- **Interactive Dashboard:** Provides an interface for exploring recommended articles.

##  Recommendation Pipeline

RecoMind uses two complementary recommendation approaches.

### 1. Content-Based Filtering

Article descriptions and metadata are transformed into numerical representations using TF-IDF.

Cosine similarity is then used to identify articles with similar textual characteristics.

### 2. Collaborative Filtering

User interaction data is analysed to identify relationships between users and articles.

This helps discover content preferences based on collective user behaviour.

### 3. Hybrid Recommendation

The content-based and collaborative recommendation scores are combined to generate a ranked list of relevant articles.

The hybrid approach aims to address limitations of relying exclusively on article text or historical interactions.

##  Dataset

**CI&T Deskdrop – Articles Sharing and Reading Dataset**

Source: [Kaggle Dataset](https://www.kaggle.com/datasets/gspmoreira/articles-sharing-reading-from-cit-deskdrop)

| Dataset | Description |
|---|---|
| shared_articles.csv | Article metadata and textual content |
| users_interactions.csv | User engagement and interaction records |

Original project dataset statistics:

- 3,047 shared articles
- 72,312 user interaction events

##  Technology Stack

| Category | Technologies |
|---|---|
| Language | Python |
| Data Processing | Pandas, NumPy |
| NLP | Scikit-learn, TF-IDF |
| Recommendation | Content-Based and Collaborative Filtering |
| Similarity | Cosine Similarity |
| Backend | Flask, REST API |
| Frontend | Streamlit |
| Visualization | Matplotlib, Plotly |

##  Project Structure

```text
RecoMind/
│
├── data/
│   └── README.md
│
├── notebooks/
│   └── recommendation_analysis.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── content_recommender.py
│   ├── collaborative_recommender.py
│   ├── hybrid_recommender.py
│   └── evaluation.py
│
├── api/
│   └── app.py
│
├── dashboard/
│   └── streamlit_app.py
│
├── tests/
│   └── test_recommendations.py
│
├── requirements.txt
├── .gitignore
├── README.md
└── LICENSE
```

##  Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/RecoMind.git
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

### 5. Prepare the dataset

Download the CI&T Deskdrop dataset from Kaggle and place the CSV files in the data directory.

Configure Kaggle API credentials locally if using automated dataset downloads.

### 6. Run the application

The application entry points and execution commands will be provided as the backend and dashboard modules are implemented.

##  Evaluation Metrics

The recommendation engine will be evaluated using:

- Precision@K
- Recall@K
- Cosine Similarity
- Recommendation relevance
- Coverage and diversity

Quantitative results will be added after running the completed implementation.

##  Future Enhancements

- [ ] Semantic embeddings using sentence-transformer models.
- [ ] User-specific recommendation profiles.
- [ ] Advanced ranking and recommendation reranking.
- [ ] Cold-start recommendation strategies.
- [ ] Recommendation explanations.
- [ ] Recommendation API deployment.
- [ ] Docker-based deployment.

##  Learning Outcomes

- Natural Language Processing and text vectorization.
- Recommendation system architectures.
- Content-based and collaborative filtering.
- Similarity computation and ranking.
- REST API development.
- Recommendation evaluation and experimentation.

##  License

This project is licensed under the MIT License.

---

**RecoMind – Discover content that matters.**
