
from pathlib import Path
import sys

import streamlit as st
import pandas as pd
import plotly.express as px


# =========================================================
# PROJECT PATHS AND IMPORTS
# =========================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.hybrid_recommender import HybridRecommender


DATA_DIR = ROOT_DIR / "data" / "processed"
RESULTS_DIR = ROOT_DIR / "results"
EDA_DIR = RESULTS_DIR / "eda"

INTERACTIONS_PATH = DATA_DIR / "cleaned_interactions.csv"
ARTICLES_PATH = DATA_DIR / "cleaned_articles.csv"
EVALUATION_PATH = RESULTS_DIR / "model_evaluation.csv"


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="RecoMind | AI Recommendations",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.stApp {
    background-color: #0E1117;
}

[data-testid="stSidebar"] {
    background-color: #151A23;
}

.main-title {
    font-size: 44px;
    font-weight: 800;
    color: #FFFFFF;
    margin-bottom: 0;
}

.subtitle {
    color: #A5AAB5;
    font-size: 16px;
    margin-bottom: 25px;
}

.article-card {
    background-color: #1B2230;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #30394A;
    margin-bottom: 10px;
}

.article-title {
    color: #FFFFFF;
    font-size: 19px;
    font-weight: 700;
}

.small-text {
    color: #A5AAB5;
    font-size: 13px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD DATASETS
# =========================================================

@st.cache_data
def load_datasets():

    interactions = pd.read_csv(INTERACTIONS_PATH)
    articles = pd.read_csv(ARTICLES_PATH)

    return interactions, articles


# =========================================================
# LOAD RECOMMENDATION MODEL
# =========================================================

@st.cache_resource(show_spinner=False)
def load_recommender(content_weight):

    model = HybridRecommender(
        content_weight=content_weight
    )

    model.build_model()

    return model


# =========================================================
# LOAD EVALUATION RESULTS
# =========================================================

@st.cache_data
def load_evaluation():

    if not EVALUATION_PATH.exists():
        return pd.DataFrame()

    return pd.read_csv(EVALUATION_PATH)


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.markdown("# 🧠 RecoMind")

st.sidebar.caption(
    "AI-Powered Hybrid Recommendation System"
)

st.sidebar.divider()

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Overview",
        "EDA Analytics",
        "Model Performance",
        "Recommendations"
    ]
)

st.sidebar.divider()

st.sidebar.markdown("### About RecoMind")

st.sidebar.caption("""
RecoMind combines Content-Based Filtering
and Collaborative Filtering to generate
personalized article recommendations.
""")

st.sidebar.divider()

st.sidebar.caption(
    "Built with Python | Scikit-learn | Streamlit"
)


# =========================================================
# PAGE 1: OVERVIEW
# =========================================================

if page == "Overview":

    st.markdown(
        '<div class="main-title">🧠 RecoMind</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Discover content tailored to your interests '
        'through intelligent hybrid recommendations.'
        '</div>',
        unsafe_allow_html=True
    )

    interactions, articles = load_datasets()

    total_users = interactions["personId"].nunique()
    total_articles = articles["contentId"].nunique()
    total_interactions = len(interactions)

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Users", f"{total_users:,}")
    col2.metric("Total Articles", f"{total_articles:,}")
    col3.metric("Interactions", f"{total_interactions:,}")
    col4.metric("Recommendation Models", "3")

    st.divider()

    st.subheader("About the Project")

    st.write("""
    RecoMind is an intelligent article recommendation
    platform designed to provide personalized content
    recommendations based on user interests and
    interaction history.

    **Content-Based Filtering**

    Uses TF-IDF vectorization and cosine similarity
    to identify articles with similar textual content.

    **Collaborative Filtering**

    Uses user-item interactions and item similarity
    to identify articles preferred by users with
    similar interests.

    **Hybrid Recommendation**

    Combines both recommendation scores using a
    weighted scoring mechanism to generate
    personalized results.
    """)

    st.divider()

    st.subheader("System Architecture")

    st.code("""
        User Interaction Data
                |
                v
        Data Preprocessing
                |
          +-----+------+
          |            |
          v            v
      Content-Based  Collaborative
       Filtering      Filtering
          |            |
          +-----+------+
                |
                v
        Hybrid Recommendation
                |
                v
        Personalized Articles
                |
                v
          Streamlit Dashboard
    """)

    st.subheader("Interaction Distribution")

    event_counts = interactions["eventType"].value_counts()

    fig = px.pie(
        values=event_counts.values,
        names=event_counts.index,
        hole=0.55,
        title="User Interaction Breakdown"
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0E1117",
        plot_bgcolor="#0E1117"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# =========================================================
# PAGE 2: EDA ANALYTICS
# =========================================================

elif page == "EDA Analytics":

    st.title("📊 Exploratory Data Analysis")

    st.caption(
        "Explore user interactions, article popularity "
        "and engagement patterns."
    )

    interactions, articles = load_datasets()

    st.subheader("Dataset Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Interactions",
        f"{len(interactions):,}"
    )

    col2.metric(
        "Unique Users",
        f"{interactions['personId'].nunique():,}"
    )

    col3.metric(
        "Unique Articles",
        f"{articles['contentId'].nunique():,}"
    )

    st.divider()

    st.subheader("1. Interaction Type Distribution")

    image_path = EDA_DIR / "interaction_distribution.png"

    if image_path.exists():
        st.image(str(image_path), use_container_width=True)

    st.subheader("2. Most Popular Articles")

    popular_path = EDA_DIR / "popular_articles.csv"

    if popular_path.exists():

        popular = pd.read_csv(popular_path)

        st.dataframe(
            popular,
            use_container_width=True,
            hide_index=True
        )

    image_path = EDA_DIR / "popular_articles.png"

    if image_path.exists():
        st.image(str(image_path), use_container_width=True)

    st.subheader("3. Most Active Users")

    active_path = EDA_DIR / "active_users.csv"

    if active_path.exists():

        active = pd.read_csv(active_path)

        st.dataframe(
            active,
            use_container_width=True,
            hide_index=True
        )

    image_path = EDA_DIR / "active_users.png"

    if image_path.exists():
        st.image(str(image_path), use_container_width=True)

    st.subheader("4. Article Popularity Distribution")

    image_path = EDA_DIR / "article_popularity_distribution.png"

    if image_path.exists():
        st.image(str(image_path), use_container_width=True)

    st.subheader("5. Interaction Trends Over Time")

    image_path = EDA_DIR / "interaction_trends.png"

    if image_path.exists():
        st.image(str(image_path), use_container_width=True)

    st.subheader("6. User Engagement Patterns")

    image_path = EDA_DIR / "user_engagement.png"

    if image_path.exists():
        st.image(str(image_path), use_container_width=True)


# =========================================================
# PAGE 3: MODEL PERFORMANCE
# =========================================================

elif page == "Model Performance":

    st.title("📈 Model Performance")

    st.caption(
        "Compare Content-Based, Collaborative "
        "and Hybrid Recommendation Models."
    )

    evaluation = load_evaluation()

    if evaluation.empty:

        st.warning(
            "Evaluation results not found. "
            "Run python -m src.evaluation first."
        )

    else:

        if "Unnamed: 0" in evaluation.columns:

            evaluation = evaluation.rename(
                columns={"Unnamed: 0": "Model"}
            )

        elif "Model" not in evaluation.columns:

            evaluation = evaluation.rename(
                columns={evaluation.columns[0]: "Model"}
            )

        st.subheader("Evaluation Metrics")

        st.dataframe(
            evaluation,
            use_container_width=True,
            hide_index=True
        )

        st.divider()

        numeric_columns = [
            column for column in evaluation.columns
            if column != "Model"
            and pd.api.types.is_numeric_dtype(evaluation[column])
        ]

        if numeric_columns:

            metric = st.selectbox(
                "Select Evaluation Metric",
                numeric_columns
            )

            fig = px.bar(
                evaluation,
                x="Model",
                y=metric,
                color="Model",
                title=f"{metric} Comparison",
                text_auto=".4f"
            )

            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0E1117",
                plot_bgcolor="#0E1117",
                showlegend=False
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            st.subheader("All Metrics Comparison")

            melted = evaluation.melt(
                id_vars="Model",
                value_vars=numeric_columns,
                var_name="Metric",
                value_name="Score"
            )

            fig2 = px.bar(
                melted,
                x="Metric",
                y="Score",
                color="Model",
                barmode="group",
                title="Recommendation Model Comparison"
            )

            fig2.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0E1117",
                plot_bgcolor="#0E1117"
            )

            st.plotly_chart(
                fig2,
                use_container_width=True
            )


# =========================================================
# PAGE 4: PERSONALIZED RECOMMENDATIONS
# =========================================================

elif page == "Recommendations":

    st.title("✨ Personalized Recommendations")

    st.caption(
        "Discover articles selected using your interaction history."
    )

    st.divider()

    st.subheader("Recommendation Settings")

    content_weight = st.slider(
        "Content-Based Filtering Weight",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05
    )

    collaborative_weight = 1 - content_weight

    col1, col2 = st.columns(2)

    col1.metric(
        "Content-Based Weight",
        f"{content_weight:.0%}"
    )

    col2.metric(
        "Collaborative Weight",
        f"{collaborative_weight:.0%}"
    )

    top_n = st.select_slider(
        "Number of Recommendations",
        options=[5, 10, 15, 20],
        value=10
    )

    st.divider()

    st.subheader("Enter User ID")

    user_id_input = st.text_input(
        "User ID",
        placeholder="Example: 3609194402293569455"
    )

    if st.button(
        "Generate Recommendations",
        type="primary",
        use_container_width=True
    ):

        if not user_id_input.strip():

            st.error("Please enter a valid User ID.")

        else:

            try:

                person_id = int(user_id_input.strip())

                with st.spinner(
                    "Building models and generating recommendations..."
                ):

                    model = load_recommender(content_weight)

                    if person_id not in model.collaborative_model.user_map:

                        st.error(
                            "User ID not found. "
                            "Please enter an existing user ID."
                        )

                        st.session_state.pop(
                            "recommendations",
                            None
                        )

                    else:

                        recommendations = model.recommend(
                            person_id=person_id,
                            top_n=top_n
                        )

                        st.session_state["recommendations"] = (
                            recommendations
                        )

                        st.session_state["recommended_user"] = (
                            person_id
                        )

            except ValueError as error:

                st.error(str(error))

            except Exception as error:

                st.error(f"Error: {error}")

    # Display recommendations

    if "recommendations" in st.session_state:

        recommendations = st.session_state["recommendations"]

        st.divider()

        st.subheader(
            f"Recommendations for User "
            f"{st.session_state['recommended_user']}"
        )

        if recommendations.empty:

            st.info(
                "No new recommendations available for this user."
            )

        else:

            st.success(
                f"{len(recommendations)} personalized articles generated!"
            )

            for index, article in recommendations.iterrows():

                st.markdown(
                    f"""
                    <div class="article-card">
                        <div class="article-title">
                            {index + 1}. {article['title']}
                        </div>
                        <br>
                        <div class="small-text">
                            Content Score:
                            {article['content_score']:.4f}
                            &nbsp; | &nbsp;
                            Collaborative Score:
                            {article['collaborative_score']:.4f}
                            &nbsp; | &nbsp;
                            Hybrid Score:
                            {article['hybrid_score']:.4f}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                url = article["url"]

                if pd.notna(url) and str(url).startswith(
                    ("http://", "https://")
                ):

                    st.markdown(f"[Read Article]({url})")

                st.write("")

            st.download_button(
                "Download Recommendations as CSV",
                data=recommendations.to_csv(
                    index=False
                ).encode("utf-8"),
                file_name="recomind_recommendations.csv",
                mime="text/csv"
            )


# =========================================================
# FOOTER
# =========================================================

st.sidebar.divider()

st.sidebar.caption(
    "RecoMind | Intelligent Content Discovery"
)
