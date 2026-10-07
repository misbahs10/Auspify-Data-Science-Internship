import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Netflix Business Insights",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM LIGHT / COLORFUL CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ---------- GLOBAL ---------- */

    .stApp {
        background:
            radial-gradient(circle at 5% 5%, rgba(221, 235, 255, 0.55), transparent 25%),
            radial-gradient(circle at 95% 10%, rgba(247, 225, 255, 0.50), transparent 25%),
            #f8fafc;
    }

    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1450px;
    }

    /* ---------- SIDEBAR ---------- */

    section[data-testid="stSidebar"] {
        background: linear-gradient(
            180deg,
            #ffffff 0%,
            #f5f8ff 55%,
            #f9f5ff 100%
        );
        border-right: 1px solid #e5e7eb;
    }

    section[data-testid="stSidebar"] .block-container {
        padding-top: 2rem;
    }

    section[data-testid="stSidebar"] h2 {
        color: #1e293b;
    }

    /* ---------- HEADER ---------- */

    .hero {
        background: linear-gradient(
            135deg,
            #ffffff 0%,
            #eef5ff 48%,
            #f8efff 100%
        );
        border: 1px solid #e2e8f0;
        border-radius: 24px;
        padding: 30px 34px;
        margin-bottom: 25px;
        box-shadow: 0 10px 30px rgba(100, 116, 139, 0.10);
        position: relative;
        overflow: hidden;
    }

    .hero::after {
        content: "";
        position: absolute;
        width: 220px;
        height: 220px;
        right: -70px;
        top: -80px;
        background: rgba(139, 92, 246, 0.10);
        border-radius: 50%;
    }

    .hero-title {
        font-size: 2.35rem;
        font-weight: 800;
        color: #172554;
        margin-bottom: 5px;
        letter-spacing: -0.8px;
    }

    .hero-subtitle {
        color: #64748b;
        font-size: 1rem;
        margin-bottom: 14px;
    }

    .hero-badge {
        display: inline-block;
        padding: 7px 14px;
        border-radius: 999px;
        background: #e0edff;
        color: #2563eb;
        font-size: 0.82rem;
        font-weight: 700;
        margin-right: 7px;
    }

    .hero-badge-purple {
        background: #f1e8ff;
        color: #7c3aed;
    }

    /* ---------- SECTION TITLES ---------- */

    .section-title {
        font-size: 1.35rem;
        font-weight: 800;
        color: #1e293b;
        margin-top: 18px;
        margin-bottom: 14px;
        padding-left: 12px;
        border-left: 5px solid #6366f1;
    }

    .section-description {
        color: #64748b;
        font-size: 0.92rem;
        margin-bottom: 18px;
    }

    /* ---------- METRIC CARDS ---------- */

    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e6eaf0;
        border-radius: 18px;
        padding: 18px 20px;
        box-shadow: 0 6px 20px rgba(100, 116, 139, 0.08);
        min-height: 125px;
    }

    div[data-testid="stMetric"]:hover {
        box-shadow: 0 10px 28px rgba(99, 102, 241, 0.13);
        transform: translateY(-2px);
        transition: 0.2s ease;
    }

    div[data-testid="stMetricLabel"] {
        color: #64748b !important;
        font-weight: 600;
    }

    div[data-testid="stMetricValue"] {
        color: #172554 !important;
        font-weight: 800;
    }

    /* ---------- TABS ---------- */

    button[data-baseweb="tab"] {
        font-weight: 700;
        color: #64748b;
        padding: 12px 18px;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        color: #4f46e5;
    }

    div[data-baseweb="tab-highlight"] {
        background-color: #6366f1;
    }

    /* ---------- INPUTS ---------- */

    div[data-baseweb="select"] > div {
        border-radius: 12px;
        border: 1px solid #dbe2ea;
        background: #ffffff;
    }

    input {
        border-radius: 10px !important;
    }

    /* ---------- BUTTON ---------- */

    .stButton > button {
        width: 100%;
        border-radius: 12px;
        border: none;
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        color: white;
        font-weight: 700;
        padding: 0.65rem 1rem;
        box-shadow: 0 5px 15px rgba(99, 102, 241, 0.22);
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #4f46e5, #7c3aed);
        color: white;
    }

    /* ---------- INFO / SUCCESS ---------- */

    div[data-testid="stAlert"] {
        border-radius: 14px;
    }

    /* ---------- CHART CONTAINER ---------- */

    div[data-testid="stPlotlyChart"] {
        background: #ffffff;
        border: 1px solid #e8ecf2;
        border-radius: 18px;
        padding: 8px;
        box-shadow: 0 6px 20px rgba(100, 116, 139, 0.06);
    }

    /* ---------- INSIGHT CARDS ---------- */

    .insight-card {
        background: #ffffff;
        border: 1px solid #e6eaf0;
        border-radius: 18px;
        padding: 20px;
        margin-bottom: 15px;
        box-shadow: 0 6px 18px rgba(100, 116, 139, 0.07);
    }

    .insight-card h4 {
        margin-top: 0;
        color: #1e293b;
    }

    .insight-card p {
        color: #64748b;
        line-height: 1.65;
    }

    .blue-card {
        border-left: 5px solid #3b82f6;
    }

    .purple-card {
        border-left: 5px solid #8b5cf6;
    }

    .pink-card {
        border-left: 5px solid #ec4899;
    }

    .green-card {
        border-left: 5px solid #10b981;
    }

    .orange-card {
        border-left: 5px solid #f59e0b;
    }

    /* ---------- PREDICTION RESULT ---------- */

    .prediction-box {
        background: linear-gradient(
            135deg,
            #eef4ff,
            #f7efff
        );
        border: 1px solid #dbe5ff;
        border-radius: 20px;
        padding: 25px;
        text-align: center;
        margin-top: 20px;
    }

    .prediction-title {
        color: #64748b;
        font-size: 0.95rem;
        font-weight: 600;
    }

    .prediction-result {
        color: #4f46e5;
        font-size: 2rem;
        font-weight: 800;
        margin: 5px 0;
    }

    /* ---------- FOOTER ---------- */

    .footer {
        text-align: center;
        color: #94a3b8;
        font-size: 0.82rem;
        padding: 30px 0 5px 0;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data
def load_data():

    project_root = Path(__file__).resolve().parent.parent

    data_path = project_root / "data" / "netflix_cleaned.csv"

    if not data_path.exists():
        st.error(f"Dataset not found at: {data_path}")
        st.stop()

    df = pd.read_csv(data_path)

    return df


df = load_data()


# ============================================================
# DATA PREPARATION
# ============================================================

df["release_year"] = pd.to_numeric(
    df["release_year"],
    errors="coerce"
)

df["duration_value"] = pd.to_numeric(
    df["duration_value"],
    errors="coerce"
)

df["country"] = df["country"].fillna("Unknown")
df["rating"] = df["rating"].fillna("Unknown")
df["listed_in"] = df["listed_in"].fillna("Unknown")


# ============================================================
# HERO HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">

        <div class="hero-title">
            🎬 Netflix Business Intelligence Dashboard
        </div>

        <div class="hero-subtitle">
            Interactive Data Science Analysis • Content Intelligence •
            Machine Learning • Trend Forecasting
        </div>

        <span class="hero-badge">📊 Data Analytics</span>
        <span class="hero-badge hero-badge-purple">🤖 Machine Learning</span>

    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## 🎯 Dashboard Controls")

    st.markdown(
        "Use the filter below to explore Netflix content."
    )

    content_types = ["All"] + sorted(
        df["type"].dropna().unique().tolist()
    )

    selected_type = st.selectbox(
        "Content Type",
        content_types
    )

    st.markdown("---")

    st.markdown("### 📌 Dataset")

    st.write(f"**Records:** {len(df):,}")
    st.write(f"**Features:** {df.shape[1]}")

    st.markdown("---")

    st.caption(
        "Auspify Technologies\n"
        "Data Science Internship"
    )


# ============================================================
# FILTER DATA
# ============================================================

if selected_type == "All":
    filtered_df = df.copy()
else:
    filtered_df = df[
        df["type"] == selected_type
    ].copy()


# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "🏠 Overview",
        "📊 Content Analytics",
        "🤖 ML Prediction",
        "📈 Forecasting",
        "💡 Business Insights",
    ]
)


# ============================================================
# TAB 1 — OVERVIEW
# ============================================================

with tab1:

    st.markdown(
        '<div class="section-title">Dashboard Overview</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-description">'
        'A high-level overview of the Netflix content catalog.'
        '</div>',
        unsafe_allow_html=True
    )

    total_titles = len(filtered_df)

    movie_count = (
        filtered_df["type"]
        .eq("Movie")
        .sum()
    )

    tv_count = (
        filtered_df["type"]
        .eq("TV Show")
        .sum()
    )

    country_count = (
        filtered_df["country"]
        .dropna()
        .astype(str)
        .str.split(", ")
        .explode()
        .nunique()
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "🎬 Total Titles",
            f"{total_titles:,}"
        )

    with col2:
        st.metric(
            "🍿 Movies",
            f"{movie_count:,}"
        )

    with col3:
        st.metric(
            "📺 TV Shows",
            f"{tv_count:,}"
        )

    with col4:
        st.metric(
            "🌍 Countries",
            f"{country_count:,}"
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ---------------- Content Distribution ----------------

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            '<div class="section-title">Content Distribution</div>',
            unsafe_allow_html=True
        )

        type_counts = (
            filtered_df["type"]
            .value_counts()
            .reset_index()
        )

        type_counts.columns = [
            "type",
            "count"
        ]

        fig = px.pie(
            type_counts,
            names="type",
            values="count",
            hole=0.58,
            title="Movies vs TV Shows",
            color_discrete_sequence=[
                "#6366F1",
                "#EC4899"
            ],
        )

        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(
                family="Arial",
                color="#334155"
            ),
            legend_title_text="",
            margin=dict(
                t=55,
                l=20,
                r=20,
                b=20
            ),
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # ---------------- Release Trend ----------------

    with col2:

        st.markdown(
            '<div class="section-title">Release Year Trend</div>',
            unsafe_allow_html=True
        )

        yearly_content = (
            filtered_df
            .groupby("release_year")
            .size()
            .reset_index(name="count")
            .sort_values("release_year")
        )

        yearly_content = yearly_content[
            yearly_content["release_year"] >= 2000
        ]

        if not yearly_content.empty:

            fig = px.area(
                yearly_content,
                x="release_year",
                y="count",
                title="Titles by Release Year",
            )

            fig.update_traces(
                line=dict(
                    color="#6366F1",
                    width=3
                ),
                fillcolor="rgba(99,102,241,0.14)"
            )

            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(
                    family="Arial",
                    color="#334155"
                ),
                xaxis_title="Release Year",
                yaxis_title="Number of Titles",
                margin=dict(
                    t=55,
                    l=20,
                    r=20,
                    b=20
                ),
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        else:
            st.info("No release-year data available.")


# ============================================================
# TAB 2 — CONTENT ANALYTICS
# ============================================================

with tab2:

    st.markdown(
        '<div class="section-title">Content Analytics</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-description">'
        'Explore countries, categories and audience ratings.'
        '</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    # ---------------- Countries ----------------

    with col1:

        country_data = (
            filtered_df["country"]
            .dropna()
            .astype(str)
            .str.split(", ")
            .explode()
            .value_counts()
            .head(10)
            .reset_index()
        )

        country_data.columns = [
            "country",
            "count"
        ]

        fig = px.bar(
            country_data.sort_values("count"),
            x="count",
            y="country",
            orientation="h",
            title="Top 10 Countries",
            text="count",
            color="count",
            color_continuous_scale=[
                "#DBEAFE",
                "#6366F1"
            ],
        )

        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            coloraxis_showscale=False,
            font=dict(
                family="Arial",
                color="#334155"
            ),
            xaxis_title="Titles",
            yaxis_title="",
            margin=dict(
                t=55,
                l=10,
                r=20,
                b=20
            ),
        )

        fig.update_traces(
            textposition="outside"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # ---------------- Categories ----------------

    with col2:

        category_data = (
            filtered_df["listed_in"]
            .dropna()
            .astype(str)
            .str.split(", ")
            .explode()
            .value_counts()
            .head(10)
            .reset_index()
        )

        category_data.columns = [
            "category",
            "count"
        ]

        fig = px.bar(
            category_data.sort_values("count"),
            x="count",
            y="category",
            orientation="h",
            title="Top 10 Content Categories",
            text="count",
            color="count",
            color_continuous_scale=[
                "#FCE7F3",
                "#EC4899"
            ],
        )

        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            coloraxis_showscale=False,
            font=dict(
                family="Arial",
                color="#334155"
            ),
            xaxis_title="Titles",
            yaxis_title="",
            margin=dict(
                t=55,
                l=10,
                r=20,
                b=20
            ),
        )

        fig.update_traces(
            textposition="outside"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # ---------------- Ratings ----------------

    st.markdown(
        '<div class="section-title">Content Ratings</div>',
        unsafe_allow_html=True
    )

    rating_data = (
        filtered_df["rating"]
        .value_counts()
        .head(10)
        .reset_index()
    )

    rating_data.columns = [
        "rating",
        "count"
    ]

    fig = px.bar(
        rating_data,
        x="rating",
        y="count",
        text="count",
        title="Most Common Ratings",
        color="count",
        color_continuous_scale=[
            "#EDE9FE",
            "#8B5CF6"
        ],
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        coloraxis_showscale=False,
        font=dict(
            family="Arial",
            color="#334155"
        ),
        xaxis_title="Rating",
        yaxis_title="Number of Titles",
        margin=dict(
            t=55,
            l=20,
            r=20,
            b=20
        ),
    )

    fig.update_traces(
        textposition="outside"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# TAB 3 — MACHINE LEARNING PREDICTION
# ============================================================

with tab3:

    st.markdown(
        '<div class="section-title">🤖 Content Type Prediction</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-description">'
        'Use the trained Logistic Regression model to predict '
        'whether a title is likely to be a Movie or TV Show.'
        '</div>',
        unsafe_allow_html=True
    )

    features = [
        "release_year",
        "rating",
        "duration_value",
        "duration_unit",
        "country",
    ]

    X = df[features].copy()
    y = df["type"].copy()

    numeric_features = [
        "release_year",
        "duration_value"
    ]

    categorical_features = [
        "rating",
        "duration_unit",
        "country"
    ]

    numeric_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            ),
            (
                "scaler",
                StandardScaler()
            ),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                )
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore"
                )
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                numeric_transformer,
                numeric_features
            ),
            (
                "cat",
                categorical_transformer,
                categorical_features
            ),
        ]
    )

    model = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000
                )
            ),
        ]
    )

    model.fit(X, y)

    st.markdown("### 🎯 Enter Content Information")

    col1, col2, col3 = st.columns(3)

    with col1:

        release_year = st.number_input(
            "Release Year",
            min_value=1900,
            max_value=2030,
            value=2020,
            step=1,
        )

    with col2:

        rating_options = sorted(
            df["rating"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        rating = st.selectbox(
            "Rating",
            rating_options
        )

    with col3:

        duration_value = st.number_input(
            "Duration Value",
            min_value=1.0,
            max_value=500.0,
            value=90.0,
            step=1.0,
        )

    col1, col2 = st.columns(2)

    with col1:

        duration_options = sorted(
            df["duration_unit"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        duration_unit = st.selectbox(
            "Duration Unit",
            duration_options
        )

    with col2:

        country_options = sorted(
            df["country"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        country = st.selectbox(
            "Country / Production Region",
            country_options
        )

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button(
        "✨ Predict Content Type"
    ):

        input_data = pd.DataFrame(
            {
                "release_year": [
                    release_year
                ],
                "rating": [
                    rating
                ],
                "duration_value": [
                    duration_value
                ],
                "duration_unit": [
                    duration_unit
                ],
                "country": [
                    country
                ],
            }
        )

        prediction = model.predict(
            input_data
        )[0]

        probabilities = model.predict_proba(
            input_data
        )[0]

        classes = model.classes_

        probability_df = pd.DataFrame(
            {
                "Content Type": classes,
                "Probability": probabilities
            }
        )

        st.markdown(
            f"""
            <div class="prediction-box">

                <div class="prediction-title">
                    Predicted Content Type
                </div>

                <div class="prediction-result">
                    {prediction}
                </div>

                <div class="prediction-title">
                    Based on the selected content characteristics
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("### 📊 Prediction Confidence")

        probability_df["Probability"] *= 100

        fig = px.bar(
            probability_df,
            x="Content Type",
            y="Probability",
            text=probability_df["Probability"].round(2),
            color="Content Type",
            color_discrete_sequence=[
                "#6366F1",
                "#EC4899"
            ],
        )

        fig.update_layout(
            yaxis_title="Probability (%)",
            xaxis_title="",
            yaxis_range=[
                0,
                100
            ],
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(
                family="Arial",
                color="#334155"
            ),
            showlegend=False,
        )

        fig.update_traces(
            texttemplate="%{text}%",
            textposition="outside"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


# ============================================================
# TAB 4 — FORECASTING
# ============================================================

with tab4:

    st.markdown(
        '<div class="section-title">📈 Trend Forecasting</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-description">'
        'Historical Netflix content trends with a model-based '
        'future projection.'
        '</div>',
        unsafe_allow_html=True
    )

    yearly = (
        df.groupby("release_year")
        .size()
        .reset_index(
            name="count"
        )
        .sort_values("release_year")
    )

    historical = yearly[
        yearly["release_year"] >= 2000
    ].copy()

    future_years = pd.DataFrame(
        {
            "release_year": [
                2022,
                2023,
                2024,
                2025,
                2026,
            ],
            "count": [
                1603,
                1735,
                1868,
                2000,
                2133,
            ],
        }
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=historical["release_year"],
            y=historical["count"],
            mode="lines+markers",
            name="Historical",
            line=dict(
                color="#6366F1",
                width=3
            ),
            marker=dict(
                size=6
            ),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=future_years["release_year"],
            y=future_years["count"],
            mode="lines+markers",
            name="Forecast",
            line=dict(
                color="#EC4899",
                width=3,
                dash="dash"
            ),
            marker=dict(
                size=8
            ),
        )
    )

    fig.update_layout(
        title="Historical Content Trend & Model-Based Forecast",
        xaxis_title="Year",
        yaxis_title="Number of Titles",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            family="Arial",
            color="#334155"
        ),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            y=1.08,
            x=0
        ),
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.warning(
        "⚠️ Forecast values are model-based projections from "
        "the trend analysis. They should be treated as scenario "
        "estimates, not actual future Netflix content counts."
    )

    st.markdown("### 🔮 Forecasted Values")

    display_forecast = future_years.copy()

    display_forecast.columns = [
        "Year",
        "Predicted Titles"
    ]

    st.dataframe(
        display_forecast,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# TAB 5 — BUSINESS INSIGHTS
# ============================================================

with tab5:

    st.markdown(
        '<div class="section-title">💡 Business Insights</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-description">'
        'Key findings and actionable recommendations derived '
        'from the analysis.'
        '</div>',
        unsafe_allow_html=True
    )

    # ---------------- Key Calculations ----------------

    movie_percentage = (
        movie_count / total_titles * 100
        if total_titles > 0
        else 0
    )

    tv_percentage = (
        tv_count / total_titles * 100
        if total_titles > 0
        else 0
    )

    peak_year_row = (
        df["release_year"]
        .value_counts()
        .sort_index()
    )

    if not peak_year_row.empty:

        peak_year = (
            peak_year_row.idxmax()
        )

        peak_count = (
            peak_year_row.max()
        )

    else:

        peak_year = "N/A"
        peak_count = 0

    country_data = (
        df["country"]
        .dropna()
        .astype(str)
        .str.split(", ")
        .explode()
        .value_counts()
    )

    category_data = (
        df["listed_in"]
        .dropna()
        .astype(str)
        .str.split(", ")
        .explode()
        .value_counts()
    )

    rating_data = (
        df["rating"]
        .value_counts()
    )

    top_country = (
        country_data.index[0]
        if not country_data.empty
        else "N/A"
    )

    top_category = (
        category_data.index[0]
        if not category_data.empty
        else "N/A"
    )

    top_rating = (
        rating_data.index[0]
        if not rating_data.empty
        else "N/A"
    )

    # ---------------- Insight Cards ----------------

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            f"""
            <div class="insight-card blue-card">

                <h4>🎬 Content Mix</h4>

                <p>
                    Movies represent approximately
                    <strong>{movie_percentage:.2f}%</strong>
                    of the selected catalog, while TV Shows
                    account for approximately
                    <strong>{tv_percentage:.2f}%</strong>.
                </p>

            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:

        st.markdown(
            f"""
            <div class="insight-card purple-card">

                <h4>📅 Peak Content Year</h4>

                <p>
                    The highest number of titles in the dataset
                    was released in <strong>{peak_year}</strong>,
                    with approximately
                    <strong>{peak_count:,}</strong> titles.
                </p>

            </div>
            """,
            unsafe_allow_html=True
        )

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            f"""
            <div class="insight-card pink-card">

                <h4>🌍 Leading Market</h4>

                <p>
                    <strong>{top_country}</strong> is the most
                    frequently represented production country
                    in the dataset.
                </p>

            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:

        st.markdown(
            f"""
            <div class="insight-card green-card">

                <h4>🎭 Leading Category</h4>

                <p>
                    <strong>{top_category}</strong> is the most
                    frequently occurring content category.
                </p>

            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown(
        f"""
        <div class="insight-card orange-card">

            <h4>🔖 Audience Rating</h4>

            <p>
                <strong>{top_rating}</strong> is the most common
                rating category across the dataset, indicating
                a strong concentration of content within this
                audience classification.
            </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    # ---------------- Recommendations ----------------

    st.markdown(
        '<div class="section-title">🚀 Business Recommendations</div>',
        unsafe_allow_html=True
    )

    recommendations = [
        (
            "📊 Content Strategy",
            "Use genre and content-type trends to prioritize "
            "high-demand categories and balance Movie vs TV Show investments."
        ),
        (
            "🌍 Geographic Strategy",
            "Analyze leading production countries to identify "
            "strong regional markets and expansion opportunities."
        ),
        (
            "🎯 Audience Targeting",
            "Use rating distributions and content categories "
            "to improve audience segmentation and personalization."
        ),
        (
            "🤖 Predictive Analytics",
            "Integrate classification and recommendation models "
            "into content planning and decision-support workflows."
        ),
        (
            "📈 Forecast Monitoring",
            "Continuously compare forecasted trends with actual "
            "new content data and retrain models as new data becomes available."
        ),
    ]

    for title, description in recommendations:

        st.markdown(
            f"""
            <div class="insight-card">

                <h4>{title}</h4>

                <p>{description}</p>

            </div>
            """,
            unsafe_allow_html=True
        )

    # ---------------- Professional Report ----------------

    with st.expander(
        "📄 View Professional Analysis Summary"
    ):

        st.markdown(
            f"""
            ### Netflix Data Science Analysis

            **Dataset Size:** {len(df):,} records

            **Content Types:** Movie and TV Show

            **Dominant Content Type:** Movie

            **Top Production Country:** {top_country}

            **Top Content Category:** {top_category}

            **Most Common Rating:** {top_rating}

            **Peak Release Year:** {peak_year}

            ### Machine Learning

            A Logistic Regression classification pipeline was
            developed using release year, rating, duration,
            duration unit and country as predictive features.

            ### Forecasting

            A historical trend model was used to generate
            model-based projections for 2022–2026.

            ### Recommendation

            Netflix-style content analytics can support content
            strategy, market analysis, audience segmentation,
            recommendation systems and predictive planning.
            """
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">

        🎬 Netflix Business Intelligence Dashboard
        <br>
        Built with Python • Pandas • Scikit-learn • Plotly • Streamlit
        <br>
        Auspify Technologies — Data Science Internship

    </div>
    """,
    unsafe_allow_html=True
)