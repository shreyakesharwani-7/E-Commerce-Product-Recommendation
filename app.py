import streamlit as st
import pandas as pd
import pickle
import os

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="ShopSense | AI Recommendations",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: #0b0f19;
    }

    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background: #111827 !important;
        border-right: 1px solid #263043;
    }

    /* Custom Cards */
    .hero-card {
        padding: 28px;
        border-radius: 18px;
        background: linear-gradient(135deg, #171d2d 0%, #111827 50%, #172033 100%);
        border: 1px solid #293449;
        margin-bottom: 20px;
    }

    .hero-title {
        font-size: 36px;
        font-weight: 800;
        letter-spacing: -0.5px;
        margin-bottom: 6px;
        color: #ffffff;
    }

    .hero-subtitle {
        font-size: 15px;
        color: #9ca3af;
        line-height: 1.5;
    }

    .badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        background: #1e293b;
        border: 1px solid #334155;
        color: #a5b4fc;
        font-size: 12px;
        font-weight: 600;
        margin-bottom: 12px;
    }

    /* KPI Metrics */
    .kpi-card {
        background: #111827;
        border: 1px solid #263043;
        border-radius: 14px;
        padding: 18px;
        margin-bottom: 15px;
    }

    .kpi-label {
        color: #9ca3af;
        font-size: 13px;
        margin-bottom: 6px;
    }

    .kpi-value {
        color: #ffffff;
        font-size: 26px;
        font-weight: 700;
    }

    .kpi-subtext {
        color: #64748b;
        font-size: 11px;
        margin-top: 4px;
    }

    /* Tab Styling Overrides */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1px solid #263043;
    }

    .stTabs [data-baseweb="tab"] {
        height: 45px;
        white-space: pre-wrap;
        background-color: #111827;
        border-radius: 8px 8px 0px 0px;
        color: #9ca3af;
        border: 1px solid #263043;
        border-bottom: none;
    }

    .stTabs [aria-selected="true"] {
        background-color: #1e293b !important;
        color: #a5b4fc !important;
        border-color: #6366f1 !important;
    }
</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD MODEL FILES (WITH CACHING FOR SPEED)
# =========================================================

@st.cache_resource
def load_model_assets():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_DIR = os.path.join(BASE_DIR, "models")

    with open(os.path.join(MODEL_DIR, "customer_factors.pkl"), "rb") as f:
        customer_factors = pickle.load(f)

    with open(os.path.join(MODEL_DIR, "product_factors.pkl"), "rb") as f:
        product_factors = pickle.load(f)

    with open(os.path.join(MODEL_DIR, "train_matrix.pkl"), "rb") as f:
        train_matrix = pickle.load(f)

    with open(os.path.join(MODEL_DIR, "product_info.pkl"), "rb") as f:
        product_info = pickle.load(f)

    predicted_scores = customer_factors @ product_factors

    return customer_factors, product_factors, train_matrix, product_info, predicted_scores


try:
    customer_factors, product_factors, train_matrix, product_info, predicted_scores = load_model_assets()
except Exception as e:
    st.error(f"Error loading model files from `/models` directory: {e}")
    st.stop()


# =========================================================
# RECOMMENDATION FUNCTION
# =========================================================

def get_recommendations(customer_id, top_n=10):
    customer_index = train_matrix.index.get_loc(customer_id)
    scores = predicted_scores[customer_index]

    recommendation_scores = pd.Series(scores, index=train_matrix.columns)
    purchased_products = train_matrix.loc[customer_id]

    # Exclude already purchased items
    recommendation_scores[purchased_products.values == 1] = -1

    recommendations = (
        recommendation_scores
        .sort_values(ascending=False)
        .head(top_n)
        .reset_index()
    )

    recommendations.columns = ["StockCode", "Score"]
    recommendations["StockCode"] = recommendations["StockCode"].astype(str)

    recommendations = recommendations.merge(
        product_info,
        on="StockCode",
        how="left"
    )

    recommendations = recommendations[["StockCode", "Description", "Score"]]
    return recommendations


# =========================================================
# SIDEBAR CONTROLS
# =========================================================

with st.sidebar:
    st.title("🛍️ ShopSense")
    st.caption("AI-Powered E-Commerce Recommendations")
    st.divider()

    st.subheader("🎯 Selection Controls")
    customer_ids = train_matrix.index.tolist()
    customer_id = st.selectbox("Select Customer ID", customer_ids)

    top_n = st.slider("Max Recommendations", min_value=4, max_value=20, value=8, step=2)

    st.divider()
    st.subheader("🔍 Real-time Filters")
    search_query = st.text_input("Filter products by name", "").strip().lower()
    min_match = st.slider("Minimum Match Percentage", 0, 100, 0, step=5)

    st.divider()
    st.caption("🧠 Model Architecture: SVD Matrix Factorization")


# =========================================================
# HERO SECTION
# =========================================================

st.markdown(
    """
    <div class="hero-card">
        <div class="badge">✦ MACHINE LEARNING RECOMMENDATION ENGINE</div>
        <div class="hero-title">Personalized Shopping Experience</div>
        <div class="hero-subtitle">
            Generating real-time product suggestions powered by collaborative filtering and matrix factorization.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# =========================================================
# KPI METRICS SECTION
# =========================================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(
        f"""<div class="kpi-card">
            <div class="kpi-label">👥 Total Customers</div>
            <div class="kpi-value">{len(train_matrix):,}</div>
            <div class="kpi-subtext">Active database profiles</div>
        </div>""",
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        f"""<div class="kpi-card">
            <div class="kpi-label">📦 Catalog Size</div>
            <div class="kpi-value">{len(train_matrix.columns):,}</div>
            <div class="kpi-subtext">Unique available SKUs</div>
        </div>""",
        unsafe_allow_html=True
    )

with col3:
    purchased_count = int(train_matrix.loc[customer_id].sum())
    st.markdown(
        f"""<div class="kpi-card">
            <div class="kpi-label">🛒 User Purchases</div>
            <div class="kpi-value">{purchased_count}</div>
            <div class="kpi-subtext">Historical orders for {customer_id}</div>
        </div>""",
        unsafe_allow_html=True
    )

with col4:
    st.markdown(
        """<div class="kpi-card">
            <div class="kpi-label">⚡ Engine Status</div>
            <div class="kpi-value" style="color: #4ade80;">Active</div>
            <div class="kpi-subtext">SVD Latent Factor Model</div>
        </div>""",
        unsafe_allow_html=True
    )

# =========================================================
# WORKSPACE TABS
# =========================================================

tab1, tab2, tab3 = st.tabs(["✨ AI Recommendations", "📜 Purchase History", "🧠 Engine Details"])

# ---------------------------------------------------------
# TAB 1: RECOMMENDATIONS
# ---------------------------------------------------------
with tab1:
    recommendations = get_recommendations(customer_id, top_n)

    # Compute percentage match scaled to the highest score
    max_score = recommendations["Score"].max()
    if max_score > 0:
        recommendations["Match"] = ((recommendations["Score"] / max_score) * 100).astype(int)
    else:
        recommendations["Match"] = 0

    # Apply search & score filters
    filtered_recs = recommendations[recommendations["Match"] >= min_match].copy()
    if search_query:
        filtered_recs = filtered_recs[
            filtered_recs["Description"].fillna("").str.lower().str.contains(search_query)
        ]

    header_col, export_col = st.columns([3, 1])
    with header_col:
        st.subheader(f"Top Suggestions for Customer #{customer_id}")
    with export_col:
        csv_data = filtered_recs.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Export Recommendations",
            data=csv_data,
            file_name=f"recommendations_{customer_id}.csv",
            mime="text/csv",
            use_container_width=True
        )

    if filtered_recs.empty:
        st.warning("No products match your filter criteria. Try adjusting the sidebar sliders/search.")
    else:
        # Display Grid Layout
        for start in range(0, len(filtered_recs), 2):
            cols = st.columns(2)
            for position, col in enumerate(cols):
                index = start + position
                if index >= len(filtered_recs):
                    break

                row = filtered_recs.iloc[index]
                description = row["Description"] if pd.notna(row["Description"]) else "Product description unavailable"
                score = float(row["Score"])
                match = int(row["Match"])

                with col:
                    st.markdown(
                        f"""
                        <div style="
                            background: #111827;
                            border: 1px solid #263043;
                            border-radius: 14px;
                            padding: 20px;
                            margin-bottom: 16px;
                        ">
                            <div style="color: #818cf8; font-size: 12px; font-weight: 700;">RANK #{index + 1}</div>
                            <div style="color: #ffffff; font-size: 18px; font-weight: 700; margin: 8px 0; min-height: 48px;">{description}</div>
                            <div style="color: #94a3b8; font-size: 12px;">Stock Code: <b>{row["StockCode"]}</b></div>
                            <div style="margin-top: 14px; color: #a5b4fc; font-size: 14px; font-weight: 700;">
                                ⭐ AI Relevance Match: {match}%
                            </div>
                            <div style="background: #273449; height: 6px; border-radius: 10px; margin-top: 6px;">
                                <div style="background: linear-gradient(90deg, #6366f1, #8b5cf6); width: {match}%; height: 6px; border-radius: 10px;"></div>
                            </div>
                            <div style="color: #64748b; font-size: 11px; margin-top: 8px;">Raw SVD Score: {score:.4f}</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

# ---------------------------------------------------------
# TAB 2: PURCHASE HISTORY
# ---------------------------------------------------------
with tab2:
    st.subheader(f"Historical Interactions for Customer #{customer_id}")
    
    # Extract items already bought by this user
    purchased_mask = train_matrix.loc[customer_id] == 1
    purchased_codes = train_matrix.columns[purchased_mask].astype(str)

    if len(purchased_codes) == 0:
        st.info("No recorded prior purchases for this profile.")
    else:
        history_df = product_info[product_info["StockCode"].isin(purchased_codes)].reset_index(drop=True)
        st.dataframe(
            history_df,
            column_config={
                "StockCode": "Product SKU",
                "Description": "Product Name"
            },
            use_container_width=True,
            hide_index=True
        )

# ---------------------------------------------------------
# TAB 3: ENGINE DETAILS
# ---------------------------------------------------------
with tab3:
    st.subheader("How the SVD Model Operates")
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("""
        #### Matrix Factorization Overview
        The recommendation engine maps both **customers** and **products** into a joint latent factor space:
        
        1. **User Vector Space:** Capture specific buying behaviors and product preferences.
        2. **Product Vector Space:** Quantifies product similarity based on co-occurrence in purchase histories.
        3. **Dot Product Scoring:** Multiplying user and product factor vectors generates predicted affinity scores.
        """)

    with col_b:
        st.markdown("""
        #### Model Performance & Design
        - **Algorithm:** Singular Value Decomposition (SVD)
        - **Cold-Start Handling:** Excludes previously purchased items automatically (`Score = -1`).
        - **Ranking Metric:** Normalizes scores dynamically to represent relative match confidence.
        """)

# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <hr style="border: none; border-top: 1px solid #263043; margin-top: 40px;">
    <div style="text-align: center; color: #64748b; font-size: 12px; padding-bottom: 20px;">
        ShopSense Engine • Powered by Streamlit, Pandas, and Scikit-learn
    </div>
    """,
    unsafe_allow_html=True
)