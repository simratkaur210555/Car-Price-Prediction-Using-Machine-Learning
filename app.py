import streamlit as st
import pandas as pd
import pickle
import plotly.graph_objects as go
import plotly.express as px

# ----------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Car Price Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# Cached loaders — avoids re-reading the CSV / unpickling the model on every
# interaction (every widget change triggers a full script rerun in Streamlit)
# ----------------------------------------------------------------------------
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_resource
def load_model(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


try:
    car = load_data("Cleaned_Car_data.csv")
    pipe = load_model("LinearRegressionModel.pkl")
except FileNotFoundError as e:
    st.error(
        f"Could not find a required file: **{e.filename}**. "
        "Make sure 'Cleaned_Car_data.csv' and 'LinearRegressionModel.pkl' "
        "are in the same folder as this script."
    )
    st.stop()

# ----------------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Poppins', sans-serif; }

.stApp {
    background: radial-gradient(circle at 20% 20%, #1b2735 0%, #0d1520 45%, #090c12 100%);
    color: #f1f3f6;
}

/* Header */
.main-title {
    font-size: 46px;
    font-weight: 800;
    text-align: center;
    margin-bottom: 0;
    background: linear-gradient(90deg, #ff8a00, #ff2d78, #7a5cff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.subtitle {
    text-align: center;
    font-size: 17px;
    color: #b8c0cc;
    margin-top: 4px;
    margin-bottom: 28px;
}

/* Glass cards */
.card {
    background: rgba(255,255,255,0.06);
    padding: 26px 28px;
    border-radius: 18px;
    border: 1px solid rgba(255,255,255,0.10);
    backdrop-filter: blur(12px);
    box-shadow: 0 8px 28px rgba(0,0,0,0.35);
    margin-bottom: 18px;
}
.card h3 { margin-top: 0; color: #ffb347; }

/* Metric-style result chips */
.result-chip {
    background: linear-gradient(135deg, rgba(255,140,0,0.18), rgba(255,45,120,0.18));
    border: 1px solid rgba(255,255,255,0.15);
    border-radius: 16px;
    padding: 22px;
    text-align: center;
}
.result-chip .label { font-size: 14px; color: #cfd6df; letter-spacing: 1px; text-transform: uppercase; }
.result-chip .value { font-size: 34px; font-weight: 800; color: #ffffff; margin-top: 4px; }

/* Buttons */
.stButton>button {
    background: linear-gradient(90deg,#ff512f,#dd2476);
    color: white;
    border: none;
    padding: 12px 30px;
    border-radius: 10px;
    font-size: 17px;
    font-weight: 600;
    width: 100%;
    box-shadow: 0 4px 18px rgba(221,36,118,0.35);
}
.stButton>button:hover {
    transform: scale(1.02);
    transition: 0.2s ease-in-out;
    box-shadow: 0 6px 24px rgba(221,36,118,0.55);
}

/* Sidebar */
section[data-testid="stSidebar"] { background: #0d1520; border-right: 1px solid rgba(255,255,255,0.06); }

/* Tabs */
.stTabs [data-baseweb="tab"] { font-size: 16px; font-weight: 600; }

hr { border-color: rgba(255,255,255,0.1); }
</style>
""",
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.markdown('<p class="main-title">🚗 AI Car Price Predictor</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Get an instant, data-driven valuation for any used car</p>',
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Sidebar — quick dataset stats, gives the app extra life without cluttering
# the main flow
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 📊 Dataset Snapshot")
    st.metric("Listings", f"{len(car):,}")
    st.metric("Brands", f"{car['company'].nunique()}")
    st.metric("Models", f"{car['name'].nunique()}")
    st.markdown("---")
    st.caption(
        "Predictions come from a Linear Regression model trained on this "
        "dataset. Treat results as an estimate, not an appraisal."
    )

# ----------------------------------------------------------------------------
# Tabs
# ----------------------------------------------------------------------------
tab_predict, tab_insights, tab_about = st.tabs(["🔮 Predict", "📈 Market Insights", "ℹ️ About"])

# ============================== PREDICT TAB =================================
with tab_predict:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 🚘 Car Details")

        company = st.selectbox("Company", sorted(car["company"].unique()))

        # Model list narrows down to the chosen company — small touch that
        # avoids nonsensical company/model combinations reaching the model.
        models_for_company = sorted(car.loc[car["company"] == company, "name"].unique())
        name = st.selectbox("Car Model", models_for_company)

        year = st.slider(
            "Manufacturing Year",
            int(car["year"].min()),
            int(car["year"].max()),
            2015,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 🛣️ Usage Details")

        kms_driven = st.number_input(
            "Kilometers Driven", min_value=0, max_value=500000, value=30000, step=1000
        )
        fuel_type = st.selectbox("Fuel Type", sorted(car["fuel_type"].unique()))
        st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    predict = st.button("Predict Price 🚀")

    if predict:
        input_df = pd.DataFrame(
            [[name, company, year, kms_driven, fuel_type]],
            columns=["name", "company", "year", "kms_driven", "fuel_type"],
        )

        with st.spinner("Crunching the numbers..."):
            prediction = float(pipe.predict(input_df)[0])
            prediction = max(prediction, 0)

        st.write("")
        res1, res2, res3 = st.columns(3)
        with res1:
            st.markdown(
                f'<div class="result-chip"><div class="label">Estimated Price</div>'
                f'<div class="value">₹ {int(prediction):,}</div></div>',
                unsafe_allow_html=True,
            )
        with res2:
            st.markdown(
                f'<div class="result-chip"><div class="label">Fair Range (±10%)</div>'
                f'<div class="value">₹ {int(prediction*0.9):,} – {int(prediction*1.1):,}</div></div>',
                unsafe_allow_html=True,
            )
        with res3:
            similar = car[(car["company"] == company) & (car["fuel_type"] == fuel_type)]
            avg_similar = similar["Price"].mean() if "Price" in car.columns and len(similar) else None
            display_val = f"₹ {int(avg_similar):,}" if avg_similar else "N/A"
            st.markdown(
                f'<div class="result-chip"><div class="label">Similar Listings Avg</div>'
                f'<div class="value">{display_val}</div></div>',
                unsafe_allow_html=True,
            )

        st.write("")

        gauge_max = max(1_000_000, int(prediction * 1.5))
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=prediction,
                number={"prefix": "₹ ", "valueformat": ",.0f"},
                title={"text": "Predicted Price"},
                gauge={
                    "axis": {"range": [0, gauge_max]},
                    "bar": {"color": "#ff512f"},
                    "bgcolor": "rgba(255,255,255,0.05)",
                    "steps": [
                        {"range": [0, gauge_max * 0.33], "color": "rgba(255,255,255,0.05)"},
                        {"range": [gauge_max * 0.33, gauge_max * 0.66], "color": "rgba(255,255,255,0.10)"},
                        {"range": [gauge_max * 0.66, gauge_max], "color": "rgba(255,255,255,0.15)"},
                    ],
                },
            )
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font={"color": "#f1f3f6"},
            height=320,
            margin=dict(l=20, r=20, t=60, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

# ============================== INSIGHTS TAB =================================
with tab_insights:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Average Price by Company")
    if "Price" in car.columns:
        avg_price = (
            car.groupby("company")["Price"].mean().sort_values(ascending=False).head(15).reset_index()
        )
        fig_bar = px.bar(
            avg_price, x="company", y="Price", color="Price", color_continuous_scale="Sunsetdark"
        )
        fig_bar.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#f1f3f6"},
            xaxis_title="", yaxis_title="Avg Price (₹)",
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("No 'Price' column found in the dataset to chart.")
    st.markdown("</div>", unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Fuel Type Distribution")
        fuel_counts = car["fuel_type"].value_counts().reset_index()
        fuel_counts.columns = ["fuel_type", "count"]
        fig_pie = px.pie(fuel_counts, names="fuel_type", values="count", hole=0.5)
        fig_pie.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", font={"color": "#f1f3f6"}, showlegend=True
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_b:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Listings by Year")
        year_counts = car["year"].value_counts().sort_index().reset_index()
        year_counts.columns = ["year", "count"]
        fig_line = px.line(year_counts, x="year", y="count", markers=True)
        fig_line.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#f1f3f6"},
        )
        st.plotly_chart(fig_line, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

# ============================== ABOUT TAB =================================
with tab_about:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Model Information")
    st.write(
        """
**Algorithm:** Linear Regression

**Features used:**
- Car Model
- Company
- Manufacturing Year
- Kilometers Driven
- Fuel Type

**Notes:** Predictions are estimates based on historical listing data and
should be used as a starting reference point, not a final valuation.
        """
    )
    st.markdown("</div>", unsafe_allow_html=True)