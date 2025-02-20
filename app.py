import streamlit as st
import joblib
import numpy as np
import os

# -------------------- Page Configurations --------------------
st.set_page_config(page_title="AI-Powered Microfinance", page_icon="💰", layout="wide")

# -------------------- Load Models Safely --------------------
def load_model(model_path):
    """Loads a model safely, checking if the file exists."""
    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        st.error(f"❌ Model file not found: {model_path}. Please retrain the model.")
        return None

# Load models
risk_model = load_model("models/credit_risk_model.pkl")
risk_scaler = load_model("models/scaler.pkl")

esg_model = load_model("models/esg_model.pkl")
esg_scaler = load_model("models/esg_scaler.pkl")

sentiment_model = load_model("models/sentiment_classifier.pkl")  # FinBERT model

# -------------------- Header --------------------
st.markdown("""
    <style>
        .big-font { font-size:30px !important; font-weight: bold; color: #2E86C1; text-align: center; }
        .subtext { font-size:18px; text-align: center; color: #566573; }
        .stButton>button { background-color: #2E86C1; color: white; font-size: 16px; font-weight: bold; }
        .stTextInput>div>div>input { background-color: #F2F3F4; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="big-font">🚀 AI-Powered Microfinance Risk Assessment</p>', unsafe_allow_html=True)
st.markdown('<p class="subtext">Leverage AI to predict loan risks, assess ESG scores, and analyze financial sentiments.</p>', unsafe_allow_html=True)

# -------------------- Layout --------------------
tab1, tab2, tab3 = st.tabs(["📊 Loan Risk Prediction", "🌱 ESG Score Assessment", "🗞️ Sentiment Analysis"])

# -------------------- Loan Risk Prediction --------------------
with tab1:
    st.subheader("📊 Loan Default Risk Prediction")

    col1, col2, col3 = st.columns(3)

    with col1:
        int_rate = st.slider("📈 Interest Rate (%)", 0.01, 0.3, 0.1)
        installment = st.number_input("💳 Installment Amount", min_value=0.0, value=100.0)

    with col2:
        log_annual_inc = st.number_input("💰 Log Annual Income", min_value=0.0, value=10.0)
        dti = st.slider("⚖️ Debt-to-Income Ratio", 0, 50, 20)

    with col3:
        fico = st.slider("🏦 FICO Score", 300, 850, 650)

    if st.button("🔍 Predict Loan Risk"):
        if risk_model and risk_scaler:
            input_features = np.array([[int_rate, installment, log_annual_inc, dti, fico]])
            input_scaled = risk_scaler.transform(input_features)

            prediction = risk_model.predict(input_scaled)
            risk_result = "🔴 High Risk" if prediction[0] == 1 else "🟢 Low Risk"
            st.success(f"**Loan Default Risk: {risk_result}**")
        else:
            st.error("❌ Loan Risk Model is not available. Please retrain it.")

# -------------------- ESG Score Prediction --------------------
with tab2:
    st.subheader("🌱 ESG Score Assessment")

    col1, col2 = st.columns(2)

    with col1:
        industry = st.selectbox("🏭 Industry", ["Finance", "Tech", "Energy", "Healthcare", "Retail", "Others"])
        exchange = st.selectbox("📊 Stock Exchange", ["NYSE", "NASDAQ", "LSE", "Other"])

    with col2:
        env = st.slider("🌍 Environmental Score", 0, 100, 50)
        soc = st.slider("🤝 Social Score", 0, 100, 50)
        gov = st.slider("🏛️ Governance Score", 0, 100, 50)

    # Encode categorical variables
    industry_map = {"Finance": 0, "Tech": 1, "Energy": 2, "Healthcare": 3, "Retail": 4, "Others": 5}
    exchange_map = {"NYSE": 0, "NASDAQ": 1, "LSE": 2, "Other": 3}

    industry_encoded = industry_map[industry]
    exchange_encoded = exchange_map[exchange]

    if st.button("🔍 Predict ESG Score"):
        if esg_model and esg_scaler:
            input_features = np.array([[env, soc, gov, industry_encoded, exchange_encoded]])
            input_scaled = esg_scaler.transform(input_features)

            esg_prediction = esg_model.predict(input_scaled)
            categories = ["🟢 Low", "🟡 Medium", "🔴 High"]
            st.success(f"**Predicted ESG Level: {categories[esg_prediction[0]]}**")
        else:
            st.error("❌ ESG Model is not available. Please retrain it.")

# -------------------- Sentiment Analysis --------------------
with tab3:
    st.subheader("🗞️ Financial Sentiment Analysis (Powered by FinBERT)")

    text = st.text_area("✍️ Enter a Financial News Headline, Social Media Post, or Company Statement")

    if st.button("💬 Analyze Sentiment"):
        if sentiment_model:
            if text.strip():
                sentiment_prediction = sentiment_model(text)  # FinBERT returns a list of dicts
                sentiment_label = sentiment_prediction[0]["label"]  # Extracts label

                # FinBERT labels are usually "positive", "neutral", "negative"
                sentiment_mapping = {
                    "positive": "🟢 Positive",
                    "neutral": "🟡 Neutral",
                    "negative": "🔴 Negative"
                }

                if sentiment_label.lower() in sentiment_mapping:
                    st.success(f"💬 **Sentiment: {sentiment_mapping[sentiment_label.lower()]}**")
                else:
                    st.warning("⚠️ Sentiment label not recognized.")
            else:
                st.warning("⚠️ Please enter text for analysis.")
        else:
            st.error("❌ Sentiment Model is not available. Please retrain it.")

# -------------------- Footer --------------------
st.markdown("""
    <br><hr><p style="text-align:center; font-size:14px; color:gray;">
    Built by QuantEdge | 2025
    </p>""", unsafe_allow_html=True)
