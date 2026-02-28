import streamlit as st
import joblib
import pandas as pd
import re
from datetime import datetime
import plotly.graph_objects as go

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(page_title="VeriLens AI", layout="wide")

# --------------------------------------------------
# LIGHT COLORFUL UI CSS
# --------------------------------------------------
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}

/* Beautiful pastel gradient background */
.stApp {
    background: linear-gradient(135deg, #f6d365, #fda085, #a1c4fd, #c2e9fb);
    background-size: 400% 400%;
    animation: gradientMove 15s ease infinite;
}

@keyframes gradientMove {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}

/* Main Card */
.main-card {
    background: rgba(255, 255, 255, 0.85);
    backdrop-filter: blur(8px);
    padding: 40px;
    border-radius: 20px;
    box-shadow: 0 8px 25px rgba(0,0,0,0.15);
}

/* Title */
.title {
    text-align: center;
    font-size: 48px;
    font-weight: 700;
    color: #333;
}

.subtitle {
    text-align: center;
    color: #555;
    margin-bottom: 30px;
}

/* Text Area */
.stTextArea textarea {
    border-radius: 15px !important;
    border: 2px solid #ddd !important;
}

/* Button */
.stButton>button {
    background: linear-gradient(45deg, #ff6a00, #ee0979);
    color: white;
    font-weight: 600;
    border-radius: 25px;
    padding: 10px 30px;
    border: none;
}

/* Result Box */
.result-box {
    padding: 25px;
    border-radius: 20px;
    text-align: center;
    font-size: 24px;
    font-weight: bold;
    margin-top: 20px;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = load_model()

# --------------------------------------------------
# CLEAN TEXT
# --------------------------------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z ]", "", text)
    return text

# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.markdown("<div class='title'>VeriLens AI</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Smart & Colorful Fake News Detection System</div>", unsafe_allow_html=True)

# --------------------------------------------------
# MAIN CARD
# --------------------------------------------------
with st.container():
    st.markdown("<div class='main-card'>", unsafe_allow_html=True)

    text = st.text_area("Paste News Article Here", height=250)

    if st.button("Analyze News"):

        if text.strip() == "":
            st.warning("Please enter news content.")
        elif len(text.split()) < 10:
            st.warning("Enter longer content for better accuracy.")
        else:
            cleaned = clean_text(text)

            with st.spinner("Analyzing with VeriLens AI..."):
                prob_fake = model.predict_proba([cleaned])[0][1]
                prob_real = 1 - prob_fake

                if prob_fake > 0.75:
                    label = "FAKE NEWS"
                    bg_color = "#ff4b4b"
                elif prob_fake < 0.25:
                    label = "REAL NEWS"
                    bg_color = "#00c853"
                else:
                    label = "UNCERTAIN"
                    bg_color = "#ffab00"

                confidence = round((max(prob_fake, prob_real)-0.5)*2*100,2)

            st.markdown(f"""
            <div class='result-box' style='background:{bg_color}; color:white;'>
                {label} <br>
                Confidence: {confidence}%
            </div>
            """, unsafe_allow_html=True)

            # Dashboard
            col1, col2 = st.columns(2)

            with col1:
                fig1 = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prob_real*100,
                    title={'text': "Credibility Score"},
                    gauge={'axis': {'range': [0,100]}}
                ))
                st.plotly_chart(fig1, use_container_width=True)

            with col2:
                fig2 = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prob_fake*100,
                    title={'text': "Fake Risk Score"},
                    gauge={'axis': {'range': [0,100]}}
                ))
                st.plotly_chart(fig2, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<br><center>© 2026 VeriLens AI | Developed by Ishita</center>", unsafe_allow_html=True)