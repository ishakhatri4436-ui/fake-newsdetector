import streamlit as st
import joblib
import pandas as pd
import re
import plotly.graph_objects as go
from streamlit_lottie import st_lottie
import requests

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="VeriLens AI", layout="wide")

# ---------------- CSS & UI ----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Poppins', sans-serif; }

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

.main-card {
    background: rgba(255,255,255,0.85);
    backdrop-filter: blur(8px);
    padding: 40px;
    border-radius: 20px;
    box-shadow: 0 8px 25px rgba(0,0,0,0.15);
}

.title {text-align:center; font-size:48px; font-weight:700; color:#333;}
.subtitle {text-align:center; color:#555; margin-bottom:30px;}

.stTextArea textarea {border-radius:15px !important; border:2px solid #ddd !important;}
.stButton>button {background:linear-gradient(45deg,#ff6a00,#ee0979); color:white; font-weight:600; border-radius:25px; padding:10px 30px; border:none; transition:0.3s;}
.stButton>button:hover {transform: scale(1.05);}
.result-box {padding:25px; border-radius:20px; text-align:center; font-size:24px; font-weight:bold; margin-top:20px; box-shadow:0 4px 20px rgba(0,0,0,0.2);}
.keyword-box {background: rgba(0,0,0,0.05); padding:15px; border-radius:15px; margin-top:15px; font-weight:600;}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# ---------------- LOTTIE ANIMATION ----------------
def load_lottie(url):
    r = requests.get(url)
    if r.status_code != 200: return None
    return r.json()

lottie = load_lottie("https://assets6.lottiefiles.com/packages/lf20_x62chJ.json")

# ---------------- HEADER ----------------
st.markdown("<div class='title'>VeriLens AI</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Smart & Colorful Fake News Detection System</div>", unsafe_allow_html=True)

if lottie:
    st_lottie(lottie, height=150)

# ---------------- MAIN CARD ----------------
with st.container():
    st.markdown("<div class='main-card'>", unsafe_allow_html=True)

    col1, col2 = st.columns([2,1])

    with col1:
        text = st.text_area("Paste News Article Here", height=250)
        uploaded_file = st.file_uploader("Or Upload .txt/.csv file", type=["txt","csv"])
        if uploaded_file:
            if uploaded_file.type == "text/csv":
                df = pd.read_csv(uploaded_file)
                st.dataframe(df.head())
                text = " ".join(df.iloc[:,0].astype(str).tolist())
            else:
                text = uploaded_file.read().decode("utf-8")

        if st.button("Analyze News"):
            if not text.strip():
                st.warning("Please enter news content.")
            elif len(text.split())<10:
                st.warning("Enter longer content for better accuracy.")
            else:
                cleaned = re.sub(r"http\S+","",text.lower())
                cleaned = re.sub(r"[^a-zA-Z ]","",cleaned)
                X = vectorizer.transform([cleaned])
                pred = model.predict(X)[0]
                prob = model.predict_proba(X)[0]

                if pred==1:
                    label="FAKE NEWS"
                    bg_color="#ff4b4b"
                    confidence=prob[1]*100
                else:
                    label="REAL NEWS"
                    bg_color="#00c853"
                    confidence=prob[0]*100

                st.markdown(f"<div class='result-box' style='background:{bg_color}; color:white;'>{label}<br>Confidence: {confidence:.2f}%</div>", unsafe_allow_html=True)

                # Top keywords
                words = cleaned.split()
                freq = pd.Series(words).value_counts().head(10)
                st.markdown("### 🔑 Top Keywords")
                st.markdown("<div class='keyword-box'>" + ", ".join(freq.index) + "</div>", unsafe_allow_html=True)

                # Gauge charts
                with col2:
                    fig1 = go.Figure(go.Indicator(mode="gauge+number", value=prob[0]*100, title={'text':"Credibility Score"}, gauge={'axis':{'range':[0,100]},'bar':{'color':'#00c853'}}))
                    st.plotly_chart(fig1,use_container_width=True)

                    fig2 = go.Figure(go.Indicator(mode="gauge+number", value=prob[1]*100, title={'text':"Fake Risk Score"}, gauge={'axis':{'range':[0,100]},'bar':{'color':'#ff4b4b'}}))
                    st.plotly_chart(fig2,use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<br><center>© 2026 VeriLens AI | Developed by Isha</center>", unsafe_allow_html=True)
