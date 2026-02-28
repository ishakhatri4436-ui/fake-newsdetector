import streamlit as st
import joblib
import pandas as pd
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from textblob import TextBlob # New: For sentiment

# 1. PAGE SETUP
st.set_page_config(page_title="VeriLens AI Ultra | Isha", layout="wide", page_icon="🛡️")

# 2. DESIGNER CSS (Modern Dark Mode)
st.markdown("""
<style>
    .stApp { background-color: #0b0e14; color: white; }
    .main-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(20px);
        border-radius: 25px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 40px;
    }
    .stMetric { background: rgba(0,0,0,0.2); padding: 15px; border-radius: 10px; border: 1px solid #4facfe; }
</style>
""", unsafe_allow_html=True)

# 3. LOAD MODEL
@st.cache_resource
def load_model():
    return joblib.load('model.pkl')

model = load_model()

# 4. APP HEADER
st.markdown("<h1 style='text-align: center; color: #00d2ff;'>🛡️ VERILENS <span style='color: white;'>ULTRA</span></h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: grey;'>Advanced Forensic Analysis Suite by Isha</p>", unsafe_allow_html=True)

# 5. INPUT SECTION
with st.container():
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    input_text = st.text_area("📄 Paste Article or URL Content", placeholder="Enter news text...", height=180)
    
    if st.button("🚀 RUN FULL FORENSIC SCAN"):
        if input_text:
            # A. PREDICTION LOGIC
            probs = model.predict_proba([input_text])[0]
            p_fake, p_real = probs[0] * 100, probs[1] * 100
            
            # B. SENTIMENT ANALYSIS
            analysis = TextBlob(input_text)
            sentiment_score = (analysis.sentiment.polarity + 1) * 50 # Convert -1 to 1 into 0 to 100
            
            # C. DISPLAY RESULTS
            st.markdown("### 🔍 Scan Overview")
            m1, m2, m3 = st.columns(3)
            m1.metric("Credibility Score", f"{p_real:.1f}%", f"{'High' if p_real > 70 else 'Low'}")
            m2.metric("Emotional Bias", f"{sentiment_score:.1f}%", "- Neutral" if 40 < sentiment_score < 60 else "Highly Emotional", delta_color="inverse")
            m3.metric("Linguistic Complexity", "Professional" if len(input_text.split()) > 100 else "Informal")

            # D. VISUAL CHARTS
            st.markdown("---")
            col_chart, col_cloud = st.columns(2)
            
            with col_chart:
                fig = go.Figure(go.Bar(
                    x=['Authentic Patterns', 'Deceptive Patterns', 'Emotional Bias'],
                    y=[p_real, p_fake, sentiment_score],
                    marker_color=['#00ffcc', '#ff3366', '#4facfe']
                ))
                fig.update_layout(title="Linguistic Feature Distribution", template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, use_container_width=True)

            with col_cloud:
                wc = WordCloud(background_color="black", width=400, height=250).generate(input_text)
                st.image(wc.to_array(), caption="Key Forensic Identifiers Found", use_container_width=True)

            # E. VERDICT BOX
            st.markdown("---")
            if p_real > 50:
                st.success(f"✅ VERDICT: This content passes the VeriLens authenticity test ({p_real:.1f}% confidence).")
            else:
                st.error(f"⚠️ VERDICT: High risk of misinformation or propaganda detected ({p_fake:.1f}% risk).")
            
            # F. DOWNLOAD FEATURE
            report = f"VeriLens AI Report\nAuthor: Isha\n\nResult: {'AUTHENTIC' if p_real > 50 else 'FAKE'}\nCredibility: {p_real:.1f}%\nBias: {sentiment_score:.1f}%\nContent: {input_text[:100]}..."
            st.download_button("📩 Download Analysis Report", report, file_name="VeriLens_Report.txt")

        else:
            st.warning("Please provide content to analyze.")
    st.markdown('</div>', unsafe_allow_html=True)

# 6. FOOTER
st.markdown("<br><p style='text-align: center; color: #4facfe;'>Developed by ISHA ❤️ 2026 | Powered by Neural Intelligence</p>", unsafe_allow_html=True)
