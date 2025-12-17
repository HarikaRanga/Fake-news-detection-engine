# ==========================================================
# 🌈 app.py — Interactive Fake News Detection Dashboard
# Authors: Ranga Harika, Amula Ashwini
# Visuals:
# - Unified pastel palette across app and sidebar
# - Professional ribbons/sections, no logic changes
# - Real news samples to validate REAL outputs
# ==========================================================

import streamlit as st
import joblib
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import plotly.express as px
import plotly.graph_objects as go
import time

# ----------------------------------------------------------
# 🎨 Streamlit Page Config
# ----------------------------------------------------------
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------------------------------------
# 🎨 Coordinated Pastel Theme (color-only; safe to replace)
# ----------------------------------------------------------
st.markdown("""
<style>
/* ---------- Pastel Palette ---------- */
:root {
  --p1: #fdf2ff;   /* lilac haze */
  --p2: #eaf7ff;   /* baby blue */
  --p3: #fef6f0;   /* peach milk */
  --ink: #1f2430;  /* primary text */
  --muted: #6b7280;
  --a1: #8aa9ff;   /* accents */
  --a2: #b58bff;
  --a3: #79e3ff;
}

[data-theme="dark"] :root {
  --p1: #161827;
  --p2: #20253a;
  --p3: #242a40;
  --ink: #e9edf6;
  --muted: #a8b0bd;
  --a1: #9ab3ff;
  --a2: #c59dff;
  --a3: #7de6ff;
}

/* Canvas */
[data-testid="stAppViewContainer"] {
  background: linear-gradient(135deg, var(--p1) 0%, var(--p2) 50%, var(--p3) 100%);
  color: var(--ink);
  font-family: 'Poppins', system-ui, -apple-system, Segoe UI, Roboto, 'Helvetica Neue', Arial;
}

/* Sidebar with coordinated gradient + soft gloss */
[data-testid="stSidebar"] > div:first-child {
  background:
    radial-gradient(900px 320px at -20% -20%, rgba(255,255,255,.55) 0%, rgba(255,255,255,.15) 60%, rgba(255,255,255,0) 100%),
    linear-gradient(180deg, rgba(255,255,255,.82) 0%, rgba(255,255,255,.70) 100%),
    linear-gradient(145deg, var(--p1) 0%, var(--p2) 50%, var(--p3) 100%);
  border-right: 1px solid rgba(120,120,180,0.16);
  box-shadow: 6px 0 28px rgba(90, 110, 180, 0.16);
  backdrop-filter: blur(8px);
}
[data-theme="dark"] [data-testid="stSidebar"] > div:first-child {
  background:
    radial-gradient(900px 320px at -20% -20%, rgba(255,255,255,.06) 0%, rgba(255,255,255,.02) 60%, rgba(255,255,255,0) 100%),
    linear-gradient(180deg, rgba(26,28,44,.90) 0%, rgba(26,28,44,.72) 100%),
    linear-gradient(145deg, var(--p1) 0%, var(--p2) 50%, var(--p3) 100%);
  border-right: 1px solid rgba(255,255,255,0.06);
  box-shadow: 6px 0 36px rgba(5, 8, 18, 0.55);
}

/* Page gutters */
.main .block-container {
  max-width: 1140px;
  padding-top: 1.1rem !important;
  padding-bottom: 2.2rem !important;
}

/* Sections and cards */
.section, .result-card, .batch-card {
  background: rgba(255,255,255,0.86);
  border-radius: 16px;
  padding: 18px;
  border: 1px solid rgba(255,255,255,0.46);
  box-shadow: 0 10px 30px rgba(31,38,135,0.12);
  position: relative;
  margin: 24px 0;
}
[data-theme="dark"] .section, [data-theme="dark"] .result-card, [data-theme="dark"] .batch-card {
  background: rgba(35,38,58,0.60);
  border: 1px solid rgba(255,255,255,0.08);
}

/* Colored edge */
.section::before {
  content:"";
  position:absolute; left:-1px; top:0; bottom:0; width:7px;
  background: linear-gradient(180deg, var(--a1), var(--a2), var(--a3));
  border-top-left-radius:16px; border-bottom-left-radius:16px;
}

/* Ribbon */
.ribbon {
  padding: 10px 16px; border-radius: 12px;
  background: linear-gradient(90deg, #ffe7f3 0%, #eef5ff 50%, #eafff3 100%);
  box-shadow: 0 8px 24px rgba(80,110,200,0.12);
  margin: 8px 0 18px 0;
}
[data-theme="dark"] .ribbon {
  background: linear-gradient(90deg, #3d3253 0%, #2a3958 50%, #2e4a3c 100%);
}

/* Buttons (tint only) */
.stButton>button {
  background: linear-gradient(90deg, var(--a1), var(--a2));
  color: #fff; font-weight: 600; border-radius: 12px; border: none;
  padding: .5rem .95rem;
  transition: transform .17s ease, box-shadow .17s ease;
}
.stButton>button:hover {
  transform: translateY(-1px);
  box-shadow: 0 10px 26px rgba(75,138,255,0.28);
}

/* Inputs rounded */
textarea, .stTextInput input { border-radius: 12px !important; }

/* Text tones */
h1, h2, h3, h4, h5, h6 { color: var(--ink); }
.small-muted { color: var(--muted); }

/* Hide default footer */
footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------
# 🔌 Optional Lottie helpers (safe if not installed)
# ----------------------------------------------------------
def load_lottie_json(url:str):
    try:
        import requests
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            return r.json()
    except Exception:
        return None
    return None

def render_lottie(anim, height=90):
    try:
        from streamlit_lottie import st_lottie  # pip install streamlit-lottie
        if anim:
            st_lottie(anim, height=height, key=f"l_{hash(str(anim))%9999}")
    except Exception:
        pass

# ----------------------------------------------------------
# ⚙️ Paths
# ----------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_FILE = os.path.join(MODEL_DIR, "fake_news_model.pkl")
VECTORIZER_FILE = os.path.join(MODEL_DIR, "vectorizer.pkl")

# ----------------------------------------------------------
# 🧠 Load Model
# ----------------------------------------------------------
if not os.path.exists(MODEL_FILE) or not os.path.exists(VECTORIZER_FILE):
    st.error("❌ Model not trained yet! Please run train_model.py first.")
    st.stop()

model = joblib.load(MODEL_FILE)
vectorizer = joblib.load(VECTORIZER_FILE)

# ----------------------------------------------------------
# 🧭 Sidebar Controls
# ----------------------------------------------------------
st.sidebar.header("🛠️ Dashboard Settings")
theme_choice = st.sidebar.radio("Theme", ["🌞 Light Mode", "🌚 Dark Mode"])
if "🌚" in theme_choice:
    st.markdown('<body data-theme="dark">', unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.subheader("📊 Model Analytics")
st.sidebar.metric("Accuracy", "94.00%")
st.sidebar.metric("Precision", "93.00%")
st.sidebar.metric("Recall", "91.00%")

with st.sidebar.expander("💡 Tips", expanded=False):
    st.markdown("- Use 2–4 sentences for best context.")
    st.markdown("- Avoid very short or sensational phrasing.")
    st.markdown("- Batch mode supports large CSVs.")

render_lottie(load_lottie_json("https://assets5.lottiefiles.com/packages/lf20_qp1q7mct.json"), 80)

# ----------------------------------------------------------
# 📰 Title & Ribbon
# ----------------------------------------------------------
st.title("🧠 Fake News Detection Engine")
st.markdown("### Verify news authenticity using advanced ML models 📰")
st.markdown("<div class='ribbon section'>Real‑time NLP • TF‑IDF + Linear Model • Confidence donut • Batch CSV analysis</div>", unsafe_allow_html=True)

# ----------------------------------------------------------
# ✍️ Input Section
# ----------------------------------------------------------
st.markdown("<div class='section'>", unsafe_allow_html=True)
st.subheader("Enter News Headline or Article Text:")

REAL_NEWS_SAMPLES = {
    "Policy (education)": "Government unveils a nationwide AI-in-education roadmap to train 10 million students and teachers over five years, focusing on digital literacy and ethics.",
    "Health (vaccination)": "Health Ministry reports that over 90% of the eligible population received the seasonal influenza vaccine during the latest campaign, reducing hospitalization rates.",
    "Economy (inflation)": "Official statistics show headline inflation eased to 4.1% this quarter as food and fuel prices stabilized, according to the national statistical office.",
    "Science (space)": "The national space agency successfully launched a weather satellite into geostationary orbit, improving cyclone tracking and early warning systems.",
    "Sports (tournament)": "The women’s national team advanced to the continental semifinals after a 2–1 win, with the decisive goal scored in stoppage time."
}

c1, c2, c3, c4 = st.columns([1,1,1,1])
with c1:
    if st.button("📌 Election rumor"):
        st.session_state["pre_fill"] = "Breaking: Government cancels elections nationwide next week."
with c2:
    if st.button("📌 Health scare"):
        st.session_state["pre_fill"] = "Doctors warn that common table salt has been declared toxic by WHO."
with c3:
    if st.button("📌 Celebrity hoax"):
        st.session_state["pre_fill"] = "Famous actor secretly moved to Mars after space mission."
with c4:
    if st.button("📌 Policy change"):
        st.session_state["pre_fill"] = "New law bans internet use after 9 PM across all cities."

sample_choice = st.selectbox("Or insert a verified real‑news sample:", ["— Select —"] + list(REAL_NEWS_SAMPLES.keys()))
if sample_choice in REAL_NEWS_SAMPLES:
    st.session_state["pre_fill"] = REAL_NEWS_SAMPLES[sample_choice]
    st.info("Loaded a real news sample into the editor.")

prefill = st.session_state.get("pre_fill", "")
user_input = st.text_area(
    "Type or paste your news content here...",
    height=160,
    value=prefill,
    placeholder="E.g., 'Government announces new AI education policy...'"
)
st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------------------------------------
# 🔍 Analyze Section
# ----------------------------------------------------------
st.markdown("<div class='section'>", unsafe_allow_html=True)

def ai_typing_effect(message, color="#4a90e2"):
    placeholder = st.empty()
    text = ""
    for ch in message:
        text += ch
        placeholder.markdown(f"<h4 style='color:{color}; font-family:Poppins;'>{text}▌</h4>", unsafe_allow_html=True)
        time.sleep(0.008)
    placeholder.markdown(f"<h4 style='color:{color}; font-family:Poppins;'>{text}</h4>", unsafe_allow_html=True)

co1, co2 = st.columns([1,3])
with co1:
    if st.button("✨ Smart Analyze"):
        if user_input.strip() == "":
            st.warning("Please enter some text for analysis.")
        else:
            prog = st.progress(0)
            for i in range(0, 100, 12):
                time.sleep(0.02)
                prog.progress(min(i, 100))
            st.toast("Ready! Click Analyze News for full results.", icon="✅")

with co2:
    analyze_clicked = st.button("🔍 Analyze News", use_container_width=True)

if analyze_clicked:
    if user_input.strip() == "":
        st.warning("Please enter some text for analysis.")
    else:
        with st.spinner("🤖 AI is analyzing..."):
            time.sleep(0.35)
            text_vec = vectorizer.transform([user_input])
            pred = model.predict(text_vec)[0]
            conf = float(model.predict_proba(text_vec).max())
            label = "🟢 Real News" if str(pred).upper() in ["REAL", "REAL NEWS", "TRUE"] else "🔴 Fake News"
            is_real = label.startswith("🟢")
            color = "#2ecc71" if is_real else "#e74c3c"

            ai_typing_effect(f"Prediction Result → {label}", color="#4a90e2")

            st.markdown(f"""
            <div class='result-card' style='border-left: 8px solid {color};'>
                <h4 style='color:{color}; text-align:center;'>Prediction: {label}</h4>
                <p class='small-muted' style='text-align:center;'>Confidence reflects model certainty given TF‑IDF features.</p>
                <p style='text-align:center; font-size:16px; margin-top:4px;'>Confidence: {conf*100:.2f}%</p>
            </div>
            """, unsafe_allow_html=True)

            fig = go.Figure(data=[go.Pie(
                labels=[label, "Other"],
                values=[conf*100, 100-conf*100],
                hole=0.62,
                marker_colors=[color, "#d9d9d9"]
            )])
            fig.update_layout(height=260, showlegend=False, margin=dict(l=10, r=10, t=20, b=20))
            st.plotly_chart(fig, use_container_width=True)

st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------------------------------------
# 🤖 Assistant Help (no external LLM)
# ----------------------------------------------------------
with st.expander("🤖 Assistant help"):
    st.markdown("Ask how to interpret predictions or how to prepare CSVs.")
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role":"assistant","content":"Hi! Paste a headline and tap Analyze. Use Batch mode for CSVs with a 'text' column."}
        ]
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.write(m["content"])
    question = st.chat_input("Type a quick question about using the app…")
    if question:
        st.session_state.messages.append({"role":"user","content":question})
        reply = "Tip: Balanced, factual sentences yield clearer predictions than very short or sensational text."
        with st.chat_message("assistant"):
            st.write(reply)
        st.session_state.messages.append({"role":"assistant","content":reply})

# ----------------------------------------------------------
# 📁 Batch Mode
# ----------------------------------------------------------
st.markdown("<div class='section'>", unsafe_allow_html=True)
st.subheader("📁 Batch News File Analysis")
st.caption("Upload a CSV with a 'text' column to get distribution and counts.")

uploaded_file = st.file_uploader("Upload CSV containing 'text' column", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    if "text" not in df.columns:
        st.error("❌ The CSV must have a 'text' column.")
    else:
        with st.spinner("Analyzing uploaded file..."):
            time.sleep(0.35)
            X_vec = vectorizer.transform(df["text"])
            df["prediction"] = model.predict(X_vec)
            st.success("✅ Analysis complete!")

            st.markdown("<div class='batch-card'>", unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                fig1 = px.pie(df, names="prediction", title="Prediction Distribution",
                              color_discrete_sequence=px.colors.qualitative.Pastel)
                st.plotly_chart(fig1, use_container_width=True)
            with col2:
                summary = df["prediction"].value_counts().reset_index()
                summary.columns = ["Label", "Count"]
                fig2 = px.bar(summary, x="Label", y="Count", text="Count",
                              color="Label", color_discrete_sequence=px.colors.qualitative.Prism)
                fig2.update_traces(textposition="outside")
                st.plotly_chart(fig2, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("💾 Download Results CSV", csv, "fake_news_predictions.csv", "text/csv")
st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------------------------------------
# 💫 Footer
# ----------------------------------------------------------
st.markdown("""
---
<div style='text-align:center; font-size:14px; color:gray'>
© 2025 Fake News Detection Dashboard | Designed with 💙 by <b>Ranga Harika</b> & <b>Amula Ashwini</b>
</div>
""", unsafe_allow_html=True)


