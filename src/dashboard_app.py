# dashboard_final.py
import streamlit as st
import pandas as pd
import numpy as np
from transformers import pipeline
import plotly.express as px
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import time

# ===================== Page Config =====================
st.set_page_config(page_title="📰 Fake News Detector Pro", layout="wide", page_icon="📰")

# ===================== Sidebar =====================
st.sidebar.title("Navigation")
mode = st.sidebar.radio("Select Mode:", ["Single Prediction", "Batch CSV Upload", "Dashboard Metrics"])
theme_choice = st.sidebar.selectbox("Select Theme:", ["Light", "Dark"])

# Apply theme
if theme_choice == "Dark":
    st.markdown("<style>body{background-color:#0E1117;color:white;}</style>", unsafe_allow_html=True)
else:
    st.markdown("<style>body{background-color:white;color:black;}</style>", unsafe_allow_html=True)

# ===================== Load DistilBERT Model =====================
@st.cache_resource(show_spinner=False)
def load_model():
    classifier = pipeline(
        "text-classification",
        model="mrm8488/distilbert-base-uncased-finetuned-fake-news",
        return_all_scores=True
    )
    return classifier

classifier = load_model()

# ===================== Helper Functions =====================
def get_keywords(text, top_n=5):
    words = [w.lower() for w in text.split() if w.isalpha() and w.lower() not in ENGLISH_STOP_WORDS]
    freq = pd.Series(words).value_counts()
    return freq.head(top_n)

def animated_confidence(label, confidence):
    st.subheader(f"Prediction: {label}")
    progress_bar = st.progress(0)
    for percent in range(0, int(confidence)+1, 2):
        progress_bar.progress(percent)
        time.sleep(0.01)
    st.write(f"Confidence: **{confidence:.2f}%**")

# ===================== Single Prediction =====================
if mode == "Single Prediction":
    st.title("📰 Fake News Single Prediction")
    user_input = st.text_area("📝 Enter news headline or article:", height=150)
    if st.button("🔍 Analyze"):
        if user_input.strip() == "":
            st.warning("Please enter text!")
        else:
            result = classifier(user_input)[0]
            pred_label = max(result, key=lambda x: x['score'])['label']
            confidence = max(result, key=lambda x: x['score'])['score'] * 100

            # Animated confidence bar
            animated_confidence(pred_label, confidence)

            # Top keywords
            keywords = get_keywords(user_input)
            st.subheader("Top Keywords Influencing Prediction:")
            st.write(", ".join(keywords.index.tolist()))

            # Interactive Pie Chart
            df_pie = pd.DataFrame({"Label":[x['label'] for x in result], "Score":[x['score']*100 for x in result]})
            fig_pie = px.pie(df_pie, names='Label', values='Score', color='Label',
                             color_discrete_map={"REAL":"#2ecc71", "FAKE":"#e74c3c"}, 
                             title="Prediction Probability Distribution")
            st.plotly_chart(fig_pie, use_container_width=True)

            # Interactive Bar Chart
            fig_bar = px.bar(df_pie, x='Label', y='Score', color='Label',
                             color_discrete_map={"REAL":"#2ecc71", "FAKE":"#e74c3c"},
                             text='Score', title="Confidence Bar Graph")
            fig_bar.update_layout(yaxis=dict(range=[0,100]))
            st.plotly_chart(fig_bar, use_container_width=True)

# ===================== Batch CSV Upload =====================
if mode == "Batch CSV Upload":
    st.title("🗂️ Batch CSV Prediction")
    uploaded_file = st.file_uploader("Upload CSV with column 'text'", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        if 'text' not in df.columns:
            st.error("CSV must have a column named 'text'")
        else:
            predictions, confidences, top_keywords = [], [], []
            progress = st.progress(0)
            for i, txt in enumerate(df['text']):
                res = classifier(txt)[0]
                pred = max(res, key=lambda x: x['score'])['label']
                conf = max(res, key=lambda x: x['score'])['score']*100
                predictions.append(pred)
                confidences.append(conf)
                top_keywords.append(", ".join(get_keywords(txt).index.tolist()))
                progress.progress(int((i+1)/len(df)*100))

            df['Prediction'] = predictions
            df['Confidence'] = confidences
            df['Top Keywords'] = top_keywords
            st.success("Batch Predictions Complete!")

            st.dataframe(df, height=400)

            # Interactive Pie Chart for distribution
            counts = df['Prediction'].value_counts().reset_index()
            counts.columns = ['Label', 'Count']
            fig_pie = px.pie(counts, names='Label', values='Count', color='Label',
                             color_discrete_map={"REAL":"#2ecc71", "FAKE":"#e74c3c"},
                             title="Overall Prediction Distribution")
            st.plotly_chart(fig_pie, use_container_width=True)

            # Average Confidence Bar Chart
            avg_conf = df.groupby('Prediction')['Confidence'].mean().reset_index()
            fig_bar = px.bar(avg_conf, x='Prediction', y='Confidence', color='Prediction',
                             color_discrete_map={"REAL":"#2ecc71", "FAKE":"#e74c3c"},
                             text='Confidence', title="Average Confidence by Prediction")
            fig_bar.update_layout(yaxis=dict(range=[0,100]))
            st.plotly_chart(fig_bar, use_container_width=True)

            # Confidence Histogram
            fig_hist = px.histogram(df, x='Confidence', color='Prediction', nbins=10,
                                    color_discrete_map={"REAL":"#2ecc71", "FAKE":"#e74c3c"},
                                    title="Confidence Distribution")
            st.plotly_chart(fig_hist, use_container_width=True)

            # Download predictions
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Predictions CSV", data=csv, file_name="predictions.csv", mime="text/csv")

# ===================== Dashboard Metrics =====================
if mode == "Dashboard Metrics":
    st.title("📊 Dashboard Metrics")
    st.markdown("Visual overview of predictions, confidence, and keywords. Upload CSV in 'Batch CSV Upload' mode to generate metrics.")

    st.info("Demo metrics are displayed here for placeholder purposes.")
    demo_df = pd.DataFrame({
        "Prediction": ["Real"]*60 + ["Fake"]*40,
        "Confidence": np.random.randint(80,100,100)
    })

    # Pie chart
    counts = demo_df['Prediction'].value_counts().reset_index()
    counts.columns = ['Label','Count']
    fig_pie = px.pie(counts, names='Label', values='Count', color='Label',
                     color_discrete_map={"REAL":"#2ecc71", "FAKE":"#e74c3c"},
                     title="Prediction Distribution")
    st.plotly_chart(fig_pie, use_container_width=True)

    # Confidence bar
    avg_conf = demo_df.groupby('Prediction')['Confidence'].mean().reset_index()
    fig_bar = px.bar(avg_conf, x='Prediction', y='Confidence', color='Prediction',
                     color_discrete_map={"REAL":"#2ecc71", "FAKE":"#e74c3c"},
                     text='Confidence', title="Average Confidence by Prediction")
    fig_bar.update_layout(yaxis=dict(range=[0,100]))
    st.plotly_chart(fig_bar, use_container_width=True)

    # Histogram
    fig_hist = px.histogram(demo_df, x='Confidence', color='Prediction', nbins=10,
                            color_discrete_map={"REAL":"#2ecc71", "FAKE":"#e74c3c"},
                            title="Confidence Distribution")
    st.plotly_chart(fig_hist, use_container_width=True)

    st.markdown("✅ This final professional dashboard now includes **interactive charts, animated confidence meters, top keywords, batch CSV upload, and a clean light/dark mode**. It’s ready for demo and presentation!")

