import streamlit as st
import pandas as pd
import tensorflow as tf
import joblib
import numpy as np

# =========================
# Load Model
# =========================
@st.cache_resource
def load_models():
    sentiment_model = joblib.load("models/SVM.pkl")
    rating_model = tf.keras.models.load_model("models/rating_model.h5")
    cluster_model = joblib.load("models/cluster_model.joblib")
    tfidf_vectorizer = joblib.load("models/tfidf_vectorizer.joblib")
    vectorizer = joblib.load("models/vectorizer.pkl")
    tokenizer = joblib.load("models/tokenizer.pkl")
    return sentiment_model, rating_model, vectorizer, tokenizer, cluster_model, tfidf_vectorizer

sentiment_model, rating_model, vectorizer, tokenizer, cluster_model, tfidf_vectorizer = load_models()

# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="Sentiment, Rating & Topic Analyzer", layout="wide")

# Sidebar
st.sidebar.title("⚙️ Opsi Analisis")
st.sidebar.write("Pilih opsi analisis yang ingin dijalankan:")

show_sentiment = st.sidebar.checkbox("Analisis Sentimen", True)
show_rating = st.sidebar.checkbox("Prediksi Rating", True)
show_topic = st.sidebar.checkbox("Clustering Topik", True)

st.title("📊 Sentiment Analysis, Rating Prediction & Topic Clustering")

# Input dari user
user_input = st.text_area("Masukkan teks ulasan:", "")

if st.button("Analisis"):
    if user_input.strip() != "":
        # Transformasi input untuk Sentiment & Rating
        X_input = vectorizer.transform([user_input])

        # Placeholder untuk hasil
        sentiment_pred, rating_pred, topic_pred = None, None, None

        # Prediksi Sentiment
        if show_sentiment:
            sentiment_pred = sentiment_model.predict(X_input)[0]

        # Prediksi Rating
        if show_rating:
            rating_pred = np.argmax(rating_model.predict(X_input), axis=1)[0] + 1

        # Prediksi Topic
        if show_topic:
            X_cluster = tfidf_vectorizer.transform([user_input])
            topic_pred = cluster_model.predict(X_cluster)[0]

        # Tampilkan hasil
        st.subheader("🔎 Hasil Analisis")
        cols = st.columns(3)

        if show_sentiment:
            cols[0].metric("Sentiment", sentiment_pred)
        if show_rating:
            cols[1].metric("Rating", rating_pred)
        if show_topic:
            cols[2].metric("Topic", topic_pred)

    else:
        st.warning("Silakan masukkan teks ulasan terlebih dahulu.")
