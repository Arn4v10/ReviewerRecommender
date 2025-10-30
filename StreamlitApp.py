
import os
os.environ["USE_TF"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import fitz
import re

def extract_full_text_from_pdf(file):
    text = ""
    try:
        pdf = fitz.open(stream=file.read(), filetype="pdf")
        for page in pdf:
            text += page.get_text("text") + " "
        return re.sub(r"\s+", " ", text).strip()
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return ""

def make_doc_text(row):
    return f"{row['title']} {row['abstract']} {row['keywords']}".strip()

def aggregate_author_scores(df, paper_scores):
    author_scores = {}

    for (_, row), score in zip(df.iterrows(), paper_scores):
        author = row['author'].strip()
        if author:
            author_scores.setdefault(author, []).append(score)

    return {a: float(np.mean(v)) for a, v in author_scores.items()}


@st.cache_resource
def load_model_and_data():
    df = pd.read_csv("papers_expanded.csv").fillna("")
    
    model = SentenceTransformer('all-MiniLM-L6-v2')
    corpus = [make_doc_text(r) for _, r in df.iterrows()]
    corpus = [re.sub(r'\s+', ' ', t).strip() for t in corpus]
    
    embeddings = model.encode(corpus, batch_size=64, convert_to_numpy=True, show_progress_bar=True)
    return df, model, embeddings

df, model, embeddings = load_model_and_data()


st.title("Reviewer Recommendation System ")
st.write("Upload a research PDF to find the most relevant reviewers.")

uploaded_pdf = st.file_uploader("Upload PDF File", type=["pdf"])
top_k = st.number_input("Top K Reviewers", min_value=1, max_value=20, value=5)

if uploaded_pdf:
    st.info("Extracting text from PDF...")
    query_text = extract_full_text_from_pdf(uploaded_pdf)

    if query_text:
        st.success("PDF text extracted successfully.")

        st.write("Encoding & computing similarity...")

        q_emb = model.encode([query_text], convert_to_numpy=True)
        sims = cosine_similarity(q_emb, embeddings).flatten()

        author_scores = aggregate_author_scores(df, sims)
        sorted_authors = sorted(author_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        st.subheader("🔝 Top Recommended Reviewers")
        results_df = pd.DataFrame({
            "Rank": range(1, len(sorted_authors)+1),
            "Author": [a for a, _ in sorted_authors],
            "Similarity Score": [f"{s:.4f}" for _, s in sorted_authors]
        })
        st.dataframe(results_df)

        st.success("Done")
    else:
        st.warning("Could not extract text from PDF.")
