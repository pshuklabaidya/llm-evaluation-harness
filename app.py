import streamlit as st

st.set_page_config(page_title="LLM Evaluation Harness", layout="wide")

st.title("LLM Evaluation Harness")
st.write(
    "A lightweight framework for evaluating LLM and RAG outputs across relevance, "
    "groundedness, completeness, hallucination risk, and regression behavior."
)

st.info("Phase 1 complete. Retrieval, scoring, and reporting will be added in later phases.")
