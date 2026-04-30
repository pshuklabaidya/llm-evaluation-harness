import sys
from pathlib import Path

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.evaluator import run_evaluation, save_results
from src.mock_llm import SUPPORTED_MODES
from src.report import build_report, save_report, load_results


DEFAULT_DATASET_PATH = "data/eval_questions.csv"
DEFAULT_RESULTS_PATH = "results/evaluation_results.csv"
DEFAULT_REPORT_PATH = "results/sample_evaluation_report.md"


st.set_page_config(
    page_title="LLM Evaluation Harness",
    layout="wide",
)


def get_existing_results() -> pd.DataFrame | None:
    """Load existing evaluation results when available."""
    path = Path(DEFAULT_RESULTS_PATH)

    if not path.exists():
        return None

    return load_results(path)


def calculate_metric_values(results: pd.DataFrame) -> dict:
    """Calculate dashboard summary metrics."""
    return {
        "total_questions": len(results),
        "average_overall": round(results["overall_score"].mean(), 2),
        "average_relevance": round(results["answer_relevance"].mean(), 2),
        "average_groundedness": round(results["groundedness"].mean(), 2),
        "average_completeness": round(results["completeness"].mean(), 2),
    }


def build_risk_dataframe(results: pd.DataFrame) -> pd.DataFrame:
    """Build hallucination-risk count table for charting."""
    risk_order = ["Low", "Medium", "High"]
    counts = results["hallucination_risk"].value_counts().to_dict()

    return pd.DataFrame(
        {
            "Risk": risk_order,
            "Count": [counts.get(risk, 0) for risk in risk_order],
        }
    ).set_index("Risk")


st.title("LLM Evaluation Harness")
st.write(
    "A lightweight evaluation dashboard for testing LLM and RAG outputs across "
    "relevance, groundedness, completeness, hallucination risk, and overall quality."
)

with st.sidebar:
    st.header("Evaluation Controls")

    dataset_path = st.text_input(
        "Evaluation dataset path",
        value=DEFAULT_DATASET_PATH,
    )

    mode = st.selectbox(
        "Mock answer mode",
        options=sorted(SUPPORTED_MODES),
        index=sorted(SUPPORTED_MODES).index("grounded"),
    )

    top_k = st.slider(
        "Retrieved context sections",
        min_value=1,
        max_value=5,
        value=2,
    )

    run_button = st.button(
        "Run Evaluation",
        type="primary",
        use_container_width=True,
    )

    st.divider()

    st.caption("The MVP uses deterministic mock answers and does not require a paid API key.")


if "results" not in st.session_state:
    st.session_state["results"] = get_existing_results()

if "report" not in st.session_state:
    st.session_state["report"] = ""

if run_button:
    try:
        results = run_evaluation(
            dataset_path=dataset_path,
            mode=mode,
            top_k=top_k,
        )

        save_results(results)

        report = build_report(results)
        save_report(report)

        st.session_state["results"] = results
        st.session_state["report"] = report

        st.success("Evaluation completed successfully.")

    except Exception as error:
        st.error(f"Evaluation failed: {error}")


results = st.session_state.get("results")

if results is None:
    st.info("Run an evaluation from the sidebar to populate dashboard results.")
    st.stop()


metrics = calculate_metric_values(results)

metric_col_1, metric_col_2, metric_col_3, metric_col_4, metric_col_5 = st.columns(5)

metric_col_1.metric("Questions", metrics["total_questions"])
metric_col_2.metric("Overall", metrics["average_overall"])
metric_col_3.metric("Relevance", metrics["average_relevance"])
metric_col_4.metric("Groundedness", metrics["average_groundedness"])
metric_col_5.metric("Completeness", metrics["average_completeness"])

st.divider()

left_col, right_col = st.columns([1, 2])

with left_col:
    st.subheader("Hallucination Risk")
    risk_df = build_risk_dataframe(results)
    st.bar_chart(risk_df)

with right_col:
    st.subheader("Score Table")

    table_columns = [
        "id",
        "question",
        "top_retrieved_title",
        "answer_relevance",
        "groundedness",
        "completeness",
        "overall_score",
        "hallucination_risk",
    ]

    st.dataframe(
        results[table_columns],
        use_container_width=True,
        hide_index=True,
    )

st.divider()

st.subheader("Question-Level Details")

for _, row in results.iterrows():
    expander_label = (
        f"Q{row['id']} - Overall {row['overall_score']} - "
        f"Risk: {row['hallucination_risk']}"
    )

    with st.expander(expander_label):
        st.markdown(f"**Question:** {row['question']}")
        st.markdown(f"**Expected Answer:** {row['expected_answer']}")
        st.markdown(f"**Generated Answer:** {row['generated_answer']}")
        st.markdown(f"**Top Retrieved Context:** {row['top_retrieved_title']}")

        unsupported = str(row.get("unsupported_sentences", "")).strip()

        if unsupported:
            st.warning(f"Unsupported sentence flags: {unsupported}")

st.divider()

st.subheader("Report Preview")

report_text = st.session_state.get("report")

if not report_text:
    report_path = Path(DEFAULT_REPORT_PATH)

    if report_path.exists():
        report_text = report_path.read_text(encoding="utf-8")
    else:
        report_text = build_report(results)

st.markdown(report_text)

st.divider()

download_col_1, download_col_2 = st.columns(2)

with download_col_1:
    st.download_button(
        label="Download Results CSV",
        data=results.to_csv(index=False),
        file_name="evaluation_results.csv",
        mime="text/csv",
        use_container_width=True,
    )

with download_col_2:
    st.download_button(
        label="Download Markdown Report",
        data=report_text,
        file_name="sample_evaluation_report.md",
        mime="text/markdown",
        use_container_width=True,
    )
