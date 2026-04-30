import argparse
import sys
from pathlib import Path
from typing import Dict

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.evaluator import run_evaluation, save_results


DEFAULT_RESULTS_CSV_PATH = Path("results/evaluation_results.csv")
DEFAULT_REPORT_OUTPUT_PATH = Path("results/sample_evaluation_report.md")


def load_results(results_csv_path: str | Path = DEFAULT_RESULTS_CSV_PATH) -> pd.DataFrame:
    """Load evaluation results from CSV."""
    path = Path(results_csv_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Evaluation results not found: {path}. Run python src/evaluator.py first."
        )

    results = pd.read_csv(path)
    results = results.fillna("")

    return results


def calculate_summary_metrics(results: pd.DataFrame) -> Dict[str, object]:
    """Calculate report-level summary metrics."""
    hallucination_counts = (
        results["hallucination_risk"]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    return {
        "total_questions": len(results),
        "average_overall_score": round(results["overall_score"].mean(), 2),
        "average_relevance": round(results["answer_relevance"].mean(), 2),
        "average_groundedness": round(results["groundedness"].mean(), 2),
        "average_completeness": round(results["completeness"].mean(), 2),
        "hallucination_counts": hallucination_counts,
    }


def format_hallucination_counts(counts: Dict[str, int]) -> str:
    """Format hallucination-risk counts for markdown."""
    low = counts.get("Low", 0)
    medium = counts.get("Medium", 0)
    high = counts.get("High", 0)

    return f"Low: {low} | Medium: {medium} | High: {high}"


def build_question_table(results: pd.DataFrame) -> str:
    """Build markdown table with question-level evaluation results."""
    lines = [
        "| ID | Top Context | Relevance | Groundedness | Completeness | Overall | Risk |",
        "|---:|---|---:|---:|---:|---:|---|",
    ]

    for _, row in results.iterrows():
        lines.append(
            "| "
            f"{row['id']} | "
            f"{row['top_retrieved_title']} | "
            f"{row['answer_relevance']} | "
            f"{row['groundedness']} | "
            f"{row['completeness']} | "
            f"{row['overall_score']} | "
            f"{row['hallucination_risk']} |"
        )

    return "\n".join(lines)


def build_question_details(results: pd.DataFrame) -> str:
    """Build detailed question-level report notes."""
    sections = []

    for _, row in results.iterrows():
        unsupported = str(row.get("unsupported_sentences", "")).strip()

        detail = [
            f"### Q{row['id']}: {row['question']}",
            "",
            f"**Expected Answer:** {row['expected_answer']}",
            "",
            f"**Generated Answer:** {row['generated_answer']}",
            "",
            f"**Top Retrieved Context:** {row['top_retrieved_title']}",
            "",
            f"**Overall Score:** {row['overall_score']}",
            "",
            f"**Hallucination Risk:** {row['hallucination_risk']}",
        ]

        if unsupported:
            detail.extend(
                [
                    "",
                    "**Unsupported Sentences:**",
                    "",
                    f"- {unsupported}",
                ]
            )

        sections.append("\n".join(detail))

    return "\n\n".join(sections)


def build_report(results: pd.DataFrame) -> str:
    """Build the full markdown evaluation report."""
    metrics = calculate_summary_metrics(results)

    question_table = build_question_table(results)
    question_details = build_question_details(results)

    report = f"""# LLM Evaluation Harness - Sample Evaluation Report

## Executive Summary

The evaluation run tested {metrics['total_questions']} synthetic questions against a controlled knowledge base. The pipeline performed retrieval, mock answer generation, rule-based scoring, hallucination-risk detection, and result export.

## Summary Metrics

| Metric | Value |
|---|---:|
| Total Questions | {metrics['total_questions']} |
| Average Overall Score | {metrics['average_overall_score']} |
| Average Relevance | {metrics['average_relevance']} |
| Average Groundedness | {metrics['average_groundedness']} |
| Average Completeness | {metrics['average_completeness']} |

## Hallucination Risk Distribution

{format_hallucination_counts(metrics['hallucination_counts'])}

## Question-Level Results

{question_table}

## Evaluation Method

The evaluation pipeline uses transparent rule-based scoring. Retrieval is performed over a synthetic markdown knowledge base using TF-IDF with topic boosting. Answer generation uses deterministic mock responses so the harness can run without paid model access.

Scoring covers:

- Answer relevance
- Groundedness
- Completeness
- Hallucination risk
- Overall quality

## Detailed Results

{question_details}

## Portfolio Notes

All data is synthetic and safe for public demonstration. The project shows practical AI quality-assurance skills for RAG systems, including retrieval testing, groundedness checks, hallucination-risk detection, and repeatable evaluation reporting.
"""

    return report


def save_report(
    report: str,
    output_path: str | Path = DEFAULT_REPORT_OUTPUT_PATH,
) -> None:
    """Save markdown report to disk."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate a markdown evaluation report."
    )

    parser.add_argument(
        "--results-csv",
        default=str(DEFAULT_RESULTS_CSV_PATH),
        help="Path to evaluation results CSV.",
    )

    parser.add_argument(
        "--output",
        default=str(DEFAULT_REPORT_OUTPUT_PATH),
        help="Path for markdown report output.",
    )

    parser.add_argument(
        "--refresh-results",
        action="store_true",
        help="Run a fresh grounded evaluation before generating the report.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.refresh_results:
        refreshed_results = run_evaluation(mode="grounded")
        save_results(refreshed_results)

    evaluation_results = load_results(args.results_csv)
    markdown_report = build_report(evaluation_results)
    save_report(markdown_report, args.output)

    print(f"Saved markdown report to: {args.output}")
