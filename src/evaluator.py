import argparse
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.mock_llm import SUPPORTED_MODES, generate_mock_answer
from src.retrieval import retrieve_context
from src.scoring import score_answer


DEFAULT_EVAL_DATASET_PATH = Path("data/eval_questions.csv")
DEFAULT_RESULTS_DIR = Path("results")
DEFAULT_CSV_OUTPUT_PATH = DEFAULT_RESULTS_DIR / "evaluation_results.csv"
DEFAULT_JSON_OUTPUT_PATH = DEFAULT_RESULTS_DIR / "evaluation_results.json"


def load_eval_dataset(file_path: str | Path = DEFAULT_EVAL_DATASET_PATH) -> pd.DataFrame:
    """Load evaluation questions from CSV."""
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Evaluation dataset not found: {path}")

    dataset = pd.read_csv(path)

    required_columns = {
        "id",
        "question",
        "expected_answer",
        "required_context",
    }

    missing_columns = required_columns.difference(dataset.columns)

    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Evaluation dataset is missing columns: {missing}")

    return dataset


def evaluate_question(
    row: pd.Series,
    mode: str = "grounded",
    top_k: int = 2,
) -> Dict[str, object]:
    """Run retrieval, answer generation, and scoring for one question."""
    question = str(row["question"])
    expected_answer = str(row["expected_answer"])

    retrieved_context = retrieve_context(question, top_k=top_k)
    answer = generate_mock_answer(
        question=question,
        retrieved_context=retrieved_context,
        mode=mode,
    )

    scores = score_answer(
        question=question,
        expected_answer=expected_answer,
        answer=answer,
        retrieved_context=retrieved_context,
    )

    top_context = retrieved_context[0] if retrieved_context else {}

    return {
        "id": row["id"],
        "question": question,
        "expected_answer": expected_answer,
        "required_context": row["required_context"],
        "mock_mode": mode,
        "top_retrieved_title": top_context.get("title", ""),
        "top_retrieval_score": top_context.get("score", 0),
        "generated_answer": answer,
        "answer_relevance": scores["answer_relevance"],
        "groundedness": scores["groundedness"],
        "completeness": scores["completeness"],
        "hallucination_risk": scores["hallucination_risk"],
        "overall_score": scores["overall_score"],
        "unsupported_sentences": " | ".join(scores["unsupported_sentences"]),
    }


def run_evaluation(
    dataset_path: str | Path = DEFAULT_EVAL_DATASET_PATH,
    mode: str = "grounded",
    top_k: int = 2,
) -> pd.DataFrame:
    """Run evaluation across all rows in the dataset."""
    if mode not in SUPPORTED_MODES:
        supported = ", ".join(sorted(SUPPORTED_MODES))
        raise ValueError(f"Unsupported mode: {mode}. Use one of: {supported}")

    dataset = load_eval_dataset(dataset_path)
    results: List[Dict[str, object]] = []

    for _, row in dataset.iterrows():
        result = evaluate_question(
            row=row,
            mode=mode,
            top_k=top_k,
        )
        results.append(result)

    return pd.DataFrame(results)


def save_results(
    results: pd.DataFrame,
    csv_output_path: str | Path = DEFAULT_CSV_OUTPUT_PATH,
    json_output_path: str | Path = DEFAULT_JSON_OUTPUT_PATH,
) -> None:
    """Save evaluation results to CSV and JSON."""
    csv_path = Path(csv_output_path)
    json_path = Path(json_output_path)

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    results.to_csv(csv_path, index=False)
    results.to_json(json_path, orient="records", indent=2)


def print_summary(results: pd.DataFrame) -> None:
    """Print a compact evaluation summary."""
    total_questions = len(results)
    average_score = round(results["overall_score"].mean(), 2)
    average_relevance = round(results["answer_relevance"].mean(), 2)
    average_groundedness = round(results["groundedness"].mean(), 2)
    average_completeness = round(results["completeness"].mean(), 2)

    hallucination_counts = (
        results["hallucination_risk"]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    print("Evaluation Summary")
    print("==================")
    print(f"Total questions: {total_questions}")
    print(f"Average overall score: {average_score}")
    print(f"Average relevance: {average_relevance}")
    print(f"Average groundedness: {average_groundedness}")
    print(f"Average completeness: {average_completeness}")
    print(f"Hallucination risk counts: {hallucination_counts}")
    print()

    print("Question-Level Results")
    print("======================")
    for _, row in results.iterrows():
        print(f"Q{row['id']}: {row['question']}")
        print(f"Top context: {row['top_retrieved_title']}")
        print(f"Overall score: {row['overall_score']}")
        print(f"Hallucination risk: {row['hallucination_risk']}")

        if row["unsupported_sentences"]:
            print(f"Unsupported: {row['unsupported_sentences']}")

        print()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run the LLM Evaluation Harness pipeline."
    )

    parser.add_argument(
        "--dataset",
        default=str(DEFAULT_EVAL_DATASET_PATH),
        help="Path to evaluation questions CSV.",
    )

    parser.add_argument(
        "--mode",
        default="grounded",
        choices=sorted(SUPPORTED_MODES),
        help="Mock answer generation mode.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=2,
        help="Number of retrieved context sections to use.",
    )

    parser.add_argument(
        "--csv-output",
        default=str(DEFAULT_CSV_OUTPUT_PATH),
        help="Path for CSV results output.",
    )

    parser.add_argument(
        "--json-output",
        default=str(DEFAULT_JSON_OUTPUT_PATH),
        help="Path for JSON results output.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    evaluation_results = run_evaluation(
        dataset_path=args.dataset,
        mode=args.mode,
        top_k=args.top_k,
    )

    save_results(
        results=evaluation_results,
        csv_output_path=args.csv_output,
        json_output_path=args.json_output,
    )

    print_summary(evaluation_results)

    print(f"Saved CSV results to: {args.csv_output}")
    print(f"Saved JSON results to: {args.json_output}")
