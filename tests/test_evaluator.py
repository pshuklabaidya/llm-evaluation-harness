from pathlib import Path

from src.evaluator import load_eval_dataset, run_evaluation


def test_load_eval_dataset_has_required_columns():
    dataset = load_eval_dataset("data/eval_questions.csv")

    required_columns = {
        "id",
        "question",
        "expected_answer",
        "required_context",
    }

    assert required_columns.issubset(dataset.columns)
    assert len(dataset) == 10


def test_run_evaluation_returns_expected_columns():
    results = run_evaluation(
        dataset_path="data/eval_questions.csv",
        mode="grounded",
        top_k=2,
    )

    expected_columns = {
        "id",
        "question",
        "expected_answer",
        "required_context",
        "mock_mode",
        "top_retrieved_title",
        "generated_answer",
        "answer_relevance",
        "groundedness",
        "completeness",
        "hallucination_risk",
        "overall_score",
        "unsupported_sentences",
    }

    assert expected_columns.issubset(results.columns)
    assert len(results) == 10
    assert results["overall_score"].mean() > 3


def test_project_data_files_exist():
    assert Path("data/eval_questions.csv").exists()
    assert Path("data/knowledge_base.md").exists()
