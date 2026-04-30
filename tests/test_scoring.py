from src.mock_llm import generate_mock_answer
from src.retrieval import retrieve_context
from src.scoring import score_answer


QUESTION = "What is groundedness in LLM evaluation?"
EXPECTED_ANSWER = (
    "Groundedness measures whether a generated answer is supported by "
    "the provided context."
)


def test_grounded_answer_scores_low_hallucination_risk():
    retrieved = retrieve_context(QUESTION, top_k=2)
    answer = generate_mock_answer(QUESTION, retrieved, mode="grounded")

    scores = score_answer(
        question=QUESTION,
        expected_answer=EXPECTED_ANSWER,
        answer=answer,
        retrieved_context=retrieved,
    )

    assert scores["hallucination_risk"] == "Low"
    assert scores["groundedness"] >= 4
    assert scores["overall_score"] >= 4


def test_hallucinated_answer_flags_unsupported_sentence():
    retrieved = retrieve_context(QUESTION, top_k=2)
    answer = generate_mock_answer(QUESTION, retrieved, mode="hallucinated")

    scores = score_answer(
        question=QUESTION,
        expected_answer=EXPECTED_ANSWER,
        answer=answer,
        retrieved_context=retrieved,
    )

    unsupported = " ".join(scores["unsupported_sentences"])

    assert "perfect production accuracy" in unsupported
    assert scores["hallucination_risk"] in {"Medium", "High"}


def test_irrelevant_answer_scores_high_hallucination_risk():
    retrieved = retrieve_context(QUESTION, top_k=2)
    answer = generate_mock_answer(QUESTION, retrieved, mode="irrelevant")

    scores = score_answer(
        question=QUESTION,
        expected_answer=EXPECTED_ANSWER,
        answer=answer,
        retrieved_context=retrieved,
    )

    assert scores["hallucination_risk"] == "High"
    assert scores["groundedness"] <= 2
    assert scores["overall_score"] < 4
