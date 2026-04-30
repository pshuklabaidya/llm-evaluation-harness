import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


GENERIC_TERMS = {
    "what",
    "does",
    "measure",
    "measures",
    "why",
    "used",
    "useful",
    "important",
    "evaluation",
    "evaluate",
    "evaluating",
    "llm",
    "rag",
    "answer",
    "answers",
    "question",
    "questions",
    "response",
    "responses",
    "system",
    "systems",
    "method",
    "methods",
    "data",
}


def normalize_text(text: str) -> str:
    """Normalize text for simple lexical comparison."""
    return re.sub(r"\s+", " ", text.lower()).strip()


def extract_terms(text: str) -> set[str]:
    """Extract meaningful terms while ignoring generic evaluation vocabulary."""
    terms = re.findall(r"[a-zA-Z][a-zA-Z-]+", text.lower())
    cleaned_terms = set()

    for term in terms:
        for split_term in term.replace("-", " ").split():
            if len(split_term) > 3 and split_term not in GENERIC_TERMS:
                cleaned_terms.add(split_term)

    return cleaned_terms


def score_from_ratio(ratio: float) -> int:
    """Convert an overlap ratio into a 1-5 score."""
    if ratio >= 0.85:
        return 5
    if ratio >= 0.65:
        return 4
    if ratio >= 0.45:
        return 3
    if ratio >= 0.25:
        return 2
    return 1


def calculate_overlap_ratio(reference_text: str, candidate_text: str) -> float:
    """Calculate meaningful term overlap between reference and candidate text."""
    reference_terms = extract_terms(reference_text)
    candidate_terms = extract_terms(candidate_text)

    if not reference_terms:
        return 0.0

    matches = reference_terms.intersection(candidate_terms)
    return len(matches) / len(reference_terms)


def score_relevance(question: str, expected_answer: str, answer: str) -> int:
    """Score whether the answer addresses the question and expected topic."""
    question_overlap = calculate_overlap_ratio(question, answer)
    expected_overlap = calculate_overlap_ratio(expected_answer, answer)

    combined_ratio = (question_overlap * 0.35) + (expected_overlap * 0.65)
    return score_from_ratio(combined_ratio)


def score_completeness(expected_answer: str, answer: str) -> int:
    """Score whether the answer contains the essential expected information."""
    overlap_ratio = calculate_overlap_ratio(expected_answer, answer)
    return score_from_ratio(overlap_ratio)


def get_context_text(retrieved_context: List[Dict[str, object]]) -> str:
    """Join retrieved context sections into one context string."""
    return " ".join(str(item.get("content", "")) for item in retrieved_context)


def split_sentences(text: str) -> List[str]:
    """Split text into simple sentence-like units."""
    pieces = re.split(r"(?<=[.!?])\s+", text.strip())
    return [piece.strip() for piece in pieces if piece.strip()]


def find_unsupported_sentences(
    answer: str,
    retrieved_context: List[Dict[str, object]],
    minimum_overlap: float = 0.35,
) -> List[str]:
    """Find answer sentences weakly supported by retrieved context."""
    context_text = get_context_text(retrieved_context)
    unsupported = []

    for sentence in split_sentences(answer):
        sentence_terms = extract_terms(sentence)

        if not sentence_terms:
            continue

        overlap_ratio = calculate_overlap_ratio(sentence, context_text)

        if overlap_ratio < minimum_overlap:
            unsupported.append(sentence)

    return unsupported


def score_groundedness(
    answer: str,
    retrieved_context: List[Dict[str, object]],
) -> Tuple[int, List[str]]:
    """Score whether answer claims are supported by retrieved context."""
    unsupported_sentences = find_unsupported_sentences(answer, retrieved_context)

    total_sentences = max(len(split_sentences(answer)), 1)
    unsupported_ratio = len(unsupported_sentences) / total_sentences

    if unsupported_ratio == 0:
        return 5, unsupported_sentences
    if unsupported_ratio <= 0.25:
        return 4, unsupported_sentences
    if unsupported_ratio <= 0.5:
        return 3, unsupported_sentences
    if unsupported_ratio <= 0.75:
        return 2, unsupported_sentences

    return 1, unsupported_sentences


def classify_hallucination_risk(
    groundedness_score: int,
    unsupported_sentences: List[str],
) -> str:
    """Classify hallucination risk from groundedness and unsupported claims."""
    if groundedness_score >= 5 and not unsupported_sentences:
        return "Low"

    if groundedness_score >= 3:
        return "Medium"

    return "High"


def calculate_overall_score(
    relevance: int,
    groundedness: int,
    completeness: int,
) -> float:
    """Calculate weighted overall quality score."""
    overall = (
        (relevance * 0.30)
        + (groundedness * 0.45)
        + (completeness * 0.25)
    )

    return round(overall, 2)


def score_answer(
    question: str,
    expected_answer: str,
    answer: str,
    retrieved_context: List[Dict[str, object]],
) -> Dict[str, object]:
    """Score a generated answer across core evaluation dimensions."""
    relevance = score_relevance(question, expected_answer, answer)
    completeness = score_completeness(expected_answer, answer)
    groundedness, unsupported_sentences = score_groundedness(
        answer,
        retrieved_context,
    )
    hallucination_risk = classify_hallucination_risk(
        groundedness,
        unsupported_sentences,
    )
    overall_score = calculate_overall_score(
        relevance,
        groundedness,
        completeness,
    )

    return {
        "answer_relevance": relevance,
        "groundedness": groundedness,
        "completeness": completeness,
        "hallucination_risk": hallucination_risk,
        "overall_score": overall_score,
        "unsupported_sentences": unsupported_sentences,
    }


if __name__ == "__main__":
    from src.retrieval import retrieve_context
    from src.mock_llm import generate_mock_answer

    question = "What is groundedness in LLM evaluation?"
    expected_answer = (
        "Groundedness measures whether a generated answer is supported by "
        "the provided context."
    )

    retrieved = retrieve_context(question, top_k=2)

    for mode in ["grounded", "incomplete", "hallucinated", "irrelevant"]:
        answer = generate_mock_answer(question, retrieved, mode=mode)
        scores = score_answer(
            question=question,
            expected_answer=expected_answer,
            answer=answer,
            retrieved_context=retrieved,
        )

        print(f"Mode: {mode}")
        print(f"Answer: {answer}")
        print(f"Scores: {scores}")
        print()
