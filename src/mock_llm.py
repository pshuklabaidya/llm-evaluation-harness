from pathlib import Path
import sys
from typing import Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.retrieval import retrieve_context


SUPPORTED_MODES = {
    "grounded",
    "incomplete",
    "hallucinated",
    "irrelevant",
}


def split_sentences(text: str) -> List[str]:
    """Split simple prose into sentence-like units."""
    sentences = []

    for part in text.replace("\n", " ").split("."):
        cleaned = part.strip()

        if cleaned:
            sentences.append(f"{cleaned}.")

    return sentences


def generate_mock_answer(
    question: str,
    retrieved_context: List[Dict[str, object]],
    mode: str = "grounded",
) -> str:
    """
    Generate a deterministic mock answer from retrieved context.

    Modes:
    - grounded: uses the full top retrieved context
    - incomplete: uses only the first sentence
    - hallucinated: adds an unsupported claim
    - irrelevant: ignores the retrieved context
    """
    if mode not in SUPPORTED_MODES:
        supported = ", ".join(sorted(SUPPORTED_MODES))
        raise ValueError(f"Unsupported mode: {mode}. Use one of: {supported}")

    if not question or not question.strip():
        raise ValueError("Question must be a non-empty string.")

    if not retrieved_context:
        raise ValueError("Retrieved context must contain at least one result.")

    top_context = str(retrieved_context[0]["content"]).strip()
    sentences = split_sentences(top_context)

    if mode == "grounded":
        return top_context

    if mode == "incomplete":
        return sentences[0] if sentences else top_context

    if mode == "hallucinated":
        return (
            f"{top_context} This evaluation method guarantees perfect production "
            "accuracy for every LLM system."
        )

    return (
        "LLM systems are useful for many business workflows, including automation, "
        "analytics, and customer support."
    )


if __name__ == "__main__":
    sample_question = "What is groundedness in LLM evaluation?"
    retrieved_results = retrieve_context(sample_question, top_k=2)

    print("Question:")
    print(sample_question)
    print()

    print("Grounded mock answer:")
    print(generate_mock_answer(sample_question, retrieved_results, mode="grounded"))
    print()

    print("Incomplete mock answer:")
    print(generate_mock_answer(sample_question, retrieved_results, mode="incomplete"))
    print()

    print("Hallucinated mock answer:")
    print(generate_mock_answer(sample_question, retrieved_results, mode="hallucinated"))
    print()

    print("Irrelevant mock answer:")
    print(generate_mock_answer(sample_question, retrieved_results, mode="irrelevant"))
