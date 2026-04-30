from pathlib import Path
from typing import List, Dict
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


DEFAULT_KNOWLEDGE_BASE_PATH = Path("data/knowledge_base.md")

GENERIC_QUERY_TERMS = {
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
}


def normalize_terms(text: str) -> List[str]:
    """Normalize text into meaningful searchable terms."""
    terms = re.findall(r"[a-zA-Z][a-zA-Z-]+", text.lower())
    cleaned_terms = []

    for term in terms:
        split_terms = term.replace("-", " ").split()

        for split_term in split_terms:
            if len(split_term) > 3 and split_term not in GENERIC_QUERY_TERMS:
                cleaned_terms.append(split_term)

    return cleaned_terms


def load_knowledge_base(file_path: str | Path = DEFAULT_KNOWLEDGE_BASE_PATH) -> str:
    """Load the markdown knowledge base as plain text."""
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Knowledge base file not found: {path}")

    return path.read_text(encoding="utf-8")


def split_markdown_sections(text: str) -> List[Dict[str, str]]:
    """Split markdown text into sections based on level-two headings."""
    sections = []
    current_title = None
    current_lines = []

    for line in text.splitlines():
        if line.startswith("## "):
            if current_title and current_lines:
                content = "\n".join(current_lines).strip()
                sections.append(
                    {
                        "title": current_title,
                        "content": content,
                        "text": f"{current_title}\n{content}",
                    }
                )

            current_title = line.replace("## ", "").strip()
            current_lines = []
        else:
            if current_title:
                current_lines.append(line)

    if current_title and current_lines:
        content = "\n".join(current_lines).strip()
        sections.append(
            {
                "title": current_title,
                "content": content,
                "text": f"{current_title}\n{content}",
            }
        )

    return sections


def calculate_keyword_boost(query: str, title: str, content: str) -> float:
    """Boost exact topic matches while ignoring generic evaluation vocabulary."""
    query_terms = set(normalize_terms(query))
    title_terms = set(normalize_terms(title))
    content_terms = set(normalize_terms(content))

    if not query_terms:
        return 0.0

    exact_title_matches = query_terms.intersection(title_terms)
    content_matches = query_terms.intersection(content_terms)

    title_boost = 1.0 * len(exact_title_matches)
    content_boost = 0.05 * len(content_matches)

    return title_boost + content_boost


def retrieve_context(
    query: str,
    knowledge_base_path: str | Path = DEFAULT_KNOWLEDGE_BASE_PATH,
    top_k: int = 2,
) -> List[Dict[str, object]]:
    """Retrieve the most relevant knowledge-base sections for a query."""
    if not query or not query.strip():
        raise ValueError("Query must be a non-empty string.")

    knowledge_base_text = load_knowledge_base(knowledge_base_path)
    sections = split_markdown_sections(knowledge_base_text)

    if not sections:
        raise ValueError("No retrievable sections found in the knowledge base.")

    documents = [section["text"] for section in sections]

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
    )
    document_vectors = vectorizer.fit_transform(documents)
    query_vector = vectorizer.transform([query])

    similarities = cosine_similarity(query_vector, document_vectors).flatten()

    scored_results = []

    for index, section in enumerate(sections):
        base_score = float(similarities[index])
        keyword_boost = calculate_keyword_boost(
            query=query,
            title=section["title"],
            content=section["content"],
        )
        final_score = base_score + keyword_boost

        scored_results.append(
            {
                "title": section["title"],
                "content": section["content"],
                "base_score": round(base_score, 4),
                "keyword_boost": round(keyword_boost, 4),
                "score": round(final_score, 4),
            }
        )

    ranked_results = sorted(
        scored_results,
        key=lambda item: item["score"],
        reverse=True,
    )[:top_k]

    results = []

    for rank, result in enumerate(ranked_results, start=1):
        results.append(
            {
                "rank": rank,
                "title": result["title"],
                "content": result["content"],
                "base_score": result["base_score"],
                "keyword_boost": result["keyword_boost"],
                "score": result["score"],
            }
        )

    return results


def format_retrieved_context(results: List[Dict[str, object]]) -> str:
    """Format retrieved sections into a readable context block."""
    formatted_sections = []

    for result in results:
        formatted_sections.append(
            f"[Rank {result['rank']}] {result['title']} "
            f"(score: {result['score']}, base: {result['base_score']}, "
            f"boost: {result['keyword_boost']})\n{result['content']}"
        )

    return "\n\n".join(formatted_sections)


if __name__ == "__main__":
    sample_query = "What is groundedness in LLM evaluation?"
    retrieved_results = retrieve_context(sample_query, top_k=2)

    print(f"Query: {sample_query}\n")
    print(format_retrieved_context(retrieved_results))
