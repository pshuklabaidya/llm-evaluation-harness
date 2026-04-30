from src.retrieval import retrieve_context, split_markdown_sections


def test_retrieve_groundedness_as_top_result():
    results = retrieve_context("What is groundedness in LLM evaluation?", top_k=2)

    assert len(results) == 2
    assert results[0]["title"] == "Groundedness"
    assert results[0]["score"] > 0


def test_retrieve_hallucination_risk_as_top_result():
    results = retrieve_context("What does hallucination risk measure?", top_k=1)

    assert len(results) == 1
    assert results[0]["title"] == "Hallucination Risk"


def test_split_markdown_sections_extracts_level_two_sections():
    markdown_text = """
# Test Knowledge Base

## First Section

First section content.

## Second Section

Second section content.
"""

    sections = split_markdown_sections(markdown_text)

    assert len(sections) == 2
    assert sections[0]["title"] == "First Section"
    assert sections[1]["title"] == "Second Section"
