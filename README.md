# LLM Evaluation Harness

A lightweight evaluation framework for testing LLM and RAG system outputs across relevance, groundedness, completeness, hallucination risk, and regression behavior.

## Overview

LLM Evaluation Harness provides a practical evaluation workflow for retrieval-augmented generation and language-model applications. The framework uses synthetic test questions, a controlled knowledge base, deterministic retrieval, answer generation, rubric-based scoring, and markdown reporting.

The MVP is designed to run without a paid API key through a mock LLM pipeline. Optional API-based model evaluation can be added later.

## Portfolio Focus

This repository demonstrates applied AI engineering skills in:

- LLM evaluation
- RAG evaluation
- Retrieval quality testing
- Groundedness assessment
- Hallucination detection
- Synthetic evaluation datasets
- Regression testing for AI workflows
- Streamlit-based stakeholder demos

## Core Features

- Load synthetic evaluation questions from CSV
- Retrieve relevant context from a small knowledge base
- Generate answers through a mock LLM pipeline
- Score responses using transparent rubric logic
- Flag unsupported claims and hallucination risk
- Produce a markdown evaluation report
- Support future extension to OpenAI, Anthropic, or IBM watsonx models

## Repository Structure

```text
llm-evaluation-harness/
├── app.py
├── README.md
├── requirements.txt
├── .env.example
├── data/
│   ├── eval_questions.csv
│   └── knowledge_base.md
├── results/
│   └── sample_evaluation_report.md
├── src/
│   ├── evaluator.py
│   ├── retrieval.py
│   ├── scoring.py
│   ├── mock_llm.py
│   └── report.py
└── tests/
