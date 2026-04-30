# LLM Evaluation Knowledge Base

## Groundedness

Groundedness measures whether a generated answer is supported by the provided context. A grounded answer stays close to retrieved evidence and avoids claims that cannot be verified from the available material.

## Hallucination Risk

Hallucination risk measures the likelihood that an answer contains invented, unsupported, or unverifiable claims. A high hallucination-risk answer may sound plausible while adding details not present in the source context.

## Answer Relevance

Answer relevance measures whether a response directly addresses the user question. A relevant answer should stay focused on the question rather than shifting to unrelated topics.

## Completeness

Completeness measures whether an answer covers the main points required by the expected answer. A complete answer does not need to be long, but it should include the essential information.

## Context Usefulness

Context usefulness measures whether retrieved context contains information needed to answer the question. Poor retrieval can cause weak answers even when the answer generator behaves correctly.

## Regression Testing

Regression testing compares evaluation results across multiple runs or versions. It helps identify whether changes to prompts, retrieval settings, or model behavior improved or degraded system quality.

## Rule-Based Evaluation

Rule-based evaluation uses transparent scoring logic instead of hidden model judgment. It is useful for portfolio projects because scoring behavior can be inspected, explained, and tested.

## LLM-As-Judge Evaluation

LLM-as-judge evaluation uses a language model to score outputs. It can capture nuance, but it also introduces cost, variability, and potential judge-model bias.

## Synthetic Evaluation Data

Synthetic evaluation data is artificially created data used for safe testing and demonstration. It avoids exposing private customer data, proprietary documents, or confidential business records.

## RAG Evaluation

RAG evaluation measures both retrieval and generation quality. A strong RAG answer should retrieve useful context, answer the question, and avoid unsupported claims.
