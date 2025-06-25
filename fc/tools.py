# tools.py

import os
import re
import ast
from typing import Tuple

from utils import (
    retrieve_qa,
    retrieve_docs_hybrid,
    retrieve_docs_manual,
    process_conversation_search,
    ocr_process_input
)

from manual_retrieval.tree_retrieval import manual_retrieval

# Global prompt module will be dynamically injected by app.py via load_course_config
# prompts = None


def qa_retrieval(query: str) -> str:
    return retrieve_qa(conversation=query, top_k=int(os.getenv("QA_TOP_K", "3")))


def textbook_retrieval(query: str) -> str:
    return retrieve_docs_hybrid(
        text=query,
        index_name=os.getenv("CONTENT_INDEX_NAME"),
        top_k=int(os.getenv("CONTENT_INDEX_TOP_K", "1")),
        semantic_reranking=True,
    )


def logistics_retrieval(query: str) -> str:
    return retrieve_docs_hybrid(
        text=query,
        index_name=os.getenv("LOGISTICS_INDEX_NAME"),
        top_k=int(os.getenv("LOGISTICS_INDEX_TOP_K", "1")),
        semantic_reranking=False,
    )


def assignment_retrieval(query: str):
    docs, gpt_num_called = manual_retrieval(query)
    return docs


TOOL_DEFS = {
  "qa_retrieval": "For retrieving similar questions and TA answers from past semesters.",
  "textbook_retrieval": "For retrieving course content from textbooks or course materials.",
  "assignment_retrieval": "For retrieving assignment materials and solutions.",
  "logistics_retrieval": "For retrieving course logistics such as deadlines, policies, and schedules."
}

TOOLS_ARGS = {
  "qa_retrieval": "Inputs a single argument \"query\".",
  "textbook_retrieval": "Inputs a single argument \"query\".",
  "assignment_retrieval": "Inputs a single argument \"query\".",
  "logistics_retrieval": "Inputs a single argument \"query\"."
}



TOOL_REGISTRY = {
    "qa_retrieval": qa_retrieval,
    "textbook_retrieval": textbook_retrieval,
    "logistics_retrieval": logistics_retrieval,
    "assignment_retrieval": assignment_retrieval,
}