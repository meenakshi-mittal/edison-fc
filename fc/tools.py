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

# Global prompt module will be dynamically injected by app.py via load_course_config
prompts = None


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


def assignment_retrieval(query: str) -> Tuple[str, str, str]:
    question_category = "assignment"
    category_mapping = ast.literal_eval(os.getenv("CATEGORY_MAPPING", "{}"))
    subcategory_mapping = ast.literal_eval(os.getenv("SUBCATEGORY_MAPPING", "{}"))

    problem_list, selected_doc, retrieved_docs = retrieve_docs_manual(
        question_category=question_category,
        question_subcategory=None,
        question_info=re.sub(r"\n+", " ", query),
        category_mapping=category_mapping,
        subcategory_mapping=subcategory_mapping,
        get_prompt=prompts.get_choose_problem_path_prompt,
    )
    return problem_list, selected_doc, retrieved_docs


TOOL_REGISTRY = {
    "qa_retrieval": qa_retrieval,
    "textbook_retrieval": textbook_retrieval,
    "logistics_retrieval": logistics_retrieval,
    "assignment_retrieval": assignment_retrieval,
}