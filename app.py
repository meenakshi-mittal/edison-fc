import os
import re
import ast
import logging
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from xml.sax.saxutils import unescape

from utils import (
    ocr_process_input,
    process_conversation_search,
    xml_to_markdown,
    log_blob,
    log_local,
    reply_to_ed,
    delete_comment
)
from fc.fc_agent import ToolCallingAgent

logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

app = Flask(__name__)
load_dotenv("keys.env")

def load_course_config(course: str) -> None:
    global prompts
    if 'ds100' in course:
        import prompts.ds100_multiturn_prompts as prompts
    elif 'ds8' in course:
        import prompts.ds8_multiturn_prompts as prompts
    elif 'cs61a' in course:
        import prompts.cs61a_multiturn_prompts as prompts
    else:
        raise ValueError(f"Unsupported course: {course}")
    load_dotenv(f'configs/{course}.env', override=True)

def get_env_list(key: str) -> list:
    return ast.literal_eval(os.getenv(key, '[]'))

@app.route("/", methods=["POST"])
def edison_fc():
    if request.headers.get("Authorization") != os.getenv("API_KEY"):
        logger.warning("Unauthorized access attempt")
        return jsonify(error="Unauthorized"), 401

    input_dict = request.json or {}
    logger.info("Received input: %s", input_dict)

    course = input_dict.get("course")
    if not course:
        return jsonify(error="Bad Request: No course specified"), 400
    load_course_config(course)

    # question_category = input_dict.get("category", "")
    # if not question_category:
    #     return jsonify(error="Bad Request: No category specified"), 400

    fields = ["thread_title", "category", "subcategory", "subsubcategory"]
    metadata = [xml_to_markdown(unescape(input_dict.get(f, ""))) for f in fields]
    metadata_str = " | ".join(
        f"{f}: {val}" for f, val in zip(fields, metadata) if val
    )
    # Preprocessing input
    # thread_title = xml_to_markdown(unescape(input_dict.get("thread_title", "")))
    processed_conversation = ocr_process_input(
        metadata=metadata_str,
        conversation_history=input_dict.get("conversation_history")
    )
    logger.info("Processed conversation: %s", processed_conversation)

    processed_conversation_search = process_conversation_search(
        processed_conversation=processed_conversation,
        prompt_summarize=prompts.get_summarize_conversation_prompt(processed_conversation[:-1])
    )
    logger.info("Processed conversation for search: %s", processed_conversation_search)

    query = processed_conversation_search
    agent = ToolCallingAgent(course=course, model="gpt-4o", seed=42)
    output_dict = agent.process_query(query=query)

    output_dict["processed_conversation"] = processed_conversation
    output_dict["processed_conversation_search"] = processed_conversation_search

    # Logging and posting
    prod = input_dict.get("prod") == "true"
    version = os.getenv("EDISON_VERSION")
    experiment_name = input_dict.get("experiment_name", "test")

    if input_dict.get("log_blob") == "true":
        log_path_blob = f"logs/{'production' if prod else 'test'}/{version if prod else experiment_name}.jsonl"
        log_blob({"inputs": input_dict, "outputs": output_dict}, log_path_blob)

    if input_dict.get("log_local") == "true":
        log_path_local = f"logs/{course}/{'production' if prod else 'test'}/{version if prod else experiment_name}.jsonl"
        log_local({"inputs": input_dict, "outputs": output_dict}, log_path_local)

    if input_dict.get("post_comment") == "true":
        reply_to_ed(
            course=course,
            id=input_dict.get("comment_id"),
            text="edison" + output_dict["llm_answer"],
            post_answer=False,
            private=True
        )

    return jsonify(output_dict)

@app.route("/public", methods=["POST"])
def public_edison_fc():
    if request.headers.get("Authorization") != os.getenv("API_KEY"):
        return jsonify(error="Unauthorized"), 401

    input_dict = request.json or {}
    logger.info("Received input: %s", input_dict)

    course = input_dict.get("course")
    if not course:
        return jsonify(error="Bad Request: No course specified"), 400
    load_course_config(course)

    version = os.getenv("EDISON_VERSION")
    question_id = input_dict.get("question_id", "")
    input_dict["text"] = xml_to_markdown(unescape(input_dict.get("text", "")))
    post_answer = "thread" in question_id

    delete_comment(course=course, id=input_dict.get("curr_comment_id"))
    delete_comment(course=course, id=input_dict.get("parent_comment_id"))
    input_dict.pop("curr_comment_id", None)
    input_dict.pop("parent_comment_id", None)

    if input_dict.get("log_blob") == "true":
        log_path_blob = f"logs/production/{version}_final.jsonl"
        log_blob(input_dict, log_path_blob)

    reply_to_ed(
        course=course,
        id=question_id.split("_")[-1],
        text=f"publicedison{'answer' if post_answer else 'comment'}{input_dict['text']}",
        post_answer=post_answer,
        private=False
    )
    return jsonify(message="Success")

if __name__ == "__main__":
    app.run(debug=True)
