import os
import json
import time

import pandas as pd
from utils import generate
from VQA import RelevanceChecker
# from tree_utils import create_TOC
import dotenv
from openai import AzureOpenAI

dotenv.load_dotenv('../keys.env')

client = AzureOpenAI(
    api_key=os.getenv("OPENAI_KEY"),
    api_version="2024-06-01",
    azure_endpoint=os.getenv("LLM_ENDPOINT")
)
model = os.getenv("MODEL_NAME")

file_cache = {}
gpt_call_count = 0

# def safe_generate(messages, temperature=0.1):
#     """
#     Calls the generate function inside a loop until it succeeds.
#     If an error occurs, it waits 1 second and retries.
#     """
#     global gpt_call_count
#     while True:
#         try:
#             output = generate(messages, temperature=temperature)
#             gpt_call_count += 1
#             return output
#         except Exception as e:
#             print(f"Error calling generate: {e}. Retrying in 1 second...")
#             time.sleep(1)

def safe_generate(messages, temperature=0.1):
    global gpt_call_count
    while True:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
            )
            gpt_call_count += 1
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error calling generate: {e}. Retrying in 1 second...")
            time.sleep(1)

def get_relevant_files(question, toc):
    """
    Uses GPT (via the generate function) to select three file names from the table of contents.
    The toc is assumed to be a simple dict mapping file names to their root key (or summary).

    The prompt asks GPT to output a JSON list of file names.
    """
    prompt = (
        f"Given the student question: \"{question}\", and the following table of contents:\n"
        f"{json.dumps(toc, indent=2)}\n"
        "Select the three file names whose content is most likely to answer the question. "
        "Output ONLY a list of exactly three file names, like so: ['hw4.json', 'lab8.json', 'projA1.json'] "
    )
    messages = [{"role": "system", "content": prompt}]
    gpt_output = safe_generate(messages, temperature=0.1)

    print(gpt_output)
    try:
        file_list = eval(gpt_output)
        if isinstance(file_list, list) and len(file_list) == 3:
            return file_list
        else:
            print("Warning: GPT did not return exactly three file names. Using first three keys from TOC.")
            return list(toc.keys())[:3]
    except Exception as e:
        print(f"Error parsing GPT output: {e}. Falling back to first three keys from TOC.")
        return list(toc.keys())[:3]


def beam_search_across_files(question, file_names, relevance_checker, beam_width=3, final_doc_count=None,
                             file_tree_folder="balanced_binary_tree",
                             use_gpt_scoring=False):
    """
    Performs beam search concurrently across the document trees from the given file names.
    When use_gpt_scoring is False, each candidate node is scored using the relevance_checker.
    When use_gpt_scoring is True, expandable node keys are gathered and a GPT call is made
    to select the best candidates.

    The additional parameter final_doc_count lets you decide how many document candidates should be
    returned at the bottommost layer. For example, you can search with a beam width of 3 throughout the tree,
    and then at the bottom only pick the best 1.

    Returns:
        list: A list of tuples (score, file_name, node_key, node_value) for the final beam candidates.
    """
    if final_doc_count is None:
        final_doc_count = beam_width

    beam = []
    for file_name in file_names:
        file_path = os.path.join(file_tree_folder, file_name)
        if os.path.exists(file_path):
            try:
                if file_name not in file_cache:
                    with open(file_path, "r", encoding="utf-8") as f:
                        file_tree = json.load(f)
                    file_cache[file_name] = file_tree
                else:
                    file_tree = file_cache[file_name]
                root_key = list(file_tree.keys())[0]
                root_value = file_tree[root_key]
                if not use_gpt_scoring:
                    time.sleep(1)
                    score = relevance_checker.check_relevance(question, root_key).item()
                else:
                    score = None
                beam.append((score, file_name, root_key, root_value))
            except Exception as e:
                print(f"Error loading file {file_name}: {e}")
        else:
            print(f"File {file_path} not found.")

    if not beam:
        return []

    while True:
        new_beam = []
        expanded = False

        if not use_gpt_scoring:
            for score, file_name, node_key, node in beam:
                if isinstance(node, dict):
                    for child_key, child_value in node.items():
                        time.sleep(0.5)
                        child_score = relevance_checker.check_relevance(question, child_key).item()
                        new_beam.append((child_score, file_name, child_key, child_value))
                    expanded = True
                else:
                    new_beam.append((score, file_name, node_key, node))
            if not expanded or not new_beam:
                break
            new_beam.sort(key=lambda x: x[0], reverse=True)
            beam = new_beam[:beam_width]
        else:
            candidate_details = {}
            candidate_display = {}
            leaf_candidates = []
            for candidate in beam:
                score, file_name, node_key, node = candidate
                if isinstance(node, dict):
                    for child_key, child_value in node.items():
                        candidate_index = len(candidate_details) + 1
                        candidate_details[candidate_index] = (file_name, child_key, child_value)
                        candidate_display[candidate_index] = child_key
                    expanded = True
                else:
                    leaf_candidates.append(candidate)
            if not expanded or not candidate_details:
                break

            # print(json.dumps(candidate_display, indent=2))

            if all(not isinstance(child_value, dict) for (file_name, child_key, child_value) in candidate_details.values()):
                num_to_select = final_doc_count
            else:
                num_to_select = beam_width

            prompt = (
                "You are an expert at relevant document selection. "
                f"Here is a student question:\n\"{question}\"\n"
                f"And here is a list of candidate document keys:\n"
                f"{json.dumps(candidate_display, indent=2)}\n"
                f"Select the top {num_to_select} most relevant document keys for answering the question. "
                "Output ONLY a list of the keys as natural numbers, e.g. [1, 3, 5]."
            )
            messages = [{"role": "system", "content": prompt}]
            time.sleep(1)
            gpt_output = safe_generate(messages, temperature=0.1)
            # print("GPT output for alternate scoring:", gpt_output)
            try:
                selected_keys = eval(gpt_output)
                if not isinstance(selected_keys, list) or len(selected_keys) == 0:
                    raise ValueError("GPT did not return a valid list")
            except Exception as e:
                print(f"Error parsing GPT output in alternate scoring: {e}. Using first {num_to_select} keys.")
                selected_keys = list(candidate_details.keys())[:num_to_select]

            for key in selected_keys:
                if key in candidate_details:
                    file_name, child_key, child_value = candidate_details[key]
                    new_beam.append((None, file_name, child_key, child_value))
            new_beam.extend(leaf_candidates)
            beam = new_beam

        if not use_gpt_scoring:
            beam.sort(key=lambda x: x[0], reverse=True)
            beam = beam[:final_doc_count]

    output = []
    for i in range(len(beam)):
        score, file_name, _, text = beam[i]
        if score:
            output.append({"Relevance": score, "File": file_name, "Text": text})
        else:
            output.append({"File": file_name, "Text": text})
    return output


def manual_retrieval(file_tree_folder, question, beam_width=3, final_doc_count=1, use_gpt_scoring=True):
    global gpt_call_count
    gpt_call_count = 0
    try:
        with open(f"{file_tree_folder}/table_of_contents.json", "r", encoding="utf-8") as f:
            toc = json.load(f)
    except Exception as e:
        print(f"Error loading table_of_contents.json: {e}")
        return

    selected_files = get_relevant_files(question, toc)
    # print("Selected files:", selected_files)

    api_key = os.getenv('OPENAI_KEY')
    model_name = os.getenv('MODEL_NAME')
    endpoint = os.getenv('LLM_ENDPOINT')

    relevance_checker = RelevanceChecker(api_key, endpoint, model_name)

    docs = beam_search_across_files(
        question,
        selected_files,
        relevance_checker,
        beam_width=beam_width,
        final_doc_count=final_doc_count,
        file_tree_folder=file_tree_folder,
        use_gpt_scoring=use_gpt_scoring
    )
    return docs, gpt_call_count


def run_test(input_path, file_tree_folder, outpath, beam_width=3, use_gpt_scoring=True, final_doc_count=None):
    """
    Main function that processes questions from an Excel file, selects relevant files, and then uses beam search
    across the document trees. The final_doc_count parameter allows you to control how many documents are returned
    from the bottommost layer.
    """
    # create_TOC(file_tree_folder)

    try:
        df = pd.read_excel(input_path)
    except Exception as e:
        print(f"Error loading Excel file: {e}")
        return

    try:
        with open(f"{file_tree_folder}/table_of_contents.json", "r", encoding="utf-8") as f:
            toc = json.load(f)
    except Exception as e:
        print(f"Error loading table_of_contents.json: {e}")
        return

    api_key = os.getenv('OPENAI_KEY')
    model_name = os.getenv('MODEL_NAME')
    endpoint = os.getenv('LLM_ENDPOINT')

    relevance_checker = RelevanceChecker(api_key, endpoint, model_name)

    output_data = []
    for idx, row in df.iterrows():
        question = row["Question"]

        docs, gpt_call_count = manual_retrieval(
            file_tree_folder,
            question,
            beam_width=beam_width,
            final_doc_count=final_doc_count,
            use_gpt_scoring=use_gpt_scoring
        )

        print(gpt_call_count)

        row_data = row.to_dict()
        for i in range(len(docs)):
            row_data[f"Doc {i+1}"] = docs[i]
        # print(row_data)

        output_data.append(row_data)

    output_df = pd.DataFrame(output_data)
    try:
        output_df.to_excel(outpath, index=False)
        print(f"Processed questions saved to {outpath}")
    except Exception as e:
        print(f"Error writing output Excel file: {e}")


if __name__ == "__main__":
    file_tree_folder = "data100_fa24/balanced_header_trinary_tree"
    beam_width = 3
    final_doc_count = 5
    use_gpt_scoring = True
    inpath = "../inputs/edison_retrieval.xlsx"
    outpath = "../outputs/gpt_choose_tree_outputs/balanced_header_trinary_tree_ask_gpt_top5.xlsx"
    run_test(inpath, file_tree_folder, outpath, beam_width, use_gpt_scoring, final_doc_count)
