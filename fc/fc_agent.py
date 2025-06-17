# fc_agent.py

import os
import re
import time
import ast
from typing import Dict, List

from utils import generate
from fc_prompts import get_main_prompt, get_response_check_prompt
from tools import TOOL_REGISTRY, prompts as tool_prompts


class ToolCallingAgent:

    def __init__(self, course: str, model: str = "gpt-4o", seed: int = 42):
        from dotenv import load_dotenv
        load_dotenv("keys.env")
        load_dotenv("my_keys.env")
        load_dotenv(f"configs/{course}.env", override=True)

        self.model = model
        self.seed = seed
        self.course = course
        self.num_calls = 0

        # Inject prompt module into tools
        if 'ds100' in course:
            import prompts.ds100_multiturn_prompts as pmod
        elif 'ds8' in course:
            import prompts.ds8_multiturn_prompts as pmod
        elif 'cs61a' in course:
            import prompts.cs61a_multiturn_prompts as pmod
        else:
            raise ValueError(f"Unsupported course: {course}")
        tool_prompts.__dict__.update(pmod.__dict__)

    def _generate(self, messages: List[Dict[str, str]]) -> str:
        full_prompt = "\n\n".join(m["content"] for m in messages)
        return generate(prompt=full_prompt)

    def _select_tools(self, query: str) -> List[str]:
        chooser_prompt = (
            "You are a backend assistant deciding which tools to call to help answer a student question.\n"
            "Tools: qa_retrieval, textbook_retrieval, assignment_retrieval, logistics_retrieval\n"
            "Return a list of tool names, like: [\"qa_retrieval\", \"logistics_retrieval\"]"
        )
        result = self._generate([
            {"role": "system", "content": chooser_prompt},
            {"role": "user", "content": f"Student question: {query}"}
        ]).strip()
        return re.findall(r'"(.*?)"', result)

    def _select_arguments(self, query: str, tools: List[str]) -> Dict[str, Dict[str, str]]:
        prompt = (
            "Given a student question and a list of tools, return a dictionary mapping each tool to its arguments.\n"
            "Format: {'qa_retrieval': {'query': '...'}, 'logistics_retrieval': {'query': '...'}}"
        )
        result = self._generate([
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"Question: {query}\nTools: {tools}"}
        ]).strip()

        try:
            match = re.search(r"\{.*\}", result, re.DOTALL)
            return ast.literal_eval(match.group(0)) if match else {}
        except:
            return {t: {"query": query} for t in tools}

    def _generate_response(self, query: str, tool_outputs: Dict[str, str]) -> str:
        messages = get_main_prompt(query)
        for val in tool_outputs.values():
            messages.append({"role": "user", "content": val})
        return self._generate(messages)

    def _final_check(self, question: str, answer: str) -> str:
        messages = get_response_check_prompt(question, answer)
        return self._generate(messages)

    def process_query(self, query: str, deterministic=False, category=None) -> Dict[str, str]:
        self.num_calls = 0

        if deterministic:
            tools = ["qa_retrieval"]
            if category == "assignment":
                tools.append("assignment_retrieval")
            elif category == "logistics":
                tools.append("logistics_retrieval")
            elif category == "content":
                tools.append("textbook_retrieval")
            arg_dict = {t: {"query": query} for t in tools}
        else:
            tools = self._select_tools(query)
            self.num_calls += 1
            arg_dict = self._select_arguments(query, tools)
            self.num_calls += 1

        tool_outputs = {}
        for t, args in arg_dict.items():
            try:
                result = TOOL_REGISTRY[t](**args)
                if isinstance(result, tuple):
                    tool_outputs[t] = "\n".join(str(x) for x in result)
                else:
                    tool_outputs[t] = result
            except Exception as e:
                tool_outputs[t] = f"Error: {e}"

        answer_0 = self._generate_response(query, tool_outputs)
        self.num_calls += 1
        final_answer = self._final_check(query, answer_0)
        self.num_calls += 1

        return {
            "question": query,
            "answer_0": answer_0,
            "llm_answer": final_answer,
            **tool_outputs,
            "tools_called": list(arg_dict.items()),
            "num_llm_calls": self.num_calls,
            "latency": time.time() - time.time()
        }