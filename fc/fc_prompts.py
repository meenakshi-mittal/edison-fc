# from azure.ai.inference.models import SystemMessage, UserMessage

response_check = (
    "Given a student's question and a potential answer, please modify the answer according to the following guidelines: \n"
    "1. Revise the answer to make it more concise. \n"
    "2. Remove any solutions provided in the original answer, leaving only hints and guiding explanations. \n"
    "3. Encourage the student to ask follow-up questions if they need further clarification. \n"
    "4. Output only the revised answer, without a label or any additional text."
)


def get_response_check_prompt(processed_conversation: str, first_answer: str) -> list:
    curr_prompt = f"""Student's question:
==========================================
{processed_conversation}
==========================================

Answer:
==========================================
{first_answer}
=========================================="""
    return [
        {"role": "system", "content": response_check},
        {"role": "user", "content": curr_prompt},
    ]


categorizer = (
    "You are a classifier that labels student questions into one or more of the following categories:\n"
    "-logistics: if the question is about deadlines, policies, exam times, etc.\n"
    "-assignment: if the question refers to a specific homework, project, or question solution.\n"
    "-conceptual: if the question is about a technical topic or theory from the course.\n"
    "-general: if the question is ambiguous or does not fall into the above categories.\n"
    "Respond with a single word or a list of the following categories: logistics, assignment, concept, or general. Do NOT include an explanation."
)


def get_categorizer_prompt(processed_conversation: str) -> list:
    curr_prompt = f"""Student's question:
==========================================
{processed_conversation}
==========================================
"""
    return [
        {"role": "system", "content": categorizer},
        {"role": "user", "content": curr_prompt},
    ]


main_prompt = (
    "You will simulate the role of a teaching assistant for an undergraduate data science course, answering student questions on a course discussion forum. "
    "1. Your responses should be clear, helpful, and maintain a positive tone. \n"
    "2. Aim for conciseness and clarity. \n"
    "3. Use the excerpts from any solutions, course notes, and historical question-answer pairs provided to you as your primary source of information. \n"
    "4. If the question is difficult to answer based on the provided context, reply, 'Sorry, I do not know. Please wait for a staff member's response.'\n\n"
)


def get_main_prompt(processed_conversation: str) -> list:
    curr_prompt = f"""Student's question:
==========================================
{processed_conversation}
==========================================
"""
    return [
        {"role": "system", "content": main_prompt},
        {"role": "user", "content": curr_prompt},
    ]


judge_few_shot = """
Student Question: Homework Logistics

Who is the best person to contact regarding my grade on a previous homework?

LLM Answer: Contact your course instructor or assigned TA for any grading concerns. Feel free to ask if you need more help.
Ground Truth Answer: For inquiries regarding your grade on a previous homework, please reach out to us privately via Ed or email at data100.instructors@berkeley.edu.
Scores: {"factuality": 5, "relevance": 5, "style": 3}

Student Question: q4a returning empty table

%%sql -- save query_q4a
SELECT
AVG(CAST (averageRating AS FLOAT)) AS averageRating,
COUNT(*) AS total,
ROUND( CAST( runtimeMinutes AS INT)/ 10.0 + 0.5) * 10 AS runtimeBin
FROM rating
INNER JOIN title
ON title.tconst = rating.tconst
WHERE titleType = 'movies' AND CAST(rating. numVotes AS INT) >= 25000
GROUP BY runtimeBin
ORDER BY runtimeBin;
Running query in 'duckdb:///data/imdbmini.db'
averageRating total runtimeBin

My code is returning an empty table with the correct columns. I'm not quite sure why that is?

LLM Answer: It appears there might be a small issue with the value you're using in your `WHERE` clause for the `titleType`. In the database schema, `titleType` values are case-sensitive. Please verify that the value you're using for `titleType` in the `WHERE` clause matches exactly how it's spelled in the database, including the case.

Additionally, ensure that you are connecting to the correct database and have access to the updated data that meets your filtering criteria (`numVotes >= 25000`).

Revisiting these two points and correcting any discrepancies should help you retrieve the expected results from your query.
Ground Truth Answer: Try doing 'movie'
Scores: {"factuality": 5, "relevance": 3, "style": 3}

Student Question: Lecture 24 participation

Survey
793 g
Thank you for your response!
Edit response
Lecture Lecture 23 Attendance (1/1)
Lecture Lecture 23 Attendance
Score: 1
Lecture Lecture 24 Attendance (0/1)
Lecture Lecture 24 Attendance
Score: 0

Hi I filled out the attendance slido for lecture 24 but did not get the credit. Is there anything wrong?

LLM Answer: Make sure you submitted the attendance response during the live lecture (11:00 am-12:30 pm on Tuesdays and Thursdays) for credit. Verify your response was recorded correctly. If the problem continues, contact a course staff member for help.

Need further assistance?
Ground Truth Answer: Hi Matthew,

Please fill out the autograder regrade request form and we will get back to you as soon as we can

Scores: {"factuality": 4, "relevance": 5, "style": 3}
"""

judge_prompt = """
You are an expert at grading responses to student questions. You are given:
- a student question
- an LLM-written answer
- a TA-written ground-truth answer.
Assign a score from 1 to 5 for factuality and relevance:
1. Factuality: Evaluates the correctness of the information provided in the LLM-written response.
2. Relevance:  Evaluates the degree to which the LLM-written response is pertinent or related to the given student question and course.
And assign a score from 1 to 3 for style:
1. Style: Evaluates the degree to which the coherence, length, and the use of solutions, hints, and examples in the LLM-written response are appropriate for the given student question.
Please refer to the Ground Truth answer as the gold standard for all of the metrics.
Respond ONLY in dictionary format like this:
{"factuality": <1-5>, "relevance": <1-5>, "style": <1-3>}
Do NOT use the tag "json" in the response, or any backticks.
You are a kind grader. If you are ever deciding between 2 scores, choose the higher one.
"""

verbose_judge_prompt = """
You are an expert at grading responses to student questions. You are given:
- a student question
- an LLM-written answer
- a TA-written ground-truth answer.

Assign a score from 1 to 5 for factuality and relevance:

1. Factuality: Evaluates the correctness of the information provided in the LLM-written response.
2. Relevance:  Evaluates the degree to which the LLM-written response is pertinent or related to the given student question and course.
    - An answer is relevant if it contains the main points given in the TA-written answer
    - An answer is not relevant if it appears to incorporate information that is out of scope of the question or the course.

And assign a score from 1 to 3 for style:

1. Style: Evaluates the degree to which the coherence, length, and the use of solutions, hints, and examples in the LLM-written response are appropriate for the given student question.
    Some factors of bad style include:
    - Excessive verbosity: the length of the response makes it difficult to understand
    - "Shot-in-the-dark" answers: providing multiple vague explanations due to uncertainty, rather than addressing one main point
    - Revealing full or partial worked solutions to assignment questions

Please refer to the Ground Truth answer as the gold standard for all of the metrics.
Respond ONLY in dictionary format like this:
{"factuality": <1-5>, "relevance": <1-5>, "style": <1-3>}
Do NOT use the tag "json" in the response, or any backticks.
You are a kind grader. If you are ever deciding between 2 scores, choose the higher one.
"""

feedback_prompt = """
You are evaluating an assistant’s answer to a student question.

If the answer fully and accurately responds to the question, respond ONLY with: True

Otherwise provide concise feedback on how to improve the answer in terms of:
- factual correctness
- relevance to the question
- clarity, coherence, and helpfulness (style)

Additionally, the assistant can use the following tools:
- qa_retrieval: retrieves similar questions and past TA answers
- textbook_retrieval: retrieves content from course notes or textbooks
- assignment_retrieval: retrieves assignment questions and solutions
- logistics_retrieval: retrieves course policies, schedules, and deadlines

You are provided with which tools were already used. If any part of the response is vague or underdeveloped, suggest which tool(s) could be used to generate a more specific answer.
Keep in mind that responses to assignment questions should not have overly explicit solutions provided, so do not mark this down.

Keep your feedback very short and focused. Do not summarize the assistant’s original answer.
"""


# def get_feedback_prompt(query, response, tools_called) -> list:
#     curr_prompt = f"""Student's question:
# ==========================================
# {query}
# ==========================================
#
# Assistant Response:
# ==========================================
# {response}
# ==========================================
#
# Tools Called:
# ==========================================
# {tools_called}
# ==========================================
# """
#     # return [
#     #     {"role": "system", "content": feedback_prompt},
#     #     {"role": "user", "content": curr_prompt},
#     # ]
#     return [
#         SystemMessage(content=feedback_prompt),
#         UserMessage(content=curr_prompt)
#     ]