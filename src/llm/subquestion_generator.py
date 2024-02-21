import json
from typing import List, Iterable, Sequence
from enum import Enum

from instructor import OpenAISchema
from pydantic import BaseModel, Field, create_model
from .gpt import llm_call


#
DEFAULT_SUBQUESTION_GENERATOR_PROMPT = """
    You are an AI assistant that specializes in breaking down complex questions into simpler, manageable sub-questions.
    You have at your disposal a pre-defined set of functions and files to utilize in answering each sub-question.
    Please remember that your output should only contain the provided function names and file names, and that each sub-question should be a full question that can be answered using a single function and a single file.
"""

DEFAULT_USER_TASK = ""


class FunctionEnum(str, Enum):
    """The function to use to answer the questions.
    Use vector_retrieval for fact-based questions such as demographics, sports, arts and culture, etc.
    Use llm_retrieval for summarization questions, such as positive aspects, history, etc.
    """

    VECTOR_RETRIEVAL = "vector_retrieval"
    LLM_RETRIEVAL = "llm_retrieval"


def generate_subquestions(
    question,
    file_names: List[str] = None,
    system_prompt=DEFAULT_SUBQUESTION_GENERATOR_PROMPT,
    user_task=DEFAULT_USER_TASK,
    llm_model="gpt-4-0613",
):
    """Generates a list of subquestions from a user question along with the
    file name and the function to use to answer the question using OpenAI LLM.
    """
    FilenameEnum = Enum("FilenameEnum", {x.upper(): x for x in file_names})
    FilenameEnum.__doc__ = f"The names of the file to use to answer the corresponding subquestion - e.g. {file_names[0]}"

    # Create pydantic class dynamically
    QuestionBundle = create_model(
        "QuestionBundle",
        question=(
            str,
            Field(
                None, description="The subquestion extracted from the user's question"
            ),
        ),
        function=(FunctionEnum, Field(None)),
        file_name=(FilenameEnum, Field(None)),
    )

    SubQuestionBundleList = create_model(
        "SubQuestionBundleList",
        subquestion_bundle_list=(
            Sequence[QuestionBundle],
            Field(
                None,
                description="A list of subquestions - each item in the list contains a question, a function, and a file name",
            ),
        ),
        __base__=BaseModel,
    )
    SubQuestionBundleList.model_rebuild()
    user_prompt = f"{user_task}\n Here is the user question: {question}"

    few_shot_examples = [
        {
            "role": "user",
            "content": "Compare the population of Atlanta and Toronto?",
        },
        {
            "role": "function",
            "name": "SubQuestionBundleList",
            "content": """
            {
                "subquestion_bundle_list": [
                    {
                        "question": "What is the population of Atlanta?",
                        "function": "vector_retrieval",
                        "file_name": "Atlanta"
                    },
                    {
                        "question": "What is the population of Toronto?"
                        "function": "vector_retrieval",
                        "file_name": "Toronto"
                    }
                ]
            }""",
        },
        {
            "role": "user",
            "content": "Summarize the history of Chicago and Houston.",
        },
        {
            "role": "function",
            "name": "SubQuestionBundleList",
            "content": """
            {
                "subquestion_bundle_list": [
                    {
                        "question": "What is the history of Chicago?",
                        "function": "llm_retrieval",
                        "file_name": "Chicago"
                    },
                    {
                        "question": "What is the history of Houston?",
                        "function": "llm_retrieval",
                        "file_name": "Houston"
                    }
                ]
            }""",
        },
    ]

    response = llm_call(
        model=llm_model,
        response_model=Iterable[SubQuestionBundleList],
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        few_shot_examples=None,
    )

    # subquestions_list = json.loads(response.choices[0].message.function_call.arguments)
    # subquestions_pydantic_obj = SubQuestionBundleList(**subquestions_list)

    return response

