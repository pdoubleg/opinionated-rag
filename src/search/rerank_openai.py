"""
This file includes the implementation of Relevance Generation and Query Generation as described in the RankGPT paper. 
For more details, refer to the paper available at: https://arxiv.org/abs/2304.09542

On calculating 'relevance':

 - If the model's response is Yes, it indicates that the passage is relevant to the query. The score is calculated by taking the 
    negative inverse of the log probability of the response, making higher probabilities result in higher relevance scores.
 
 - If the response is No, the passage is deemed not relevant, and the score is calculated similarly but takes the positive inverse to 
    distinguish it from relevant scores.
 
 - If the model's response is ambiguous or not clearly Yes or No, the function looks at the top log probabilities for both Yes and No. 
    It calculates the score based on which of the two has a higher log probability, following the same logic for relevance as above.

"""

from dotenv import load_dotenv

load_dotenv()
from math import exp
import openai

from src.utils.logging import setup_colored_logging

logger = setup_colored_logging(__name__)

client = openai.OpenAI()

FEW_SHOT_EXAMPLE = """Given a passage and a question, predict whether the passage includes an answer to the question by producing either `Yes` or `No`.

Passage: Its 25 drops per ml, you guys are all wrong. If it is water, the standard was changed 15 - 20 years ago to make 20 drops = 1mL. The viscosity of most things is temperature dependent, so this would be at room temperature. Hope this helps.
Query: how many eye drops per ml
Does the passage answer the query?
Answer: Yes

Passage: RE: How many eyedrops are there in a 10 ml bottle of Cosopt? My Kaiser pharmacy insists that 2 bottles should last me 100 days but I run out way before that time when I am using 4 drops per day.In the past other pharmacies have given me 3 10-ml bottles for 100 days.E: How many eyedrops are there in a 10 ml bottle of Cosopt? My Kaiser pharmacy insists that 2 bottles should last me 100 days but I run out way before that time when I am using 4 drops per day.
Query: how many eye drops per ml
Does the passage answer the query?
Answer: No

Passage: : You can transfer money to your checking account from other Wells Fargo. accounts through Wells Fargo Mobile Banking with the mobile app, online, at any. Wells Fargo ATM, or at a Wells Fargo branch. 1 Money in — deposits.
Query: can you open a wells fargo account online
Does the passage answer the query?
Answer: No

Passage: You can open a Wells Fargo banking account from your home or even online. It is really easy to do, provided you have all of the appropriate documentation. Wells Fargo has so many bank account options that you will be sure to find one that works for you. They offer free checking accounts with free online banking.
Query: can you open a wells fargo account online
Does the passage answer the query?
Answer: Yes
"""

ZERO_SHOT_EXAMPLE = """Given a passage and a question, predict whether the passage includes an answer to the question by producing either `Yes` or `No`."""


import pandas as pd


def generate_gpt_relevance(
    query: str,
    passage: str,
    instruction: str = ZERO_SHOT_EXAMPLE,
    model: str = "gpt-3.5-turbo-16k",
) -> float:
    """
    Reranks a given passage based on its relevance to a query using OpenAI's GPT model.

    This function sends a prompt to the OpenAI API to determine if the passage answers the given query.
    It then calculates a relevance score based on the response tokens' log probabilities.

    Args:
        query (str): The query to be answered.
        passage (str): The passage to be evaluated for relevance to the query.
        instruction (str, optional): Instruction for the model. Defaults to ZERO_SHOT_EXAMPLE.
        model (str, optional): The model to be used for the completion. Defaults to 'gpt-3.5-turbo-16k'.

    Returns:
        float: A relevance score indicating the relevance of the passage to the query.
               A higher score indicates higher relevance.
    """
    prompt = f"{instruction}\nPassage: {passage}\nQuery: {query}\nDoes the passage answer the query?\nAnswer:"
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        seed=42,
        temperature=0,
        max_tokens=2,
        logprobs=True,
        top_logprobs=5,
    )
    text = response.choices[0].message.content
    token_logprobs = response.choices[0].logprobs.content[0].logprob
    top_logprobs = response.choices[0].logprobs.content[0].top_logprobs
    top_logprobs = {word.token: word.logprob for word in top_logprobs}

    if "Yes" in text:
        logprobs = token_logprobs
        logprobs = -1 / logprobs
        rel = logprobs
    elif "No" in text:
        logprobs = token_logprobs
        logprobs = 1 / logprobs
        rel = logprobs
    else:
        if (
            " Yes" in top_logprobs
            and " No" in top_logprobs
            and top_logprobs[" Yes"] > top_logprobs[" No"]
        ):
            logprobs = top_logprobs[" Yes"]
            logprobs = -1 / logprobs
            rel = logprobs
        elif (
            " Yes" in top_logprobs
            and " No" in top_logprobs
            and top_logprobs[" Yes"] < top_logprobs[" No"]
        ):
            logprobs = top_logprobs[" No"]
            logprobs = 1 / logprobs
            rel = logprobs
        elif " Yes" in top_logprobs:
            logprobs = top_logprobs[" Yes"]
            logprobs = -1 / logprobs
            rel = logprobs
        elif " No" in top_logprobs:
            logprobs = top_logprobs[" No"]
            logprobs = 1 / logprobs
            rel = logprobs
        elif "yes" in text.lower():
            rel = 0
        else:
            rel = -1000000

    logger.info(f"Predicted 'is relevant': {text} - Score: {rel}")
    return rel, text


def apply_gpt_relevance_to_df(
    query: str,
    df: pd.DataFrame,
    text_column: str,
    score_col_name: str = "score",
    pred_col_name: str = "prediction",
) -> pd.DataFrame:
    """
    Applies the generate_gpt_relevance function to a dataframe and adds a relevance score and prediction.

    This function iterates over each row in the dataframe, applies the generate_gpt_relevance function
    to the passage specified by text_column, and adds the relevance score and prediction to new columns.
    It then sorts the dataframe based on the relevance score in descending order and returns it.

    Args:
        query (str): The query to be answered.
        df (pd.DataFrame): The dataframe containing passages to be evaluated.
        text_column (str): The column name in the dataframe that contains the passages.
        score_col_name (str): The name to be given to the column containing gpt_relevance score. Defaults to 'score'.
        pred_col_name (str): The name to be given to the column containing the prediction. Defaults to 'prediction'.

    Returns:
        pd.DataFrame: The dataframe with added columns for relevance score and prediction, sorted by relevance score in descending order.
    """
    # Apply generate_gpt_relevance and expand the returned tuple into two columns
    df[[score_col_name, pred_col_name]] = df.apply(
        lambda row: pd.Series(generate_gpt_relevance(query, row[text_column])), axis=1
    )
    return df.sort_values(by=score_col_name, ascending=False)
