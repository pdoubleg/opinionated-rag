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
import openai

from src.utils.logging import setup_colored_logging

logger = setup_colored_logging(__name__)

client = openai.OpenAI()

FEW_SHOT_EXAMPLE = """Context: Given a user query and a search result, determine whether the search result is relevant to the user query. Answer by producing either `Yes` or `No`. 
The query pertains to legal issues within the insurance sector, emphasizing the nuances of different policies, case law, and statutes, and their applicable scenarios.

Passage: Commercial property insurance typically covers roof damage on an actual cash value basis, meaning a deduction for depreciation is applied which is non-recoverable.
Query: Is damage from a fallen tree covered under standard homeowners insurance?
Is the passage relevant the query?
Answer: No

Passage: The HO3 homeowners policy covers tree damage to structures on an open peril basis, however personal property contained inside of a building is covered for named perils only.
Query: Is damage from a fallen tree covered under standard homeowners insurance?
Is the passage relevant the query?
Answer: Yes

Passage: In 578 N.E.2d 926 the court establishes that insurers have a broad duty to defend their insured if the allegations in the underlying complaint potentially fall within the policy’s coverage.
Query: The complaint alleges that our insured's employee stole belongings from a hotel guest's room. The applicable coverage for the stolen items is unclear, and may be denied under the intentional acts exclusion. Given the coverage issue do we have a duty to defend the insured?
Is the passage relevant the query?
Answer: Yes

Passage: 363 Ill. App. 3d 335 clarifies the process for adding an entity as an additional insured under a Comprehensive General Liability (CGL) policy, and principles for filing cross-motions for summary judgment.
Query: Our insured allowed a friend to drive their rental car, which we cover under a personal auto policy. While driving the friend encountered a hail storm causing cosmetic damage to the rental car. Is the damage covered even though the friend was driving?
Is the passage relevant the query?
Answer: No
"""

ZERO_SHOT_EXAMPLE = """Given a passage and a question, predict whether the passage is relevant to the question by producing either `Yes` or `No`."""


import pandas as pd


def generate_gpt_relevance(
    query: str,
    passage: str,
    instruction: str = FEW_SHOT_EXAMPLE,
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
    prompt = f"{instruction}\nPassage: {passage}\nQuery: {query}\nIs the passage relevant the query?\nAnswer:"
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
