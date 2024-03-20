from dotenv import load_dotenv
load_dotenv()
from math import exp
import openai
import pandas as pd
from tenacity import retry, wait_random_exponential, stop_after_attempt
import tiktoken

def rerank_with_openai(df: pd.DataFrame,
                       query: str, 
                       text_col_name: str = "text",
                       model_name: str = 'gpt-3.5-turbo-16k',
    ) -> pd.DataFrame:
    # Get encodings for our target classes
    tokens = [" Yes", " No"]
    tokenizer = tiktoken.encoding_for_model(model_name)
    ids = [tokenizer.encode(token) for token in tokens]
    
    system_message = """
    You are an Assistant responsible for helping detect whether the retrieved document is relevant to the query. \
    For a given input, you need to output a single token: "Yes" or "No" indicating the retrieved document is relevant to the query.
    """

    input_prompt = '''
    Query: What are the legal considerations for insurance coverage in cases of professional negligence for architects and engineers?
    Document: """The policy outlines the scope of insurance coverage available to architects and engineers in the event of claims made against them for professional negligence. It specifies the conditions under which the insurer will indemnify the insured against liability incurred as a result of errors, omissions, or negligent acts in the performance of professional services. Key exclusions include claims arising from willful misconduct, known liabilities prior to the policy inception, and contractual liabilities exceeding the scope of standard professional duties. The document further details the process for filing a claim, the insurer's rights to defend or settle claims, and the duty of the insured to cooperate with the insurer in the claims process."""
    Relevant: Yes

    Query: How does the 'sudden and accidental' coverage provision apply to property damage claims in commercial property insurance?
    Document: """This excerpt from a legal journal reviews several landmark cases that have interpreted the 'sudden' provision within commercial property insurance policies. It examines how courts have variously defined 'sudden' — with some interpreting it as referring strictly to the timing of the incident (unexpected) and others considering the onset and nature of the damage."""
    Relevant: Yes

    Query: How do insurance policies differentiate between coverage for natural disasters deemed 'acts of God' versus those considered preventable?
    Document: """This study offers a comprehensive review of how different insurance models globally have responded to natural disasters, including a discussion on public versus private insurance mechanisms, risk pooling, and government interventions. The document assesses the effectiveness of these models in disaster response and recovery, providing case studies from recent events."""
    Relevant: Yes

    Query: Under what circumstances can environmental damage claims be excluded from general liability insurance policies?
    Document: """This white paper explores the broader role of general liability insurance policies in encouraging businesses to adopt more sustainable practices. It discusses various strategies insurance companies are implementing to incentivize environmental responsibility among policyholders, such as offering premium discounts for eco-friendly business operations and integrating sustainability criteria into their risk assessment processes. """
    Relevant: No

    Query: {query}
    Document: """{document}"""
    Relevant:
    '''
    
    def document_relevance(query, document):
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model='gpt-3.5-turbo-16k',
            messages=[
                {
                    "role": "system",
                    "content": system_message,
                },
                {
                    "role": "user",
                    "content": input_prompt.format(query=query, document=document),
                },

            ],
            temperature=0,
            logprobs=True,
            logit_bias={ids[0][0]: 5, ids[1][0]: 5},
        )

        return (
            query,
            document,
            response.choices[0].logprobs.content[0].token,
            response.choices[0].logprobs.content[0].logprob
        )
    
    output_list = []
    for index, row in df.iterrows():
        document = row[text_col_name]
        try:
            output_list.append((index,) + document_relevance(query, document))
        except Exception as e:
            print(e)
        
    output_df = pd.DataFrame(
        output_list, columns=["index", "query", text_col_name, "prediction", "logprobs"]
    ).set_index('index')
    # Convert logprobs into probability
    output_df["probability"] = output_df["logprobs"].apply(exp)
    # Reorder based on likelihood of being Yes
    output_df["yes_probability"] = output_df.apply(
        lambda x: x["probability"] * -1 + 1
        if x["prediction"] == "No"
        else x["probability"],
        axis=1,
    )
    # Return reranked results
    reranked_df = df.join(output_df, rsuffix='_reranked')
    df_out = reranked_df.sort_values(by=["yes_probability"], ascending=False)
    return df_out
