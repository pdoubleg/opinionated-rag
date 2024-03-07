from tqdm.asyncio import tqdm
from typing import Dict, List, Any, Literal, Optional
import instructor
import openai
from pydantic import BaseModel, Field
from tenacity import Retrying, AsyncRetrying, stop_after_attempt, wait_fixed


class CitationAnalysis(BaseModel):
    """Information about a legal citation"""

    citation: str = Field(
        ...,
        description="The Citation specified by the user.",
    )
    legal_question: str = Field(
        ...,
        description="A concise and well-structured legal question. For example: 'Plaintiff slipped and fell in the hotel lobby. Is the hotel liable?'",
    )
    rule: str = Field(
        ...,
        description="A concise legal ruling, decision, or authority. For example: 'If a hotel knows its floors are wet, it has a duty to take reasonable steps to avoid such injury.'",
    )
    application: str = Field(
        ...,
        description="Application, or potential application of the rule. For example: 'The hotel acted negligently'.",
    )
    citation_reference: str = Field(
        ...,
        description="A concise explanation of the way in which the citation was referenced or used in the context.",
    )
    

def analyze_citation(
    cited_opinion_bluebook_citation: str, excerpt: str
) -> CitationAnalysis:
    client = instructor.patch(openai.OpenAI())
    return client.chat.completions.create(
        model="gpt-4-turbo-preview",
        response_model=CitationAnalysis,
        max_retries=Retrying(
            stop=stop_after_attempt(5),
            wait=wait_fixed(1),
        ),
        messages=[
            {
                "role": "system",
                "content": "Your role is to extract information about a legal citation using the context, which is an excerpt from a subsequent legal proceeding that referenced the citation of interest.",
            },
            {
                "role": "user",
                "content": f"Your task focuses on citation: **{cited_opinion_bluebook_citation}**",
            },
            {   "role": "user", "content": f"Here is the context: {excerpt}"},
        ],
    )
    
    

async def analyze_citations(citation, context):
    client = instructor.patch(openai.AsyncOpenAI())
    return await client.chat.completions.create(
        model="gpt-4-turbo-preview",
        response_model=CitationAnalysis,
        max_retries=AsyncRetrying(
        stop=stop_after_attempt(3),
        wait=wait_fixed(1),
    ),
        messages=[
            {
                "role": "system",
                "content": "Your role is to extract information about a legal citation using the context, which is an excerpt from a subsequent legal proceeding that referenced the citation of interest.",
            },
            {"role": "user", "content": f"Your task focuses on citation: **{citation}**"},
            {"role": "user", "content": f"Here is the context: {context}"}
        ]
    )
    
    
async def process_citations_with_progress(decisions_context):
    results = []
    for forward_case in tqdm(decisions_context, total=len(decisions_context)):
        result = await analyze_citations(forward_case.context_citation, forward_case.context)
        results.append(result)
    return results

