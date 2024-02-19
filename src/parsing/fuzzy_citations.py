import instructor

from typing import List
import logging
from openai import OpenAI
from pydantic import Field, BaseModel, FieldValidationInfo, model_validator

logger = logging.getLogger(__file__)

client = instructor.patch(OpenAI())


class Fact(BaseModel):
    statement: str = Field(
        ..., 
        description="A legal statement, argument or assertion."
    )
    substring_phrase: List[str] = Field(
        ...,
        description="String quote long enough to evaluate the truthfulness of the fact",
    )

    @model_validator(mode="after")
    def validate_sources(self, info: FieldValidationInfo) -> "Fact":
        """
        For each substring_phrase, find the span of the substring_phrase in the context.
        If the span is not found, remove the substring_phrase from the list.
        """
        if info.context is None:
            logger.info("No context found, skipping validation")
            return self

        # Get the context from the info
        text_chunks = info.context.get("text_chunk", None)

        # Get the spans of the substring_phrase in the context
        spans = list(self.get_spans(text_chunks))
        logger.info(
            f"Found {len(spans)} span(s) for from {len(self.substring_phrase)} citation(s)."
        )
        # Replace the substring_phrase with the actual substring
        self.substring_phrase = [text_chunks[span[0] : span[1]] for span in spans]
        return self

    def _get_span(self, quote, context, errs=10):
        import regex

        minor = quote
        major = context

        errs_ = 0
        s = regex.search(f"({minor}){{e<={errs_}}}", major)
        while s is None and errs_ <= errs:
            errs_ += 1
            s = regex.search(f"({minor}){{e<={errs_}}}", major)

        if s is not None:
            yield from s.spans()

    def get_spans(self, context):
        for quote in self.substring_phrase:
            yield from self._get_span(quote, context)


class QuestionAnswer(instructor.OpenAISchema):
    """
    Class representing a question and its answer as a list of facts each one should have a soruce.
    each sentence contains a body and a list of sources."""

    question: str = Field(..., description="Question that was asked")
    answer: List[Fact] = Field(
        ...,
        description="Body of the answer, each fact should be its seperate object with a body and a list of sources",
    )

    @model_validator(mode="after")
    def validate_sources(self) -> "QuestionAnswer":
        """
        Checks that each fact has some sources, and removes those that do not.
        """
        logger.info(f"Validating {len(self.answer)} facts")
        self.answer = [fact for fact in self.answer if len(fact.substring_phrase) > 0]
        logger.info(f"Found {len(self.answer)} facts with sources")
        return self


def ask_ai(question: str, context: str) -> QuestionAnswer:
    completion = client.chat.completions.create(
        model="gpt-4-turbo",
        temperature=0,
        functions=[QuestionAnswer.openai_schema],
        function_call={"name": QuestionAnswer.openai_schema["name"]},
        messages=[
            {
                "role": "system",
                "content": "You are a world class algorithm to answer questions with correct and exact citations. ",
            },
            {"role": "user", "content": "Answer question using the following context"},
            {"role": "user", "content": f"{context}"},
            {"role": "user", "content": f"Question: {question}"},
            {
                "role": "user",
                "content": "Tips: Make sure to cite your sources, and use the exact words from the context.",
            },
        ],
    )

    # Creating an Answer object from the completion response
    return QuestionAnswer.from_response(
        completion, validation_context={"text_chunk": context}
    )


