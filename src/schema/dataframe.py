from instructor import OpenAISchema
from pydantic import Field
from typing import List, Any
import openai


class RowData(OpenAISchema):
    row: List[Any] = Field(..., description="The values for each row")


class Dataframe(OpenAISchema):
    """
    Class representing a dataframe. This class is used to convert
    data into a frame that can be used by pandas.
    """

    data: List[RowData] = Field(
        ...,
        description="Correct rows of data aligned to column names, Nones are allowed",
    )
    columns: List[str] = Field(
        ...,
        description="Column names relevant from source data, should be in snake_case",
    )

    def to_pandas(self):
        import pandas as pd

        columns = self.columns
        data = [row.row for row in self.data]

        return pd.DataFrame(data=data, columns=columns)


def dataframe(data: str) -> Dataframe:
    completion = openai.ChatCompletion.create(
        model="gpt-4",
        temperature=0.1,
        functions=[Dataframe.openai_schema],
        function_call={"name": Dataframe.openai_schema["name"]},
        messages=[
            {
                "role": "system",
                "content": """Map the pyramids, their build date, builders into a dataframe and correctly define the correct columns and rows""",
            },
            {
                "role": "user",
                "content": f"{data}",
            },
        ],
        max_tokens=1000,
    )
    return Dataframe.from_response(completion)


if __name__ == "__main__":
    with open("pyramids.txt", "r", encoding="utf-8") as file:
        text = file.read()
    df = dataframe(
        text,
    )

    print(df.to_pandas())
    # save the dataframe to a csv
    df.to_pandas().to_csv("pyramids.csv", index=False)
