import re
from typing import Union
import dspy
import pandas as pd


def normalize(
    label: str,
    do_lower: bool = True,
    strip_punct: bool = True,
    split_colon: bool = False,
) -> str:
    # Sometimes models wrongfully output a field-prefix, which we can remove.
    if split_colon:
        label = label.split(":")[1] if ":" in label else label

    # Remove leading and trailing newlines
    label = label.strip("\n")

    # Remove leading and trailing punctuation and newlines
    if strip_punct:
        label = re.sub(r"^[^\w\s]+|[^\w\s]+$", "", label, flags=re.UNICODE)

    # Remove leading and trailing newlines
    label = label.strip("\n")

    # NOTE: lowering the labels might hurt for case-sensitive ontologies.
    if do_lower:
        return label.strip().lower()
    else:
        return label.strip()


# given a comma-separated string of labels, parse into a list
def extract_labels_from_string(
    labels: str,
    do_lower: bool = True,
    strip_punct: bool = True,
    split_colon: bool = False,
) -> list[str]:
    return [
        normalize(r, do_lower=do_lower, strip_punct=strip_punct)
        for r in labels.split(",")
    ]


# given a list of comma-separated string of labels, parse into a list
def extract_labels_from_strings(
    labels: list[str],
    do_lower: bool = True,
    strip_punct: bool = True,
    split_colon: bool = False,
) -> list[str]:
    labels = [
        normalize(
            r, do_lower=do_lower, strip_punct=strip_punct, split_colon=split_colon
        )
        for r in labels
    ]
    labels = ", ".join(labels)
    return extract_labels_from_string(
        labels, do_lower=do_lower, strip_punct=strip_punct, split_colon=split_colon
    )
    
    
def get_dspy_examples(
    validation_df: Union[pd.DataFrame, None],
    test_df: pd.DataFrame,
    n_validation: int = None,
    n_test: int = None,
) -> tuple[list[dspy.Example], list[dspy.Example]]:
    valset, testset = [], []

    n_validation = float("+inf") if not n_validation else n_validation
    n_test = float("+inf") if not n_test else n_test

    if validation_df is not None:
        for _, example in validation_df.iterrows():
            if len(valset) >= n_validation:
                break
            valset.append(example.to_dict())
        valset = [dspy.Example(**x).with_inputs("text") for x in valset]

    for _, example in test_df.iterrows():
        if len(testset) >= n_test:
            break
        testset.append(example.to_dict())
    testset = [dspy.Example(**x).with_inputs("text") for x in testset]

    # print(len(valset), len(testset))
    return valset, testset