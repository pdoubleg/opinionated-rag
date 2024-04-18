import uuid
from collections.abc import Callable
from concurrent.futures import as_completed
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from typing import Generic
from typing import TypeVar

from src.utils.logging import setup_colored_logging

logger = setup_colored_logging(__name__)

R = TypeVar("R")


def run_functions_tuples_in_parallel(
    functions_with_args: list[tuple[Callable, tuple]],
    allow_failures: bool = False,
    max_workers: int | None = None,
) -> list[Any]:
    """
    Executes multiple functions in parallel and returns a list of the results in the same order as the input.

    Args:
        functions_with_args: List of tuples, each containing a function callable and a tuple of its arguments.
        allow_failures: If True, continues execution even if a function raises an exception. The result for that
                        function will be None. If False (default), raises the exception and stops execution.
        max_workers: Maximum number of worker threads to use. If None (default), uses the number of input functions.

    Returns:
        list: A list of results in the same order as the input functions. If a function raised an exception and
              allow_failures is True, its result will be None.

    Raises:
        Exception: If allow_failures is False and any of the functions raised an exception.
    """
    workers = (
        min(max_workers, len(functions_with_args))
        if max_workers is not None
        else len(functions_with_args)
    )

    if workers <= 0:
        return []

    results = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_index = {
            executor.submit(func, *args): i
            for i, (func, args) in enumerate(functions_with_args)
        }

        for future in as_completed(future_to_index):
            index = future_to_index[future]
            try:
                results.append((index, future.result()))
            except Exception as e:
                logger.exception(f"Function at index {index} failed due to {e}")
                results.append((index, None))

                if not allow_failures:
                    raise

    results.sort(key=lambda x: x[0])
    return [result for index, result in results]


class FunctionCall(Generic[R]):
    """
    Container for run_functions_in_parallel, fetch the results from the output of
    run_functions_in_parallel via the FunctionCall.result_id.
    """

    def __init__(
        self, func: Callable[..., R], args: tuple = (), kwargs: dict | None = None
    ):
        self.func = func
        self.args = args
        self.kwargs = kwargs if kwargs is not None else {}
        self.result_id = str(uuid.uuid4())

    def execute(self) -> R:
        return self.func(*self.args, **self.kwargs)


def run_functions_in_parallel(
    function_calls: list[FunctionCall],
    allow_failures: bool = False,
) -> dict[str, Any]:
    """
    Executes a list of FunctionCalls in parallel and returns a dictionary of the results.

    Args:
        function_calls: List of FunctionCall objects to execute in parallel.
        allow_failures: If True, continues execution even if a function raises an exception. The result for that
                        function will be None. If False (default), raises the exception and stops execution.

    Returns:
        dict: A dictionary where the keys are the result_id of each FunctionCall and the values are the
              corresponding results. If a function raised an exception and allow_failures is True, its result
              will be None.

    Raises:
        Exception: If allow_failures is False and any of the functions raised an exception.
    """
    results = {}
    with ThreadPoolExecutor(max_workers=len(function_calls)) as executor:
        future_to_id = {
            executor.submit(func_call.execute): func_call.result_id
            for func_call in function_calls
        }

        for future in as_completed(future_to_id):
            result_id = future_to_id[future]
            try:
                results[result_id] = future.result()
            except Exception as e:
                logger.exception(f"Function with ID {result_id} failed due to {e}")
                results[result_id] = None

                if not allow_failures:
                    raise

    return results
