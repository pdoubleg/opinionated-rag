import logging
import os.path
from typing import no_type_check
import logging
from functools import lru_cache, partial
from typing import Optional
from rich.logging import RichHandler  # type: ignore
from rich.markup import escape  # type: ignore
import colorlog
from rich.console import Console



@lru_cache()
def get_logger(
    name: Optional[str] = None,
) -> logging.Logger:
    """
    Retrieves a logger with the given name, or the root logger if no name is given.

    Args:
        name: The name of the logger to retrieve.

    Returns:
        The logger with the given name, or the root logger if no name is given.

    Example:
        Basic Usage of `get_logger`
        ```python
        from logging import get_logger

        logger = get_logger("logging.test")
        logger.info("This is a test") # Output: logging.test: This is a test

        debug_logger = get_logger("logging.debug")
        debug_logger.debug_kv("TITLE", "log message", "green")
        ```
    """
    parent_logger = logging.getLogger("logging")

    if name:
        # Append the name if given
        if not name.startswith(parent_logger.name + "."):
            logger = parent_logger.getChild(name)
        else:
            logger = logging.getLogger(name)
    else:
        logger = parent_logger

    add_logging_methods(logger)
    return logger


def setup_logging(
    level: Optional[str] = None,
) -> None:
    logger = get_logger()

    if level is not None:
        logger.setLevel(level)
    else:
        logger.setLevel(logging.INFO)

    logger.handlers.clear()

    handler = RichHandler(rich_tracebacks=True, markup=False)
    formatter = logging.Formatter("%(name)s: %(message)s")
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    logger.propagate = False
    return logger


def add_logging_methods(logger: logging.Logger) -> None:
    def log_style(level: int, message: str, style: Optional[str] = None):
        if not style:
            style = "default on default"
        message = f"[{style}]{escape(str(message))}[/]"
        logger.log(level, message, extra={"markup": True})

    def log_kv(
        level: int,
        key: str,
        value: str,
        key_style: str = "default on default",
        value_style: str = "default on default",
        delimiter: str = ": ",
    ):
        logger.log(
            level,
            f"[{key_style}]{escape(str(key))}{delimiter}[/][{value_style}]{escape(str(value))}[/]",
            extra={"markup": True},
        )

    setattr(logger, "debug_style", partial(log_style, logging.DEBUG))
    setattr(logger, "info_style", partial(log_style, logging.INFO))
    setattr(logger, "warning_style", partial(log_style, logging.WARNING))
    setattr(logger, "error_style", partial(log_style, logging.ERROR))
    setattr(logger, "critical_style", partial(log_style, logging.CRITICAL))

    setattr(logger, "debug_kv", partial(log_kv, logging.DEBUG))
    setattr(logger, "info_kv", partial(log_kv, logging.INFO))
    setattr(logger, "warning_kv", partial(log_kv, logging.WARNING))
    setattr(logger, "error_kv", partial(log_kv, logging.ERROR))
    setattr(logger, "critical_kv", partial(log_kv, logging.CRITICAL))


# Define a function to set up the colored logger
def setup_colored_logging(name) -> logging.Logger:
    logger = setup_logger(name)
    # Define the log format with color codes
    log_format = "%(log_color)s%(asctime)s - %(levelname)s - %(message)s%(reset)s"
    # Create a color formatter
    color_formatter = colorlog.ColoredFormatter(
        log_format,
        datefmt="%Y-%m-%d %H:%M:%S",
        reset=True,
        log_colors={
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "red,bg_white",
        },
    )
    # Configure the root logger to use the color formatter
    handler = logging.StreamHandler()
    handler.setFormatter(color_formatter)
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.propagate = False
    return logger

def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Set up a logger of module `name` at a desired level.
    Args:
        name: module name
        level: desired logging level
    Returns:
        logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.hasHandlers():
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def setup_console_logger(name: str) -> logging.Logger:
    logger = setup_logger(name)
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def setup_file_logger(
    name: str,
    filename: str,
    append: bool = False,
    log_format: bool = False,
    propagate: bool = False,
) -> logging.Logger:
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    file_mode = "a" if append else "w"
    logger = setup_logger(name)
    handler = logging.FileHandler(filename, mode=file_mode)
    handler.setLevel(logging.INFO)
    if log_format:
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
    else:
        formatter = logging.Formatter("%(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = propagate
    return logger


def setup_loggers_for_package(package_name: str, level: int) -> None:
    """
    Set up loggers for all modules in a package.
    This ensures that log-levels of modules outside the package are not affected.
    Args:
        package_name: main package name
        level: desired logging level
    Returns:
    """
    import importlib
    import pkgutil

    package = importlib.import_module(package_name)
    for _, module_name, _ in pkgutil.walk_packages(
        package.__path__, package.__name__ + "."
    ):
        module = importlib.import_module(module_name)
        setup_logger(module.__name__, level)


class RichFileLogger:
    def __init__(self, log_file: str, append: bool = False, color: bool = True):
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        self.log_file = log_file
        if not append:
            if os.path.exists(self.log_file):
                os.remove(self.log_file)
        self.file = None
        self.console = None
        self.append = append
        self.color = color

    @no_type_check
    def log(self, message: str) -> None:
        with open(self.log_file, "a") as f:
            if self.color:
                console = Console(file=f, force_terminal=True, width=200)
                console.print(message)
            else:
                print(message, file=f)
