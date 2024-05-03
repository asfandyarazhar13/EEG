import logging
import sys
from typing import Optional


class CustomFormatter(logging.Formatter):
    """
    CustomFormatter changes the color of log level names based on the severity of the log message.
    """
    # Color escape sequences
    COLOR_RED = "\033[91m"
    COLOR_GREEN = "\033[92m"
    COLOR_YELLOW = "\033[93m"
    COLOR_BLUE = "\033[94m"
    COLOR_RESET = "\033[0m"
    COLOR_GREY = "\033[90m"
    COLOR_DEFAULT = "\033[38;21m"
    COLOR_BOLD_RED = "\033[1;31m"
    COLOR_CYAN = "\033[36m"

    # Mapping log level to corresponding colors
    FORMATS = {
        logging.INFO: COLOR_GREEN,
        logging.DEBUG: COLOR_GREY,
        logging.WARNING: COLOR_YELLOW,
        logging.ERROR: COLOR_RED,
        logging.CRITICAL: COLOR_BOLD_RED
    }

    def formatTime(self, record, datefmt=None):
        """
        Format the creation time of the record to include color.
        
        Args:
            record: The log record.
            datefmt (Optional[str]): A string specifying the date format.
        
        Returns:
            str: A string representing the time with purple color.
        """
        asctime = super().formatTime(record, datefmt)
        return f"{self.COLOR_CYAN}{asctime}{self.COLOR_RESET}"

    def format(self, record):
        """
        Format the specified record as text.
        
        Args:
            record: The log record.
        
        Returns:
            str: Formatted log record with colored level name.
        """
        color = self.FORMATS.get(record.levelno, self.COLOR_DEFAULT)
        record.levelname = f"{color}{record.levelname}{self.COLOR_RESET}"
        return super().format(record)


def setup_logger(level: int = logging.INFO, filename: Optional[str] = None, mode: str = "w") -> None:
    """
    Set up the logger with both console and file handlers.

    Args:
        level (int): The threshold level for the logger.
        filename (Optional[str]): Filename to log to. If None, logging to file is skipped.
        mode (str): Mode to open the file. Default is 'w' for write, which will overwrite the file.
    """
    # Define formatter for the console that includes colors
    color_formatter = CustomFormatter(
        fmt="%(asctime)s [%(levelname)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Define a basic formatter for file output
    basic_formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Setup the console handler with the color formatter
    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setFormatter(color_formatter)

    # Initialize the logger with the console handler
    handlers = [console_handler]

    # If a filename is provided, setup the file handler
    if filename:
        file_handler = logging.FileHandler(filename, mode)
        file_handler.setFormatter(basic_formatter)
        handlers.append(file_handler)

    # Configure the root logger
    logging.basicConfig(level=level, handlers=handlers)
