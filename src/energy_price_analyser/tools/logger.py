"""
Logger utility for Jarvis.

This module has advanced logging functionalities, including automatic config,
personalized format and log levels management

Main Features:
- Automatic logger configuration
- Personalized format with timestamp and colored levels
- Automatic rotation of Log files
- Support for logging on file and console
- Configurable log levels management

Use examples:
    >>> from src.tools.logger import setup_logger, get_logger
    >>> setup_logger('jarvis', log_level='INFO')
    >>> logger = get_logger('jarvis')
    >>> logger.info('Application started successfully')
    >>> logger.error('Error during data processing')

Author: Lorenzo
Date: 2025
Version: 0.1.0
"""

import logging
import logging.handlers
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional
from multiprocessing import Queue
from logging.handlers import QueueHandler, QueueListener


class ColoredFormatter(logging.Formatter):
    """
    Personalized formatter to add colors at log messages.

    This class extend logging.Formatter to add colors ANSI at different
    log levels, improving message readability on console

    Attributes:
        COLORS (dict): mapping of different log levels with ANSI colors codes
    ------
    Note:
    """

    # ANSI color codes for different log levels
    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
        "RESET": "\033[0m",  # Reset color
    }

    def format(self, record: logging.LogRecord) -> str:
        """
        Format the log record adding colors.

        Args:
            record (logging.LogRecord): The log record to format

        Returns:
            str: The formatted message with colors
        ------
        Note:
        """
        # Add color to log level
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = (
                f"{self.COLORS[levelname]}{levelname}{self.COLORS['RESET']}"
            )

        return super().format(record)


# Global queue & listener (gestiti solo nel processo principale)
_log_queue = None
_listener = None


def setup_logger(
    name: str = "jarvis",
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
    console_output: bool = True,
    file_output: bool = True,
    use_queue: bool = False,
    queue: Optional[Queue] = None,
) -> logging.Logger:
    """
    Configure and return a personalized logger for Jarvis project.

    This function configures a logger with personalized formatting,
    automatic file rotation support and output on console and file.

    Args:
        name (str): Logger name. Default: 'jarvis'
        log_level (str): Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL').
                        Default: 'INFO'
        log_file (Optional[str]): Log file path. If None, uses 'logs/jarvis.log'.
                                 Default: None
        max_bytes (int): Maximum log file size before rotation.
                        Default: 10MB
        backup_count (int): Number of backup files to keep. Default: 5
        console_output (bool): If True, enables console output. Default: True
        file_output (bool): If True, enables file output. Default: True
        use_queue (bool): If True, use queue for logging. Default: False
        queue (Optional[Queue]): Queue for logging. Default: None

    Returns:
        logging.Logger: The configured logger

    Raises:
        ValueError: If log level is not valid
        OSError: If log directory cannot be created
    ------
    Note:
        Example:
            >>> logger = setup_logger('jarvis', 'DEBUG', 'logs/app.log')
            >>> logger.info('Logger configured successfully')
    """

    global _log_queue, _listener

    valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    if log_level.upper() not in valid_levels:
        raise ValueError(f"Invalid log level: {log_level}. Valid: {valid_levels}")

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))

    if logger.handlers:
        return logger

    if use_queue and queue is not None:
        handler = QueueHandler(queue)
        logger.addHandler(handler)
        return logger

    # Formatter
    formatter = ColoredFormatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    handlers = []

    if console_output:
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)
        handlers.append(ch)

    if file_output:
        if log_file is None:
            log_file = Path("logs") / f"{name}.log"
        else:
            log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        fh = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
        )
        fh.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        handlers.append(fh)

    if use_queue:
        _log_queue = queue or Queue()
        _listener = QueueListener(_log_queue, *handlers)
        _listener.start()

    for h in handlers:
        logger.addHandler(h)

    return logger


def get_logger(name: str = "jarvis") -> logging.Logger:
    """
    Return an existing logger or create a new one if it doesn't exist.

    Args:
        name (str): Logger name. Default: 'jarvis'

    Returns:
        logging.Logger: The requested logger
    ------
    Note:
        Example:
            >>> logger = get_logger('jarvis')
            >>> logger.info('Using existing logger')
    """
    return logging.getLogger(name)


def log_function_call(func):
    """
    Decorator to automatically log function calls.

    This decorator automatically records the start and end of each
    call to the decorated function, along with parameters and result.

    Args:
        func: The function to decorate

    Returns:
        function: The decorated function
    ------
    Note:
        Example:
            >>> @log_function_call
            >>> def process_data(data):
            >>>     return data * 2
            >>>
            >>> result = process_data([1, 2, 3])
            # Log: INFO - Call to process_data with args: ([1, 2, 3],), kwargs: {}
            # Log: INFO - process_data completed in X.XXX seconds
    """

    def wrapper(*args, **kwargs):
        logger = get_logger("jarvis.function")

        # Log function call start
        logger.debug(f"Call to {func.__name__} with args: {args}, kwargs: {kwargs}")

        start_time = datetime.now()
        try:
            result = func(*args, **kwargs)
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.debug(f"{func.__name__} completed in {execution_time:.3f} seconds")
            return result
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.error(
                f"{func.__name__} failed after {execution_time:.3f} seconds: {str(e)}"
            )
            raise

    return wrapper


def log_performance(operation_name: str):
    """
    Decorator to measure and log operation performance.

    Args:
        operation_name (str): Name of the operation to log

    Returns:
        function: Decorator function
    ------
    Note:
        Example:
            >>> @log_performance("data_processing")
            >>> def process_data():
            >>>     # processing code
            >>>     pass
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            logger = get_logger("jarvis")
            start_time = datetime.now()

            try:
                result = func(*args, **kwargs)
                execution_time = (datetime.now() - start_time).total_seconds()
                logger.info(
                    f"Performance {operation_name}: {execution_time:.3f} seconds"
                )
                return result
            except Exception as e:
                execution_time = (datetime.now() - start_time).total_seconds()
                logger.error(
                    f"Performance {operation_name} failed after {execution_time:.3f} seconds: {str(e)}"
                )
                raise

        return wrapper

    return decorator


def old_setup_logger(
    name: str = "jarvis",
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
    console_output: bool = True,
    file_output: bool = True,
) -> logging.Logger:
    #!DEPRECATED

    # Log level validation
    valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    if log_level.upper() not in valid_levels:
        raise ValueError(
            f"Invalid log level: {log_level}. " f"Valid levels: {valid_levels}"
        )

    # Log file path configuration
    if log_file is None:
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        log_file = log_dir / f"{name}.log"
    else:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

    # Logger creation
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))

    # Avoid handler duplication
    if logger.handlers:
        return logger

    # Message format
    formatter = ColoredFormatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler (only if requested)
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level.upper()))
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File handler (only if requested)
    if file_output:
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
        )
        file_handler.setLevel(getattr(logging, log_level.upper()))

        # File formatter (without colors)
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger


# Default logger configuration
if __name__ == "__main__":
    # Logger usage example
    logger = setup_logger("jarvis", "DEBUG")
    logger.info("Logger configured successfully")
    logger.debug("Debug message")
    logger.warning("Warning message")
    logger.error("Error message")
