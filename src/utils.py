"""
Utility Functions Module
Provides retry logic, error handling decorators, and common utilities.
"""

import time
import functools
import logging
from typing import Callable, TypeVar, Any, Optional
from pathlib import Path

# Set up logger
logger = logging.getLogger(__name__)

T = TypeVar('T')


def retry(
    max_retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,),
    logger_instance: Optional[logging.Logger] = None
) -> Callable:
    """
    Retry decorator with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
        backoff: Multiplier for delay after each retry
        exceptions: Tuple of exceptions to catch and retry on
        logger_instance: Optional logger instance for logging retries
    
    Returns:
        Decorated function with retry logic
    """
    log = logger_instance or logger
    
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            current_delay = delay
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        log.warning(
                            f"Attempt {attempt + 1}/{max_retries + 1} failed for {func.__name__}: {str(e)}. "
                            f"Retrying in {current_delay:.2f} seconds..."
                        )
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        log.error(
                            f"All {max_retries + 1} attempts failed for {func.__name__}: {str(e)}"
                        )
                        raise
            
            # This should never be reached, but type checker needs it
            if last_exception:
                raise last_exception
            
        return wrapper
    return decorator


def safe_file_operation(operation: str = "read"):
    """
    Decorator for safe file I/O operations with retry logic.
    
    Args:
        operation: Type of operation ("read", "write", "delete")
    
    Returns:
        Decorated function with file I/O error handling
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        @retry(max_retries=3, delay=0.5, exceptions=(IOError, OSError, PermissionError))
        def wrapper(*args, **kwargs) -> T:
            try:
                # Ensure directory exists for write operations
                if operation == "write":
                    file_path = kwargs.get('file_path') or kwargs.get('output_path') or args[0] if args else None
                    if file_path:
                        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
                        logger.debug(f"Ensured directory exists for {file_path}")
                
                return func(*args, **kwargs)
            except FileNotFoundError as e:
                logger.error(f"File not found during {operation}: {e}")
                raise
            except PermissionError as e:
                logger.error(f"Permission denied during {operation}: {e}")
                raise
            except IOError as e:
                logger.error(f"I/O error during {operation}: {e}")
                raise
            except Exception as e:
                logger.error(f"Unexpected error during {operation}: {e}")
                raise
        
        return wrapper
    return decorator


def log_execution_time(func: Callable[..., T]) -> Callable[..., T]:
    """
    Decorator to log function execution time.
    
    Returns:
        Decorated function with execution time logging
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> T:
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.debug(f"{func.__name__} executed in {execution_time:.2f} seconds")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"{func.__name__} failed after {execution_time:.2f} seconds: {e}")
            raise
    
    return wrapper


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    log_format: Optional[str] = None
) -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path for logging
        log_format: Optional custom log format
    
    Returns:
        Configured logger instance
    """
    if log_format is None:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Create formatter
    formatter = logging.Formatter(log_format)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    
    # Remove existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        try:
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.DEBUG)  # More verbose in file
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
            logger.info(f"Logging to file: {log_file}")
        except Exception as e:
            logger.warning(f"Could not set up file logging: {e}")
    
    return root_logger

