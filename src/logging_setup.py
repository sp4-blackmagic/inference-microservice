import os
import logging
from logging.handlers import RotatingFileHandler
from rich.logging import RichHandler
import rich.traceback

def setup_logging(module_name: str = "app") -> logging.Logger:
    """Configure application logging with both file and console output."""
    # Setup rich traceback handling
    rich.traceback.install()

    # Get environment setting for file logging (default to false)
    enable_file_logging = os.getenv("ENABLE_FILE_LOGGING", "false").lower() in ("true", "1", "yes")
    
    # Configure root logger with RichHandler for console
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)]
    )

    root_logger = logging.getLogger()
    
    # Only add file logging if enabled
    if enable_file_logging:
        # Create logs directory
        os.makedirs('logs', exist_ok=True)
        
        # Add file handler with detailed format
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
        )
        file_handler = RotatingFileHandler(
            f'logs/{module_name}.log',
            maxBytes=10485760,  # 10MB
            backupCount=5
        )
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)

    # Return a logger for the calling module
    return logging.getLogger(module_name)