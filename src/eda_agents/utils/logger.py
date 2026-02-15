import logging
import os
import sys
from logging.handlers import RotatingFileHandler

def setup_logger(name: str = "eda_agents", level: int = logging.INFO, log_file: str = "logs/eda_agents.log") -> logging.Logger:
    """
    Sets up a logger with a standard configuration including both console and file output.
    
    Args:
        name (str): The name of the logger.
        level (int): The logging level.
        log_file (str): Path to the log file.
        
    Returns:
        logging.Logger: The configured logger.
    """
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        logger.setLevel(level)
        
        # Create logs directory if it doesn't exist
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        # Console Handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        
        # File Handler (Rotating)
        file_handler = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5)
        file_handler.setLevel(level)
        
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
        )
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
        
    return logger

# Primary logger instance
logger = setup_logger()
logger.info("Logging system initialized.")
