import logging
import sys

def setup_logger(name: str = "eda_agents", level: int = logging.INFO) -> logging.Logger:
    """
    Sets up a logger with a standard configuration.
    
    Args:
        name (str): The name of the logger.
        level (int): The logging level.
        
    Returns:
        logging.Logger: The configured logger.
    """
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        logger.setLevel(level)
        
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        
        logger.addHandler(handler)
        
    return logger

logger = setup_logger()
