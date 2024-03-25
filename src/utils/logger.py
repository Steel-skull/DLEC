import logging

def setup_custom_logger(name):
    """
    Creates a custom logger with the specified name and configuration.
    
    Args:
        name (str): The name of the logger to create. Typically, this is the name of the module.

    Returns:
        logging.Logger: The configured logger.
    """
    
    formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(module)s - %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    
    return logger
