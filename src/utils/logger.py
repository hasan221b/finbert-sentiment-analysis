import logging
import os
from src.utils.config import config

#Ensure logs directory exist

os.makedirs("logs", exist_ok=True)

def setup_logger (name, log_file, level = logging.INFO):
    #Function to set up logger with a specific log file and level

    formatter = logging.Formatter(config['logging']['format'])
    
    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger

api_logger = setup_logger('api_logger', config['logging']['api_log_file'],logging.INFO)
model_logger = setup_logger('model_log_file', config['logging']['model_log_file'],logging.INFO)
preprocessing_logger = setup_logger("preprocessing_logger", config["logging"]["preprocessing_log_file"], logging.INFO)
