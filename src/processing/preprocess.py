import re
from src.utils.logger import preprocessing_logger

def clean_text(text):

    '''Cleaning text: Remove URLs, special characters and convert to lowercase
    
        Args:
            text: User input text.
        Returns:
            text: cleaned text
    '''
    preprocessing_logger.info(f'Cleaning text: {text[:50]}...')

    #Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+','',text, flags = re.MULTILINE)
    #Remove special characters (keep revelant ones like $, %)
    text = re.sub(r'[^\w\s$%]','',text)
    #Convert to lowercase
    text = text.lower()
    
    preprocessing_logger.info(f'Cleaned text: {text[:50]}...')
    return text