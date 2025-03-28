import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.utils.logger import model_logger
from src.utils.config import config
import math

class FinBertModel:
    '''Initializing FinBERT model and main functions'''
    def __init__(self, model_path='ProsusAI/finbert'):
        #Load pre-trained FinBERT model and tokenizer
        try:
            model_logger.info('Initializing FinBERT model..')
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.labels = ['positive','negative','neutral']
            model_logger.info('Model initialization succeeded')
        except Exception as e:
            model_logger.error(f'Failed to load FinBERT model: {e}')
            raise RuntimeError('Model initialization failed')
    
    def predict_sentiment(self, text:str):
        '''Preform sentiment analysis on the input text
        
            Args: 
                text : User input text.

            Returns:
                sentiment : Sentiment class (Positive/Negative).
                confidence: Class accuracy confidence score.
                probabilities: Probabilities of each classes.
        '''
        try:
            model_logger.info(f'Predicting sentiment for the text: {text[:50]}...') #log the first 50 chars only
            inputs = self.tokenizer(
                text, return_tensors = 'pt',
                truncation = True,
                max_length = 512,
                padding = True
            )
            with torch.no_grad():
                outputs = self.model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=1).numpy()[0]
            predicted_label = self.labels[probabilities.argmax()]

            model_logger.info(f'Prediction: {predicted_label} (Confidence: {max(probabilities):.4f})')

            return {
                'sentiment' : predicted_label,
                'confidence' : float(probabilities.max()),
                'probabilities': {
                    label: float(prob) for label, prob in zip(self.labels, probabilities)
                },
            }
        except Exception as e:
            model_logger.error(f'Failed to make prediction: {e}')
            raise RuntimeError('Model prediction failed')          
    
    def compound_score(self, pos_prob:float, neg_prob:float, neu_prob: float):
        '''Calculating compound score for the text
        
            Args: 
                pos_prob: Positive class probability.
                neg_prob: Negative class probabilty.
                neu_prob: Neutral class probabilty.

            Returns:
                compound: Compound score for the text.   
        
        '''
        try:
            model_logger.info('Calculating compound score for the text...')
            numerator = pos_prob - neg_prob
            denominator = math.sqrt((pos_prob + neg_prob + neu_prob) ** 2 + 15)
            compound = numerator / denominator
            model_logger.info(f'Compound score: {compound}')
            return compound
        
        except Exception as e:
            model_logger.error(f'Failed to calculate compound: {e}')
            raise RuntimeError('Compound calculation failed')
    
    def ranking(self, text:str, pos_prob:float, neg_prob:float, neu_prob:float, comp: float):
        '''Calculating text ranking
        
            Args:   
                text: User input text.
                pos_prob: Positive class probability.
                neg_prob: Negative class probabilty.
                neu_prob: Neutral class probabilty.
                compound: Compound score for the text.
            Returns:
                news : User input text
                score : User input text rank score.
        '''
        try:
            model_logger.info('Calculating rank score for text...')
            rank_score = (pos_prob * 2) + (neg_prob * 1) - (neg_prob * 3) + (comp * 5)

            return {
                'news' : text,
                'score' : rank_score
            }
        except Exception as e:
            model_logger.error(f'Failed to calculate rank score: {e}')
            raise RuntimeError('Rank score calculation failed')