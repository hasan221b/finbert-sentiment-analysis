from fastapi import FastAPI, HTTPException
from src.models.finbert_model import FinBertModel
from src.schemas.data_models import TextInput, PredictionOutput
from src.processing.preprocess import clean_text
from src.utils.logger import api_logger
from src.utils.config import config
from src.utils.file_handler import append_to_csv, read_csv_sorted

class SentimentAnalysisAPI:
    '''Chatbot application using FastAPI
    
    
    Attributes:
        app (FastAPI): The FastAPI application.
        Model(FinBERT): The NLP sentiment anslysis.
    '''
    def __init__(self):
        '''Initializing FastAPI app'''
        self.app = FastAPI()
        self._initialize_model()
        self._setup_routes()

    def _initialize_model(self):
        '''Initializing FinBERT model'''
        try:
            self.finbert_model = FinBertModel()
        except Exception as e:
            api_logger.error(f'Failed to initialize model: {str(e)}')
            raise RuntimeError('Model initialization failed')

    def _setup_routes(self):
        @self.app.get('/')
        async def read_root():
            api_logger.info('Root endpoint accessed.')
            return {'message': 'Sentiment analysis API is running'}

        @self.app.get('/health')
        async def health_check():
            api_logger.info('Health check endpoint accessed.')
            return {'status': 'healthy'}

        @self.app.post('/predict', response_model=PredictionOutput)
        async def sentiment_predict(input_data: TextInput):
            '''Sentiment analysis function
            
                Args:
                    TextInput.text : user input text.

                Returns:
                    sentiment : sentiment class (Positive/Negative/Neutral).
                    confidence: class accuracy confidence score.
                    probabilities: probabilities of each classes.
                    compound : compound score for the text.
                    rank_score: text rank score.
            '''
            try:
                api_logger.info(f'Received prediction request: {input_data.text[:50]}...')
                text = clean_text(input_data.text)
                result = self.finbert_model.predict_sentiment(text)
                compound = self.finbert_model.compound_score(result['probabilities']['positive'],
                                                             result['probabilities']['negative'],
                                                             result['probabilities']['neutral'])
                rank = self.finbert_model.ranking(text,
                                                 result['probabilities']['positive'],
                                                 result['probabilities']['negative'],
                                                 result['probabilities']['neutral'],
                                                 compound)
                api_logger.info('Calculations completed')
                api_logger.info('Saving Sentiment and score...')
                append_to_csv(rank)
                api_logger.info('Saving completed')

                if 'error' in result:
                    raise HTTPException(status_code=500, detail=result['error'])

                return {'sentiment': result['sentiment'],
                        'confidence': result['confidence'],
                        'probabilities': result['probabilities'],
                        'compound': compound,
                        'rank_score': rank['score']}

            except Exception as e:
                api_logger.error(f"Prediction failed: {e}")
                raise HTTPException(status_code=500, detail="Prediction failed")

        @self.app.get('/ranking')
        async def news_ranking():
            '''Getting the rank of every text we have on our df
            
                Args: 
                    None
                Return: 
                    sorted_df: A sorted df with text & ranks
            
            '''
            sorted_df = read_csv_sorted(ascending=False)
            return sorted_df

app = SentimentAnalysisAPI().app
