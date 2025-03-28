import pytest
from src.models.finbert_model import FinBertModel
from src.processing.preprocess import clean_text

@pytest.fixture
def model():
    return FinBertModel()

def test_model_initialization(model):
    assert model is not None, 'Model is not initialized. It should not be None'

def test_sentiment_prediction(model):
    sample_text = 'I love this product! Its amazing'
    sample_text = clean_text(sample_text)
    result = model.predict_sentiment(sample_text)

    compound = model.compound_score(result['probabilities']['positive'],
                                     result['probabilities']['negative'],
                                     result['probabilities']['neutral'])
    rank = model.ranking(sample_text,
                                result['probabilities']['positive'],
                                result['probabilities']['negative'],
                                result['probabilities']['neutral'],
                                compound)
    
    assert isinstance(result, dict), 'Result should be dictionary'
    assert 'sentiment' in result, 'Sentiment key should be in the result'
    assert result["sentiment"] in ["positive", "negative", "neutral"], "Unexpected sentiment label"
    total_prob = sum(result['probabilities'].values())
    assert abs(total_prob - 1.0) < 1e-3, f'Probabilities sum should be close to 1, got {total_prob}'