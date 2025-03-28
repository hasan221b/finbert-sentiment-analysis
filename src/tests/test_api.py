import pytest
from fastapi.testclient import TestClient
from src.api.app import SentimentAnalysisAPI

@pytest.fixture
def client():
    api = SentimentAnalysisAPI()
    return TestClient(api.app)

def test_health_check(client):
    response = client.get('/health')
    assert response.status_code == 200
    assert response.json() == {'status': 'healthy'}

def test_root(client):
    response = client.get('/')
    assert response.status_code == 200
    assert response.json() == {'message': 'Sentiment analysis API is running'}

def test_sentiment_prediction(client):
    sample_text = {'text': 'This is a fantastic product!'}
    response = client.post('/predict', json=sample_text)
    assert response.status_code == 200
    result = response.json()
    
    assert 'sentiment' in result, 'Sentiment key should be in the response'
    assert 'confidence' in result, 'Confidence score should be included'
    assert 'probabilities' in result, 'Probabilities should be included'

    assert isinstance(result['confidence'], float), 'Confidence should be a float'
    assert isinstance(result['probabilities'], dict), 'Probabilities should be a dictionary'

    total_prob = sum(result['probabilities'].values())
    assert abs(total_prob - 1.0) < 1e-3, f'Probabilities sum should be close to 1, got {total_prob}'

def test_invalid_input(client):
    response = client.post('/predict', json={})
    assert response.status_code == 422, 'Should return 422 for missing text field'