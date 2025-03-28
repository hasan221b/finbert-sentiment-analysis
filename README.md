# End-to-End sentiment analysis MLOPs project 
## Overview
This project provides a sentiment analysis tool specifically designed for financial texts using the FinBERT model. It's built to process and analyze sentiment in financial news, reports, and other related content, helping users gauge market sentiment from textual data.

## Project Structure <br>
The repository is organized as follows:

= `finbert-sentiment-analysis/`
 - `config/` - Configuration files  
 - `data/` - Sample datasets  
 - `log/` - Application and model logs  
 - `models/` - Saved model and tokenizer  
 - `src/` - Main source code  
   - `api/` - API implementation  
   - `models/` - Model-related code  
   - `processing/` - Text preprocessing  
   - `schemas/` - Data models  
   - `utils/` - Utility functions  
 - `tests/` - Test cases  
 - `Dockerfile` - Docker configuration  
 - `requirements.txt` - Python dependencies  
 - `README.md` - Project documentation  

## Features
1. Pre-trained FinBERT model for financial sentiment analysis
2. Custom text preprocessing for financial texts
3. REST API endpoint for sentiment analysis
4. Comprehensive logging system
5. Docker support for easy deployment

## Getting Started
### Prerequisites<br>

Python 3.9 or higher

### Installation
1. Clone the repository:
```
git clone https://github.com/hasan221b/finbert-sentiment-analysis.git
cd finbert-sentiment-analysis
```

### Install dependencies:

```
pip install -r requirements.txt
```
### Running the Application

To start the API service locally:

```
uvicorn src.api.app:app --reload
```

### Usage
The API provides a single endpoint for sentiment analysis:

```
POST /analyze
Content-Type: application/json
```
```

{
    "text": "Your financial text here..."
}
```
Example response:

```
{
    "sentiment": "positive",
    "confidence": 0.92
}
```
### Testing
Run the test suite with:

```
python -m pytest tests/
```

### Configuration

Modify config/config.yml to adjust:

1. Model parameters

2. Logging settings

3. API configurations

### Contributing
Contributions are welcome. Please fork the repository and submit a pull request with your changes.
