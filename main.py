from src.api.app import SentimentAnalysisAPI
import uvicorn

if __name__ == "__main__":
    uvicorn.run(SentimentAnalysisAPI().app, host="0.0.0.0", port=8000)
