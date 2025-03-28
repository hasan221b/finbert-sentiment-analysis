import pandas as pd
from src.utils.logger import preprocessing_logger
from typing import Dict, List, Optional
import os

CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "news.csv")

def append_to_csv(data: Dict[str, float]) -> None:
    try:
        df = pd.DataFrame([data])
        df.to_csv(CSV_PATH, mode = 'a', index=False, header=False)
        
        preprocessing_logger.info(f'Appended data to CSV: {data}')

    except Exception as e:
        preprocessing_logger.error(f'Failed to append data to CSV: {str(e)}')
        raise RuntimeError('CSV append operation failed')

def read_csv_sorted(ascending: bool = False) -> List[Dict[str, float]]:
    try:
        df = pd.read_csv(CSV_PATH)
        sorted_df = df.sort_values(by='score', ascending=ascending)
        preprocessing_logger.info(f'Read and sorted CSV data. Total rows: {len(sorted_df)}')
        return sorted_df.to_dict("records")
    except Exception as e:
        preprocessing_logger.error(f'Failed to read or sort CSV: {str(e)}')
        raise RuntimeError('CSV read operation failed')

