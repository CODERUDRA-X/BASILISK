"""
live_ingestion.py
-----------------
Connects Project Basilisk to real-world Hugging Face data streams.
"""

import numpy as np
import pandas as pd
from datasets import load_dataset
from .data_pipeline import STATE_DIM

def fetch_live_chaos(hf_token: str = None) -> np.ndarray:
    """Pulls live geopolitical and sentiment data from Hugging Face."""
    
    # The actual Red Ticker in your terminal
    print("\033[91m▲ // DATA VECTOR: PARSING SOVAI/NEWS_SENTIMENT CHANNELS...\033[0m")
    
    try:
        # Connect to HF Hub and pull the latest dataset split
        dataset = load_dataset("sovai/news_sentiment", split="train", token=hf_token)
        df = dataset.to_pandas()
        
        # Extract numerical features for our 24D state space
        live_data = df.select_dtypes(include=[np.number]).values
        
        if live_data.shape[1] < STATE_DIM:
            # Pad missing dimensions with zeros to satisfy the math contract
            padding = np.zeros((live_data.shape[0], STATE_DIM - live_data.shape[1]))
            live_data = np.hstack([live_data, padding])
            
        print("\033[92m✔ LIVE DATA INGESTION COMPLETE.\033[0m")
        return live_data[-500:, :STATE_DIM]
        
    except Exception as e:
        # Fallback keeps Basilisk alive if the HF API is gated or down
        print(f"\033[93m⚠ HF API LOCKED/DOWN. INJECTING SYNTHETIC CHAOS...\033[0m")
        rng = np.random.default_rng(42)
        return rng.normal(0.0, 1.0, size=(500, STATE_DIM)).cumsum(axis=0)

if __name__ == "__main__":
    # Test the live connection locally
    test_tensor = fetch_live_chaos()
    print(f"Tensor Shape Locked: {test_tensor.shape}")