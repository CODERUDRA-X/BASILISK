"""
data_pipeline.py
-----------------
Multi-modal data pipeline for macro-economic "chaos indicator" state vectors.
Extracts Geo, Weather, and Port data and compresses them into a 24D latent state (z_t).
"""

from __future__ import annotations
import numpy as np
import torch
import pandas as pd
from dataclasses import dataclass
from typing import Optional

# Neural Latent Dimensions
GEO_LATENT_DIM = 8
WEATHER_LATENT_DIM = 8
PORT_LATENT_DIM = 8
STATE_DIM = GEO_LATENT_DIM + WEATHER_LATENT_DIM + PORT_LATENT_DIM  # 24

@dataclass
class RawDataConfig:
    source_type: str = "csv"
    path_or_uri: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    resample_freq: str = "1D"

class MacroChaosDataPipeline:
    def __init__(self, config: RawDataConfig, seed: Optional[int] = None):
        self.config = config
        self.rng = np.random.default_rng(seed)

    def _fetch_raw(self) -> np.ndarray:
        # Import the live ingestion module
        from .live_ingestion import fetch_live_chaos
        
        # Fetch the live stream (Pass your HF Token here if the dataset is private)
        raw = fetch_live_chaos(hf_token="YOUR_HUGGINGFACE_TOKEN_HERE")
        return raw
    
    def _engineer_features(self, raw: np.ndarray) -> np.ndarray:
        mean = raw.mean(axis=0, keepdims=True)
        std = raw.std(axis=0, keepdims=True) + 1e-8
        normalized = (raw - mean) / std
        return normalized.astype(np.float32)

    def build_episode_tensor(self) -> torch.Tensor:
        """
        Returns a (T, 24) float32 tensor representing z_t.
        The environment will append portfolio data to this later to make it 30D.
        """
        raw = self._fetch_raw()
        features = self._engineer_features(raw)
        return torch.from_numpy(features)

class OfflineTrajectoryDataset:
    def __init__(self, pipeline: MacroChaosDataPipeline, n_episodes: int = 20):
        self.pipeline = pipeline
        self.n_episodes = n_episodes
        self.episodes: list[torch.Tensor] = []

    def _load_logged_transitions(self):
        for _ in range(self.n_episodes):
            self.episodes.append(self.pipeline.build_episode_tensor())

    def load(self):
        self._load_logged_transitions()
        return self.episodes