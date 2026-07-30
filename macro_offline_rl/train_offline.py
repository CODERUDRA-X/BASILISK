"""
train_offline.py
-----------------
Offline RL training loop. Upgraded to handle the v1 Mathematical Contract
(OBS_DIM=30, action_dim=4). 
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from .data_pipeline import MacroChaosDataPipeline, OfflineTrajectoryDataset, RawDataConfig
from .env import OBS_DIM, MarketSimConfig
from .policy import TransformerChaosPolicy

class WindowedOfflineDataset(Dataset):
    """
    Mock Offline Dataset generator. In production, this will load real historical 
    (obs, action, reward) logs. Here, we mock the 30D observation and 4D actions.
    """
    def __init__(self, episodes: list[torch.Tensor], window_len: int = 16):
        self.window_len = window_len
        self.samples = []
        for ep in episodes:
            T = ep.shape[0]
            # episode tensor is 24D (z_t). We pad 6 dimensions to mock w_{t-1}, V_t, DD_t.
            mock_obs_ep = torch.cat([ep, torch.zeros(T, 6)], dim=-1)

            for t in range(window_len, T - 1):
                state_window = mock_obs_ep[t - window_len : t]
                next_state = mock_obs_ep[t]
                action = self._synthesize_action(state_window)
                reward = self._synthesize_reward(state_window, action)
                self.samples.append((state_window, action, reward, next_state))

    @staticmethod
    def _synthesize_action(state_window: torch.Tensor) -> torch.Tensor:
        # Generate 4-dimensional action instead of 1D
        return torch.tanh(torch.randn(4))

    @staticmethod
    def _synthesize_reward(state_window: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        chaos_proxy = state_window[-1, :24].norm() # Norm of z_t only
        reward = -action.mean() * chaos_proxy * 0.1
        return reward.unsqueeze(0)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def collate_fn(batch):
    states, actions, rewards, next_states = zip(*batch)
    return (
        torch.stack(states),
        torch.stack(actions),
        torch.stack(rewards).squeeze(1),
        torch.stack(next_states),
    )

def train_offline(
    n_episodes: int = 20,
    window_len: int = 16,
    batch_size: int = 32,
    epochs: int = 5,
    lr: float = 3e-4,
    cql_alpha: float = 1.0,
    device: str = "cpu",
) -> TransformerChaosPolicy:
    
    pipeline = MacroChaosDataPipeline(RawDataConfig(source_type="csv"), seed=0)
    raw_dataset = OfflineTrajectoryDataset(pipeline, n_episodes=n_episodes)
    episodes = raw_dataset.load()

    windowed = WindowedOfflineDataset(episodes, window_len=window_len)
    loader = DataLoader(windowed, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    n_assets = MarketSimConfig().n_assets
    policy = TransformerChaosPolicy(window_len=window_len, state_dim=OBS_DIM, action_dim=n_assets).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0

        for state_window, action, reward, next_state in loader:
            state_window = state_window.to(device)
            action = action.to(device)
            reward = reward.to(device)

            pred_mean, log_std, value = policy(state_window)

            bc_loss = F.mse_loss(pred_mean, action)
            value_loss = F.mse_loss(value.squeeze(-1), reward)

            # Simplified CQL penalty
            random_actions = torch.empty_like(pred_mean).uniform_(-1, 1)
            _, _, random_value = policy(state_window)
            conservative_penalty = (random_value.squeeze(-1) - value.squeeze(-1)).mean()

            loss = bc_loss + value_loss + cql_alpha * conservative_penalty

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        print(f"[epoch {epoch + 1}/{epochs}] avg_loss={avg_loss:.4f}")

    return policy

if __name__ == "__main__":
    trained_policy = train_offline()
    torch.save(trained_policy.state_dict(), "macro_chaos_policy.pt")
    print("Saved trained policy to macro_chaos_policy.pt")