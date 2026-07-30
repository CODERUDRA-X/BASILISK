"""
evaluate.py
-----------
Rolls a trained TransformerChaosPolicy forward through MacroChaosEnv to
sanity-check behavior on held-out (or freshly synthesized) trajectories.

This is the one place actual env.step() calls happen — used only for
evaluation/visualization, never for training (training is fully offline).
"""

from __future__ import annotations

import numpy as np
import torch

from .data_pipeline import MacroChaosDataPipeline, RawDataConfig
from .env import MacroChaosEnv, MarketSimConfig, OBS_DIM
from .policy import TransformerChaosPolicy


def evaluate_policy(policy: TransformerChaosPolicy, window_len: int = 16, device: str = "cpu"):
    # Load pipeline and environment
    pipeline = MacroChaosDataPipeline(RawDataConfig(source_type="csv"), seed=123)
    env = MacroChaosEnv(pipeline)

    obs, info = env.reset()
    history = [obs]  # rolling temporal buffer fed to the transformer
    total_reward = 0.0

    print("🚀 Starting Chaos Agent Evaluation...\n")

    done = False
    while not done:
        # Build the temporal window: pad with the earliest observation
        # until we have `window_len` steps of real history.
        window = history[-window_len:]
        if len(window) < window_len:
            window = [window[0]] * (window_len - len(window)) + window

        window_arr = np.array(window, dtype=np.float32)
        state_window = torch.from_numpy(window_arr).to(device).unsqueeze(0)
        
        with torch.no_grad():
            # Action is now a 4D vector [SPY, GLD, USO, VIXY]
            action = policy.act(state_window, deterministic=True).squeeze(0).cpu().numpy()

        # -- Single temporal step: t -> t+1 through the chaos-indicator series --
        obs, reward, terminated, truncated, step_info = env.step(action)
        history.append(obs)
        total_reward += reward
        done = terminated or truncated

    print(f"🏁 Episode finished.")
    print(f"📊 Total Steps: {env.t}")
    print(f"💰 Final Portfolio Value: {step_info['portfolio_value']:.4f}")
    print(f"📈 Total Reward: {total_reward:.4f}")
    
    if step_info.get("dd_breach"):
        print(f"💀 FATAL: Terminated early due to Drawdown Breach! (DD: {step_info['drawdown']:.2%})")
    else:
        print(f"🏆 SUCCESS: Agent survived the chaos without breaching risk limits!")

    return total_reward


if __name__ == "__main__":
    # obs is now [z_t, w_{t-1}, V_t, DD_t] = OBS_DIM-wide, and the action
    # space is an n_assets-wide weight vector -- both must match env.py.
    n_assets = MarketSimConfig().n_assets
    policy = TransformerChaosPolicy(window_len=16, state_dim=OBS_DIM, action_dim=n_assets)
    
    # Optional: Load trained weights if they exist
    # try:
    #     policy.load_state_dict(torch.load("macro_chaos_policy.pt"))
    #     print("✅ Loaded macro_chaos_policy.pt successfully.")
    # except FileNotFoundError:
    #     print("⚠️ No trained weights found. Running with random initialization.")
        
    evaluate_policy(policy)