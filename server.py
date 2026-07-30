"""
server.py
---------
FastAPI server to connect our Chaos Agent Backend with the React Frontend UI.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import torch
import uvicorn

from macro_offline_rl.data_pipeline import MacroChaosDataPipeline, RawDataConfig
from macro_offline_rl.env import MacroChaosEnv, MarketSimConfig, OBS_DIM
from macro_offline_rl.policy import TransformerChaosPolicy

app = FastAPI(title="Chaos Agent API")

# Allow Frontend (React) to talk to this Backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the Agent and Environment globally
n_assets = MarketSimConfig().n_assets
policy = TransformerChaosPolicy(window_len=16, state_dim=OBS_DIM, action_dim=n_assets)
try:
    policy.load_state_dict(torch.load("macro_chaos_policy.pt", weights_only=True))
    print("✅ Loaded trained weights!")
except:
    print("⚠️ Using random weights.")

pipeline = MacroChaosDataPipeline(RawDataConfig(source_type="csv"), seed=123)
env = MacroChaosEnv(pipeline)

@app.get("/")
def read_root():
    return {"status": "Chaos Agent Backend is Live 🚀"}

@app.get("/run_simulation")
def run_simulation():
    """Runs one full episode and sends the data to the frontend for charting."""
    obs, _ = env.reset()
    history = [obs]
    
    chart_data = []
    done = False
    
    while not done:
        window = history[-16:]
        if len(window) < 16:
            window = [window[0]] * (16 - len(window)) + window
            
        state_window = torch.from_numpy(np.array(window, dtype=np.float32)).unsqueeze(0)
        
        with torch.no_grad():
            action = policy.act(state_window, deterministic=True).squeeze(0).numpy()
            
        obs, reward, terminated, truncated, info = env.step(action)
        history.append(obs)
        
        # Collect data for the React Chart
        chart_data.append({
            "step": info["timestep"],
            "portfolio_value": info["portfolio_value"],
            "drawdown": info["drawdown"],
            "chaos_signal": info["chaos_signal"]
        })
        
        done = terminated or truncated

    return {
        "status": "terminated_due_to_risk" if terminated else "completed",
        "total_steps": env.t,
        "final_value": info["portfolio_value"],
        "chart_data": chart_data
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)