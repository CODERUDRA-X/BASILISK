"""
server.py
---------
FastAPI WebSocket server to connect our Chaos Agent Backend with the React Frontend UI.
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import torch
import uvicorn
import asyncio

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

# Old Health Check Route (So you know it's running)
@app.get("/")
def read_root():
    return {"status": "Chaos Agent WebSocket Backend is Live 🚀"}

# New LIVE WebSocket Route for Cinematic UI
@app.websocket("/ws/simulate")
async def websocket_simulation(websocket: WebSocket):
    await websocket.accept()
    obs, _ = env.reset()
    history = [obs]
    done = False
    
    try:
        while not done:
            window = history[-16:]
            if len(window) < 16:
                window = [window[0]] * (16 - len(window)) + window
                
            state_window = torch.from_numpy(np.array(window, dtype=np.float32)).unsqueeze(0)
            
            # Real Anomaly Detection using data variance
            # A variance > 1.15 in normalized data is a genuine 1.5-sigma market shock
            data_variance = torch.var(state_window).item()
            is_shock = data_variance > 1.15
            
            if is_shock:
                await websocket.send_json({"type": "SHOCK", "variance": round(data_variance, 2)})
                # System freezes here until Human Sovereign (YOU) responds
                auth_msg = await websocket.receive_text() 
                
            with torch.no_grad():
                action = policy.act(state_window, deterministic=True).squeeze(0).numpy()
                
            obs, reward, terminated, truncated, info = env.step(action)
            history.append(obs)
            done = terminated or truncated
            
            # Send step-by-step data to React
            await websocket.send_json({
                "type": "STEP",
                "step": info["timestep"],
                "portfolio_value": info["portfolio_value"],
                "status": "TERMINATED" if terminated else "ACTIVE"
            })
            
            # Artificial delay so the chart draws smoothly like a real trading terminal
            await asyncio.sleep(0.1)
            
    except WebSocketDisconnect:
        print("⚠️ Sovereign Disconnected. Socket Closed.")

# The Uvicorn runner to keep the server alive!
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)