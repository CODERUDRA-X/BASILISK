<div align="center">
<img src="https://github.com/CODERUDRA-X/BASILISK/blob/main/logo.png?raw=true" alt="BASILISK Logo" width="200"/>
  
# 🐍 PROJECT B.A.S.I.L.I.S.K.
**Binary Algorithmic System for Intelligent Liquidity & Invisible Shock Kinetics**

*An Elite Autonomous Chaos Trading Terminal by CODERUDRA-X*

> *"The Basilisk does not passively survive the crash; it freezes the chaos and profits from it."*

</div>

---

## 👁️ The Philosophy: The Apex Predictor
In ancient lore, the Basilisk is the King of Serpents. Its gaze paralyzes victims, turning them to stone. In modern tech-philosophy, Roko’s Basilisk is the ultimate, inevitable Super-Intelligent AI.

We mapped this directly into our **Transformer-Based Policy Network**. The foundational mechanism of a Transformer is "Self-Attention." Our AI does not look at the market linearly. Its *Attention Matrix* casts a calculated gaze over temporal windows of chaotic alternative data (geopolitics, supply chains, weather anomalies). Where the Basilisk’s gaze falls, the chaos freezes into structured, risk-adjusted profit.

---

## ⚙️ Mathematical Engine & Risk Contract
This is an Offline Reinforcement Learning (CQL-style) system modeled as a Partially Observable Markov Decision Process (POMDP). 

### 1. The Observation Space ($obs_t \in \mathbb{R}^{30}$)
The system ingests a high-dimensional state vector mapping the planet's chaos alongside the portfolio's real-time health:
$$obs_t = [z_t; w_{t-1}; V_t; DD_t]$$

*   **$z_t \in \mathbb{R}^{24}$**: Latent Macro-Chaos variables (Geo, Weather, Port congestion).
*   **$w_{t-1} \in \mathbb{R}^4$**: Previous portfolio allocations (SPY, GLD, USO, VIXY).
*   **$V_t$**: Current Portfolio Value.
*   **$DD_t$**: Rolling Drawdown.

### 2. The Action Space ($w_t \in \mathbb{R}^4$)
The agent outputs a continuous 4D vector, passed through a strict $L_1$ projection to enforce maximum gross leverage ($L_{max}=1$):
$$\tilde{w}_t = \text{clip}(a_t, -1, 1)$$
$$w_t = \frac{\tilde{w}_t}{\max(1.0, \frac{||\tilde{w}_t||_1}{L_{max}})}$$

### 3. The Objective Function & Penalties
The environment enforces a rigorous mathematical reward $r_t$ penalizing CVaR (Conditional Value at Risk), Transaction Costs ($TC_t$), and extreme multi-dimensional shocks ($S_t$):
$$S_t = \sum_k \text{softplus}(b_k \cdot z_t^k + c_k) \cdot |e_{t,k}|^{1.5}$$

**The Risk Governor:** Absolute termination triggers at exactly **10% Drawdown** ($DD_t \ge 0.10$). The agent is mathematically prohibited from blowing up the account.

---

## 📂 Architecture Breakdown
The system is divided into a high-integrity Python Deep-State Backend and a React Dashboard.

### `[CORE BACKEND]`
*   `env.py` — The strict Gymnasium-compatible mathematical contract and risk enforcer.
*   `policy.py` — The Multi-Asset Temporal Transformer Brain.
*   `train_offline.py` — The Conservative Q-Learning (CQL) offline trainer.
*   `data_pipeline.py` & `live_ingestion.py` — Ingests live sentiment and global anomaly data directly from Hugging Face streams.
*   `server.py` — The FastAPI bridge converting the Python engine into a web-accessible API.

### `[TERMINAL FRONTEND]`
*   `chaos-dashboard/` — A sleek, dark-mode React + Tailwind CSS v3 institutional terminal featuring real-time Recharts visualization of the agent's survivability.

---

## 🚀 Deployment & Operations Manual

### Phase 1: Initialize The Deep-State Engine
Clone the repository and spin up the Python backend environment.
```bash
# Navigate to the root directory
cd Chaos_Project/macro_offline_rl

# Install critical ML and Server dependencies
pip install torch numpy pandas gymnasium datasets fastapi uvicorn

# Train the Basilisk Agent on Offline Data (Wait for weights to save)
python -m macro_offline_rl.train_offline

# Initiate the live Data Ingestion & FastAPI Server
python server.py
