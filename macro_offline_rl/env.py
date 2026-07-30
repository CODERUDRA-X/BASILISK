"""
env.py: ChaosTradingEnv v1 -- risk-adjusted RL environment.
Gymnasium + numpy only. Implements the strict Mathematical Contract.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .data_pipeline import (
    STATE_DIM,
    GEO_LATENT_DIM,
    WEATHER_LATENT_DIM,
    PORT_LATENT_DIM,
    MacroChaosDataPipeline,
)


def _softplus(x: float) -> float:
    """Numerically stable softplus: log(1 + e^x)."""
    return max(x, 0.0) + math.log1p(math.exp(-abs(x)))


@dataclass
class RewardConfig:
    """Lambda weights and thresholds for r_t."""
    lambda_var: float = 0.5
    lambda_cvar: float = 0.3
    lambda_soft: float = 1.0
    lambda_breach: float = 5.0
    lambda_sh: float = 1.0
    lambda_tc: float = 1.0
    alpha: float = 0.05              # CVaR tail probability
    dd_limit: float = 0.10          # hard DD limit; also the termination trigger
    cvar_window: int = 60           # rolling loss-history window
    chaos_scale_norm: float = 5.0   # normalizes ||z_t|| before tanh
    kappa_tc: float = 0.0010
    kappa_impact: float = 0.0020


@dataclass
class MarketSimConfig:
    """Placeholder R_{t+1} generator: R ~ N(mu, Sigma_t)."""
    n_assets: int = 4
    mu_baseline: float = 0.0003
    base_vol: float = 0.01
    base_corr: float = 0.2
    chaos_vol_scale: float = 0.5


@dataclass
class ShockConfig:
    """Init scale for the mocked shock-exposure model (B, b_k, c_k)."""
    init_scale: float = 0.3


OBS_DIM = STATE_DIM + MarketSimConfig().n_assets + 2  # z_t (24) + w_{t-1} (4) + V_t (1) + DD_t (1) = 30


class MacroChaosEnv(gym.Env):
    """ChaosTradingEnv v1. obs = [z_t, w_{t-1}, V_t, DD_t]."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        pipeline: MacroChaosDataPipeline,
        episode_tensor: Optional[np.ndarray] = None,
        reward_config: Optional[RewardConfig] = None,
        market_config: Optional[MarketSimConfig] = None,
        shock_config: Optional[ShockConfig] = None,
        initial_capital: float = 1.0,
        l_max: float = 1.0,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.pipeline = pipeline
        self.reward_config = reward_config or RewardConfig()
        self.market_config = market_config or MarketSimConfig()
        self.shock_config = shock_config or ShockConfig()
        self.initial_capital = initial_capital
        self.l_max = l_max

        self._episode_tensor = (
            episode_tensor if episode_tensor is not None else pipeline.build_episode_tensor().numpy()
        )
        self.T = self._episode_tensor.shape[0]

        n = self.market_config.n_assets
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(STATE_DIM + n + 2,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(n,), dtype=np.float32)

        # Placeholder return model: R_{t+1} ~ N(mu, Sigma_t).
        mc = self.market_config
        self.mu = np.full(n, mc.mu_baseline)
        vol = np.full(n, mc.base_vol)
        corr = np.full((n, n), mc.base_corr) + np.eye(n) * (1.0 - mc.base_corr)
        self.base_cov = np.outer(vol, vol) * corr

        # Mocked shock-exposure model: B, b_k, c_k, fixed for this env's lifetime.
        rng = np.random.default_rng(seed)
        s = self.shock_config.init_scale
        self.B = rng.normal(0.0, s, size=(n, 3))
        self.b_geo = rng.normal(0.0, s, size=(GEO_LATENT_DIM,))
        self.b_weather = rng.normal(0.0, s, size=(WEATHER_LATENT_DIM,))
        self.b_port = rng.normal(0.0, s, size=(PORT_LATENT_DIM,))
        self.c = rng.normal(0.0, s, size=(3,))

        self._min_value = 1e-3 * initial_capital
        self.loss_history: deque = deque(maxlen=self.reward_config.cvar_window)

        self.t = 0
        self.V = initial_capital
        self.peak_value = initial_capital
        self.prev_weights = np.zeros(n, dtype=np.float64)
        self.drawdown = 0.0

    def _build_obs(self, z_t: np.ndarray) -> np.ndarray:
        """Concatenate [z_t, w_{t-1}, V_t, DD_t] per the state-space contract."""
        obs = np.concatenate([z_t, self.prev_weights, [self.V], [self.drawdown]])
        return obs.astype(np.float32)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        if options and options.get("resample_trajectory", False):
            self._episode_tensor = self.pipeline.build_episode_tensor().numpy()
            self.T = self._episode_tensor.shape[0]

        self.t = 0
        self.V = self.initial_capital
        self.peak_value = self.initial_capital
        self.prev_weights = np.zeros(self.market_config.n_assets, dtype=np.float64)
        self.drawdown = 0.0
        self.loss_history.clear()

        obs = self._build_obs(self._episode_tensor[self.t])
        info = {"timestep": 0, "portfolio_value": self.V, "drawdown": self.drawdown}
        return obs, info

    def _simulate_asset_returns(self, chaos_signal: float) -> Tuple[np.ndarray, np.ndarray]:
        """Placeholder R_{t+1}; swap freely, only (R, Sigma) is the contract."""
        mc = self.market_config
        mult = 1.0 + mc.chaos_vol_scale * chaos_signal
        Sigma_t = self.base_cov * (mult ** 2)
        R_next = self.np_random.multivariate_normal(self.mu, Sigma_t)
        return R_next, Sigma_t

    def _transaction_costs(self, w_t: np.ndarray) -> Tuple[float, float]:
        """TC_t: linear in L1 turnover. I_t: nonlinear, quadratic in turnover."""
        turnover = w_t - self.prev_weights
        tc = self.reward_config.kappa_tc * float(np.sum(np.abs(turnover))) * self.V
        impact = self.reward_config.kappa_impact * float(np.sum(turnover ** 2)) * self.V
        return tc, impact

    def _shock_penalty(self, z_t: np.ndarray, w_t: np.ndarray) -> float:
        """S_t = sum_k softplus(b_k . z_t^k + c_k) * |e_t,k|^1.5."""
        z_geo = z_t[:GEO_LATENT_DIM]
        z_weather = z_t[GEO_LATENT_DIM:GEO_LATENT_DIM + WEATHER_LATENT_DIM]
        z_port = z_t[GEO_LATENT_DIM + WEATHER_LATENT_DIM:]

        lambda_t = np.array([
            _softplus(float(self.b_geo @ z_geo) + self.c[0]),
            _softplus(float(self.b_weather @ z_weather) + self.c[1]),
            _softplus(float(self.b_port @ z_port) + self.c[2]),
        ])
        e_t = self.B.T @ w_t  # portfolio exposure per shock modality
        return float(np.sum(lambda_t * np.abs(e_t) ** 1.5))

    def step(self, action: np.ndarray):
        rc = self.reward_config
        z_t = self._episode_tensor[self.t]

        # Action projection: clip to [-1,1], then leverage-normalize to L_max.
        w_tilde = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        l1 = float(np.sum(np.abs(w_tilde)))
        w_t = w_tilde / max(1.0, l1 / self.l_max)

        # Chaos signal drives the placeholder return model's volatility.
        chaos_signal = math.tanh(float(np.linalg.norm(z_t)) / rc.chaos_scale_norm)
        R_next, Sigma_t = self._simulate_asset_returns(chaos_signal)
        TC_t, I_t = self._transaction_costs(w_t)

        # Dynamics: V_{t+1} = V_t(1 + w_t^T R_{t+1}) - TC_t - I_t.
        gross_return = float(w_t @ R_next)
        V_next = max(self.V * (1.0 + gross_return) - TC_t - I_t, self._min_value)

        # Term 1: log-growth.
        d_log_v = math.log(V_next) - math.log(self.V)

        # Term 2: sigma_t^2 = w_t^T Sigma_t w_t.
        sigma_p2 = float(w_t @ Sigma_t @ w_t)

        # Term 3: CVaR_0.95 from rolling realized (net-of-cost) losses.
        realized_loss = -(V_next / self.V - 1.0)
        self.loss_history.append(realized_loss)
        losses = np.sort(np.asarray(self.loss_history))[::-1]
        n_tail = max(1, int(np.ceil(rc.alpha * len(losses))))
        cvar_95 = float(np.mean(losses[:n_tail]))

        # Term 4: P_DD, soft quartic + hard breach indicator.
        self.peak_value = max(self.peak_value, V_next)
        drawdown = max(0.0, (self.peak_value - V_next) / self.peak_value)
        breach = drawdown >= rc.dd_limit
        p_dd = rc.lambda_soft * (drawdown / rc.dd_limit) ** 4 + rc.lambda_breach * float(breach)

        # Term 5: multi-dimensional shock tensor.
        S_t = self._shock_penalty(z_t, w_t)

        reward = (
            d_log_v
            - rc.lambda_var * sigma_p2
            - rc.lambda_cvar * cvar_95
            - p_dd
            - rc.lambda_sh * S_t
            - rc.lambda_tc * TC_t
        )

        self.V = V_next
        self.prev_weights = w_t
        self.drawdown = drawdown
        self.t += 1

        terminated = bool(breach)          # true terminal state: DD limit breached
        truncated = self.t >= self.T - 1   # artificial cutoff: out of historical data
        next_obs = self._build_obs(self._episode_tensor[min(self.t, self.T - 1)])

        info = {
            "timestep": self.t,
            "chaos_signal": chaos_signal,
            "portfolio_value": self.V,
            "drawdown": self.drawdown,
            "dd_breach": breach,
            "reward_components": {
                "d_log_v": d_log_v,
                "variance_penalty": rc.lambda_var * sigma_p2,
                "cvar_penalty": rc.lambda_cvar * cvar_95,
                "dd_penalty": p_dd,
                "shock_penalty": rc.lambda_sh * S_t,
                "tc_penalty": rc.lambda_tc * TC_t,
            },
        }
        return next_obs, float(reward), terminated, truncated, info

    def render(self):
        pass