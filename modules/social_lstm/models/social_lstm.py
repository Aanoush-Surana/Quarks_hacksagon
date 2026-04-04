"""
Social LSTM – Alahi et al. (CVPR 2016), adapted for road-scene trajectory prediction.

Architecture
────────────
For each agent i at time t:
  1. Embed (x, y) displacement with a linear layer  →  e_t^i
  2. Build occupancy grid H_t^i: a binary grid of neighbours in a fixed radius
  3. Pool hidden states of neighbours weighted by H_t^i  →  H_pool_t^i
  4. LSTM input  = [e_t^i ; H_pool_t^i]
  5. LSTM output → decode (μx, μy, σx, σy, ρ)  (bivariate Gaussian)

Loss: negative log-likelihood of ground-truth next position under the predicted
      bivariate Gaussian (only for observed / non-NaN targets).

References
──────────
Alahi et al., "Social Force for Pedestrian Navigation", CVPR 2016
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


# ── Occupancy-grid social pooling ────────────────────────────────────────────

class SocialPooling(nn.Module):
    """
    Build the social pooling tensor for each agent.

    For agent i, we discretise a neighbourhood_size × neighbourhood_size grid
    around its current position. Each cell (r, c) in the grid contains the
    sum of hidden states of all other agents whose positions fall in that cell.

    Parameters
    ----------
    hidden_dim        : LSTM hidden state size
    grid_size         : number of cells along each grid axis (N × N)
    neighbourhood_size: physical radius of the grid in metres (or px)
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        grid_size: int = 8,
        neighbourhood_size: float = 32.0,
    ):
        super().__init__()
        self.hidden_dim   = hidden_dim
        self.grid_size    = grid_size
        self.nb_size      = neighbourhood_size

        # Project grid → pooled vector
        self.pool_proj = nn.Linear(
            grid_size * grid_size * hidden_dim, hidden_dim
        )

    def forward(
        self,
        hidden:   torch.Tensor,   # (A, hidden_dim)
        pos:      torch.Tensor,   # (A, 2)  current positions
        mask:     torch.Tensor,   # (A,)    bool – which agents are real
    ) -> torch.Tensor:            # (A, hidden_dim)
        """
        Returns the pooled social context vector for each agent.
        """
        A = hidden.size(0)
        G = self.grid_size
        N = self.nb_size
        cell = (2 * N) / G    # physical size of one cell

        # Pool result: (A, hidden_dim)
        pooled = torch.zeros(A, self.hidden_dim, device=hidden.device)

        for i in range(A):
            if not mask[i]:
                continue

            # Skip if this agent's own position is NaN
            if torch.isnan(pos[i]).any():
                continue

            # Relative positions of all other agents w.r.t. agent i
            rel = pos - pos[i].unsqueeze(0)   # (A, 2)
            grid = torch.zeros(G * G * self.hidden_dim, device=hidden.device)

            for j in range(A):
                if i == j or not mask[j]:
                    continue
                rx, ry = rel[j, 0].item(), rel[j, 1].item()
                # Skip if either relative coordinate is NaN or out of range
                if rx != rx or ry != ry:   # NaN check (faster than math.isnan)
                    continue
                if abs(rx) >= N or abs(ry) >= N:
                    continue
                # Cell indices
                col = int((rx + N) / cell)
                row = int((ry + N) / cell)
                col = min(max(col, 0), G - 1)
                row = min(max(row, 0), G - 1)
                idx = (row * G + col) * self.hidden_dim
                grid[idx: idx + self.hidden_dim] += hidden[j]

            pooled[i] = self.pool_proj(grid)

        return pooled                           # (A, hidden_dim)


# ── Social LSTM ──────────────────────────────────────────────────────────────

class SocialLSTM(nn.Module):
    """
    Social LSTM trajectory prediction model.

    Parameters
    ----------
    embedding_dim       : size of the (x,y) input embedding
    hidden_dim          : LSTM hidden / cell state size
    pred_len            : number of future time steps to predict
    dropout             : dropout probability (applied to LSTM input)
    grid_size           : social pooling grid dimension
    neighbourhood_size  : social pooling radius (same units as input coords)
    """

    def __init__(
        self,
        embedding_dim:      int   = 64,
        hidden_dim:         int   = 128,
        pred_len:           int   = 12,
        dropout:            float = 0.0,
        grid_size:          int   = 8,
        neighbourhood_size: float = 32.0,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim    = hidden_dim
        self.pred_len      = pred_len

        # (x, y) → embedding
        self.input_embed = nn.Sequential(
            nn.Linear(2, embedding_dim),
            nn.ReLU(),
        )

        # Social pooling module
        self.social_pool = SocialPooling(
            hidden_dim=hidden_dim,
            grid_size=grid_size,
            neighbourhood_size=neighbourhood_size,
        )

        # LSTM: input = embedding + pooled social context
        self.lstm = nn.LSTMCell(
            input_size  = embedding_dim + hidden_dim,
            hidden_size = hidden_dim,
        )

        self.dropout = nn.Dropout(dropout)

        # Decode LSTM hidden state → bivariate Gaussian parameters
        # Output: (μx, μy, log_σx, log_σy, tanh_ρ)
        self.output_layer = nn.Linear(hidden_dim, 5)

    # ── helpers ─────────────────────────────────────────────────────────────

    def _init_hidden(self, A: int, device: torch.device):
        h = torch.zeros(A, self.hidden_dim, device=device)
        c = torch.zeros(A, self.hidden_dim, device=device)
        return h, c

    @staticmethod
    def _decode_gaussian(
        raw: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Convert raw 5-d output to (mu, sigma, rho).
          mu    : (A, 2)
          sigma : (A, 2)  – standard deviations (> 0)
          rho   : (A,)    – correlation coefficient (−1, 1)
        """
        mu    = raw[:, :2]
        sigma = torch.exp(raw[:, 2:4]) + 1e-6      # log-space → positive
        rho   = torch.tanh(raw[:, 4])
        return mu, sigma, rho

    # ── forward ─────────────────────────────────────────────────────────────

    def forward(
        self,
        obs:  torch.Tensor,   # (T_obs, A, 2)
        mask: torch.Tensor,   # (A,)  bool
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode observation sequence, then auto-regressively decode pred_len steps.

        Returns
        -------
        pred_mu    : (T_pred, A, 2)   mean predicted positions
        pred_sigma : (T_pred, A, 2)   standard deviations
        pred_rho   : (T_pred, A)      correlation coefficients
        """
        T_obs, A, _ = obs.shape
        device       = obs.device

        h, c = self._init_hidden(A, device)

        # Replace NaN positions (partially-observed agents) with zeros
        # so they don't corrupt the embedding or social pooling
        obs_clean = obs.clone()
        obs_clean[torch.isnan(obs_clean)] = 0.0

        # ── encode observation ──────────────────────────────────────────────
        for t in range(T_obs):
            pos  = obs_clean[t]                             # (A, 2)
            emb  = self.dropout(self.input_embed(pos))      # (A, emb)
            soc  = self.social_pool(h, pos, mask)           # (A, hidden)
            inp  = torch.cat([emb, soc], dim=-1)            # (A, emb+hidden)
            h, c = self.lstm(inp, (h, c))

        # ── decode prediction ───────────────────────────────────────────────
        pred_mu    = []
        pred_sigma = []
        pred_rho   = []

        # Start from last clean observed position; zero inactive agents
        cur_pos = obs_clean[-1].clone()    # (A, 2)
        cur_pos[~mask] = 0.0

        for _ in range(self.pred_len):
            emb  = self.dropout(self.input_embed(cur_pos))
            soc  = self.social_pool(h, cur_pos, mask)
            inp  = torch.cat([emb, soc], dim=-1)
            h, c = self.lstm(inp, (h, c))

            raw        = self.output_layer(h)               # (A, 5)
            mu, sg, rh = self._decode_gaussian(raw)

            # Only advance position for active agents
            delta = mu * mask.unsqueeze(-1).float()
            cur_pos = cur_pos + delta

            pred_mu.append(cur_pos)
            pred_sigma.append(sg)
            pred_rho.append(rh)

        pred_mu    = torch.stack(pred_mu,    dim=0)   # (T_pred, A, 2)
        pred_sigma = torch.stack(pred_sigma, dim=0)   # (T_pred, A, 2)
        pred_rho   = torch.stack(pred_rho,   dim=0)   # (T_pred, A)

        return pred_mu, pred_sigma, pred_rho


# ── Loss ─────────────────────────────────────────────────────────────────────

def bivariate_nll_loss(
    mu:     torch.Tensor,   # (T, A, 2)
    sigma:  torch.Tensor,   # (T, A, 2)
    rho:    torch.Tensor,   # (T, A)
    target: torch.Tensor,   # (T, A, 2)
    mask:   torch.Tensor,   # (A,) bool
) -> torch.Tensor:
    """
    Negative log-likelihood of a bivariate Gaussian.
    NaN targets (partially-observed agents in pred horizon) are excluded.
    """
    T, A, _ = target.shape

    # Mask out agents not present and NaN positions
    valid_agent  = mask.unsqueeze(0).expand(T, A)          # (T, A)
    not_nan      = ~torch.isnan(target).any(dim=-1)         # (T, A)
    valid        = valid_agent & not_nan                    # (T, A)

    if valid.sum() == 0:
        return torch.tensor(0.0, requires_grad=True, device=mu.device)

    sx  = sigma[..., 0]        # (T, A)
    sy  = sigma[..., 1]        # (T, A)

    dx  = (target[..., 0] - mu[..., 0]) / (sx + 1e-6)
    dy  = (target[..., 1] - mu[..., 1]) / (sy + 1e-6)

    r2  = 1 - rho ** 2 + 1e-6
    z   = dx**2 + dy**2 - 2 * rho * dx * dy

    log_p = (
        -0.5 * z / r2
        - torch.log(2 * np.pi * sx * sy * torch.sqrt(r2))
    )

    loss = -log_p[valid].mean()
    return loss


# ── Convenience: sample from predicted distribution ──────────────────────────

def sample_trajectories(
    mu:     torch.Tensor,    # (T_pred, A, 2)
    sigma:  torch.Tensor,    # (T_pred, A, 2)
    rho:    torch.Tensor,    # (T_pred, A)
    n_samples: int = 1,
) -> torch.Tensor:           # (n_samples, T_pred, A, 2)
    """
    Draw samples from the predicted bivariate Gaussian at each step.
    Useful for stochastic trajectory roll-outs.
    """
    T, A, _ = mu.shape
    samples  = torch.zeros(n_samples, T, A, 2, device=mu.device)

    for t in range(T):
        sx = sigma[t, :, 0]   # (A,)
        sy = sigma[t, :, 1]   # (A,)
        r  = rho[t]            # (A,)

        # Cholesky of the 2×2 covariance for each agent
        L11 = sx
        L21 = r * sy
        L22 = sy * torch.sqrt(1 - r**2 + 1e-6)

        eps1 = torch.randn(n_samples, A, device=mu.device)
        eps2 = torch.randn(n_samples, A, device=mu.device)

        samples[:, t, :, 0] = mu[t, :, 0] + L11 * eps1
        samples[:, t, :, 1] = mu[t, :, 1] + L21 * eps1 + L22 * eps2

    return samples