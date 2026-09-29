"""
Energy-Adaptive Cross-Modal Attention (EACMA) -- Section 3.4 / 3.5.

Implements:
  Eq (7)-(8): modality-level, energy-aware scoring f_theta(h_i, e_i)
  Eq (9):     temporal attention with a PRE-softmax entropy penalty
              (this is the corrected formulation -- the penalty is
              applied to the raw score s_i,j before softmax, which is
              what guarantees sum_j beta_i,j = 1 by construction).
  Eq (11):    EACMA cross-modal attention with the real-time energy
              feedback term beta_energy * log(e_i^t).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ModalityScoringFunction(nn.Module):
    """f_theta(h_i, e_i) = W^T tanh(W_h h_i + W_e e_i + b)   -- Eq (8)"""

    def __init__(self, feature_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.w_h = nn.Linear(feature_dim, hidden_dim, bias=False)
        self.w_e = nn.Linear(1, hidden_dim, bias=True)
        self.w = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, h_i: torch.Tensor, e_i: torch.Tensor) -> torch.Tensor:
        # h_i: [B, D], e_i: [B, 1]
        return self.w(torch.tanh(self.w_h(h_i) + self.w_e(e_i))).squeeze(-1)


class TemporalScoringFunction(nn.Module):
    """
    g_phi(h_i,t) -- two-layer MLP, hidden dim 64, tanh activation,
    linear output projection. Trained jointly under Eq (15).
    """

    def __init__(self, feature_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, h_i_t: torch.Tensor) -> torch.Tensor:
        return self.net(h_i_t).squeeze(-1)  # [B, T]


class TemporalEntropyAttention(nn.Module):
    """
    Eq (9), corrected two-step formulation:
        s*_{i,j} = s_{i,j} - gamma * log( sum_t' exp(g_phi(h_{i,t'})) )
        beta_{i,j} = softmax_j(s*_{i,j})
    The entropy-style penalty is applied BEFORE the softmax, so
    sum_j beta_{i,j} == 1 holds by construction (verified in tests).
    """

    def __init__(self, feature_dim: int, gamma: float = 0.1, hidden_dim: int = 64):
        super().__init__()
        self.g_phi = TemporalScoringFunction(feature_dim, hidden_dim)
        self.gamma = gamma

    def forward(self, h_i: torch.Tensor) -> torch.Tensor:
        # h_i: [B, T, D]
        s = self.g_phi(h_i)  # [B, T]  == g_phi(h_{i,t})
        penalty = self.gamma * torch.logsumexp(s, dim=-1, keepdim=True)
        s_star = s - penalty
        beta = F.softmax(s_star, dim=-1)  # sums to 1 by construction
        return beta


class EACMABlock(nn.Module):
    """
    Full EACMA cross-modal attention block, Eq (11):
        alpha_i^t = softmax( (E_budget^t (.) W_energy (.) Q_i K_j^T) / sqrt(d_k)
                             + beta_energy * log(e_i^t) )
    """

    def __init__(self, feature_dim: int, num_modalities: int, d_k: int = 32):
        super().__init__()
        self.d_k = d_k
        self.q_proj = nn.Linear(feature_dim, d_k)
        self.k_proj = nn.Linear(feature_dim, d_k)
        self.w_energy = nn.Parameter(torch.ones(num_modalities))
        self.beta_energy = nn.Parameter(torch.tensor(0.1))  # Table S2 default

    def forward(self, feats: torch.Tensor, e_budget: torch.Tensor,
                e_it: torch.Tensor) -> torch.Tensor:
        # feats: [B, M, D] per-modality features; e_budget: [B, 1];
        # e_it: [B, M] real-time energy consumption per modality
        q = self.q_proj(feats)
        k = self.k_proj(feats)
        scores = torch.einsum("bid,bjd->bij", q, k) / (self.d_k ** 0.5)
        scores = scores * e_budget.unsqueeze(-1) * self.w_energy.view(1, -1, 1)
        energy_term = self.beta_energy * torch.log(e_it.clamp_min(1e-6))
        scores = scores + energy_term.unsqueeze(1)
        return F.softmax(scores, dim=-1)
