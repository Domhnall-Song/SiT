"""CouetteDiffusion path for the SiT transport framework.

Implements two interpolant variants derived from the non-stationary Couette
flow / heat equation:

  * "time" — scalar mixing schedule using the erfc spatial profile,
    alpha(t) = erfc(eta_max * t), sigma(t) = sqrt(1 - alpha**2).

  * "freq" — anisotropic per-Fourier-mode heat-kernel decay,
    H(k, t) = exp(-nu * K^2 * tau(t)) applied through rfftn/irfftn.

Convention: the *public* API of this class follows the prompt's SiT
convention with t in [0, 1], t=0 being clean data and t=1 being pure noise.
This is the opposite of the existing ICPlan / LinearPath in this fork
(where t=0 is noise and t=1 is data); the convention swap is applied only
at the integration boundary in ``plan_transport``.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
from torch import Tensor


def _align(t: Tensor, x: Tensor) -> Tensor:
    """Reshape ``t`` of shape (B,) to broadcast against ``x``."""
    if t.dim() == 0:
        t = t.unsqueeze(0)
    return t.view(t.size(0), *([1] * (x.dim() - 1)))


class CouettePath:
    def __init__(
        self,
        mode: str = "time",
        eta_max: float = 3.0,
        nu: float = 1.0,
        tau_max: Optional[float] = None,
        alpha_min: float = 1e-4,
        freq_axes: Tuple[int, ...] = (-2, -1),
    ):
        if mode not in ("time", "freq"):
            raise ValueError(f"Unknown CouettePath mode={mode!r}")
        self.mode = mode
        self.eta_max = float(eta_max)
        self.nu = float(nu)
        self._tau_max_user = None if tau_max is None else float(tau_max)
        self.alpha_min = float(alpha_min)
        self.freq_axes = tuple(int(a) for a in freq_axes)
        self._k2_cache: dict = {}

    # ---------------------------------------------------------------
    # Time-domain primitives
    # ---------------------------------------------------------------
    def _alpha_time(self, t: Tensor) -> Tensor:
        return torch.special.erfc(self.eta_max * t)

    def _d_alpha_time(self, t: Tensor) -> Tensor:
        c = -2.0 * self.eta_max / math.sqrt(math.pi)
        return c * torch.exp(-(self.eta_max ** 2) * t * t)

    def _sigma_time(self, t: Tensor) -> Tensor:
        a = self._alpha_time(t)
        return torch.sqrt(torch.clamp(1.0 - a * a, min=0.0))

    def _d_sigma_time(self, t: Tensor) -> Tensor:
        a = self._alpha_time(t)
        da = self._d_alpha_time(t)
        sigma = self._sigma_time(t).clamp_min(1e-6)
        return -a * da / sigma

    # ---------------------------------------------------------------
    # Frequency-domain helpers
    # ---------------------------------------------------------------
    def _tau_max(self, x_shape) -> float:
        if self._tau_max_user is not None:
            return self._tau_max_user
        # Worst-case K^2 = d/4 from rfftn over |freq_axes| axes (see app. A.3).
        d = len(self.freq_axes)
        return -4.0 * math.log(self.alpha_min) / (self.nu * d)

    def _resolved_axes(self, ndim: int):
        return [a if a >= 0 else ndim + a for a in self.freq_axes]

    def _freq_k2(self, x_shape, device, dtype) -> Tensor:
        key = (tuple(x_shape), str(device), dtype)
        cached = self._k2_cache.get(key)
        if cached is not None:
            return cached
        ndim = len(x_shape)
        axes = self._resolved_axes(ndim)
        last_axis = axes[-1]
        K2 = torch.zeros([1] * ndim, device=device, dtype=dtype)
        for ax in axes:
            N = x_shape[ax]
            if ax == last_axis:
                # rfft along this axis: bins 0 .. N//2
                k = torch.arange(N // 2 + 1, device=device, dtype=dtype)
            else:
                # standard fft along this axis
                k = torch.fft.fftfreq(N, d=1.0 / N).to(device=device, dtype=dtype)
            shape = [1] * ndim
            shape[ax] = k.numel()
            K2 = K2 + (k.view(shape) / N) ** 2
        self._k2_cache[key] = K2
        return K2

    def freq_kernel(self, x_shape, t: Tensor, device, dtype) -> Tensor:
        """H(k, t) broadcastable to ``rfftn(x_data)`` over ``freq_axes``."""
        if self.mode != "freq":
            raise NotImplementedError("freq_kernel is freq-mode only")
        K2 = self._freq_k2(x_shape, device, dtype)
        tau_max = self._tau_max(x_shape)
        ndim = len(x_shape)
        if t.dim() == 0:
            t = t.unsqueeze(0)
        t_b = t.view(-1, *([1] * (ndim - 1))).to(device=device, dtype=dtype)
        return torch.exp(-self.nu * K2 * (t_b * tau_max))

    def _freq_arrays(self, x_data: Tensor, t: Tensor):
        """Return (axes, spatial_sizes, K2, tau_max, H, sqrt_term)."""
        x_shape = x_data.shape
        ndim = len(x_shape)
        axes = self._resolved_axes(ndim)
        spatial_sizes = [x_shape[a] for a in axes]
        device, dtype = x_data.device, x_data.dtype
        K2 = self._freq_k2(x_shape, device, dtype)
        tau_max = self._tau_max(x_shape)
        if t.dim() == 0:
            t = t.unsqueeze(0)
        t_b = t.view(-1, *([1] * (ndim - 1))).to(device=device, dtype=dtype)
        H = torch.exp(-self.nu * K2 * (t_b * tau_max))
        sqrt_term = torch.sqrt(torch.clamp(1.0 - H * H, min=0.0))
        return axes, spatial_sizes, K2, tau_max, H, sqrt_term

    # ---------------------------------------------------------------
    # Public scalar API (prompt convention: t=0 -> data, t=1 -> noise)
    # ---------------------------------------------------------------
    def compute_alpha_t(self, t: Tensor) -> Tensor:
        """Scalar alpha(t). In freq mode, returns the RMS over modes."""
        if self.mode == "time":
            return self._alpha_time(t)
        # freq mode: RMS over modes (acts as an effective scalar schedule
        # used only for train_eps / sample_eps clamping; the actual
        # per-mode corruption happens inside ``q_sample`` / ``plan``).
        return self._alpha_time(t)

    def compute_sigma_t(self, t: Tensor) -> Tensor:
        if self.mode == "time":
            return self._sigma_time(t)
        return self._sigma_time(t)

    def compute_d_alpha_t(self, t: Tensor) -> Tensor:
        return self._d_alpha_time(t)

    def compute_d_sigma_t(self, t: Tensor) -> Tensor:
        return self._d_sigma_time(t)

    # ---------------------------------------------------------------
    # Forward marginal and velocity target
    # ---------------------------------------------------------------
    def q_sample(self, x_data: Tensor, t: Tensor, eps: Tensor) -> Tensor:
        if self.mode == "time":
            a = _align(self._alpha_time(t), x_data)
            s = _align(self._sigma_time(t), x_data)
            return a * x_data + s * eps
        axes, spatial_sizes, _, _, H, sqrt_term = self._freq_arrays(x_data, t)
        X = torch.fft.rfftn(x_data, dim=axes)
        E = torch.fft.rfftn(eps, dim=axes)
        X_t = H * X + sqrt_term * E
        return torch.fft.irfftn(X_t, s=spatial_sizes, dim=axes)

    def plan(self, t: Tensor, x_data: Tensor, x_noise: Tensor) -> Tuple[Tensor, Tensor]:
        """Return (x_t, velocity_target) under the prompt convention."""
        if self.mode == "time":
            a = _align(self._alpha_time(t), x_data)
            s = _align(self._sigma_time(t), x_data)
            da = _align(self._d_alpha_time(t), x_data)
            ds = _align(self._d_sigma_time(t), x_data)
            x_t = a * x_data + s * x_noise
            v = da * x_data + ds * x_noise
            return x_t, v

        axes, spatial_sizes, K2, tau_max, H, sqrt_term = self._freq_arrays(x_data, t)
        X = torch.fft.rfftn(x_data, dim=axes)
        E = torch.fft.rfftn(x_noise, dim=axes)
        X_t = H * X + sqrt_term * E
        x_t = torch.fft.irfftn(X_t, s=spatial_sizes, dim=axes)
        # d/dt H = -nu * K^2 * tau_max * H
        dH = -self.nu * K2 * tau_max * H
        # d/dt sqrt(1 - H^2) = -H * dH / sqrt(1 - H^2)
        dsqrt = -H * dH / sqrt_term.clamp_min(1e-6)
        V_freq = dH * X + dsqrt * E
        v = torch.fft.irfftn(V_freq, s=spatial_sizes, dim=axes)
        return x_t, v

    # ---------------------------------------------------------------
    # Conversions among velocity / score / noise (time mode)
    # ---------------------------------------------------------------
    # Derivation for time mode. Let a = alpha(t), s = sigma(t),
    # da = alpha'(t), ds = sigma'(t). Because s = sqrt(1 - a^2) we get
    # ds = -a*da/s, hence det(J) = a*ds - s*da = -da/s. The closed form
    # of the inverse map (x_t, v) -> (x_data, eps) simplifies to
    #     eps = s * (da * x_t - a * v) / da
    #     score = -eps / s = (a*v - da*x_t) / da = (a/da)*v - x_t
    #     v_from_score = (score + x_t) * (da / a)
    # which uses only well-conditioned ratios in (0, 1).
    def get_score_from_velocity(self, v: Tensor, x_t: Tensor, t: Tensor) -> Tensor:
        if self.mode == "time":
            a = _align(self._alpha_time(t), x_t)
            da = _align(self._d_alpha_time(t), x_t)
            return (a * v - da * x_t) / da
        return self._freq_score_from_velocity(v, x_t, t)

    def get_noise_from_velocity(self, v: Tensor, x_t: Tensor, t: Tensor) -> Tensor:
        if self.mode == "time":
            a = _align(self._alpha_time(t), x_t)
            s = _align(self._sigma_time(t), x_t)
            da = _align(self._d_alpha_time(t), x_t)
            return s * (da * x_t - a * v) / da
        return self._freq_noise_from_velocity(v, x_t, t)

    def get_velocity_from_score(self, score: Tensor, x_t: Tensor, t: Tensor) -> Tensor:
        if self.mode == "time":
            a = _align(self._alpha_time(t), x_t).clamp_min(1e-6)
            da = _align(self._d_alpha_time(t), x_t)
            return (score + x_t) * (da / a)
        return self._freq_velocity_from_score(score, x_t, t)

    # ---------------- freq-mode conversions (per-mode) -------------
    def _freq_score_from_velocity(self, v, x_t, t):
        axes, spatial_sizes, K2, tau_max, H, sqrt_term = self._freq_arrays(x_t, t)
        V = torch.fft.rfftn(v, dim=axes)
        X_t = torch.fft.rfftn(x_t, dim=axes)
        dH = -self.nu * K2 * tau_max * H
        # eps_k = sqrt(1-H^2) * (H*V - dH*X_t) / det, with det = -dH/sqrt(1-H^2).
        H_safe = H.clamp_min(1e-6)
        # Score in freq: -eps_k / sqrt(1-H^2_k); equivalently (H*V - dH*X_t)/(dH).
        dH_safe = torch.where(dH.abs() < 1e-12, torch.full_like(dH, 1e-12), dH)
        S_freq = (H * V - dH * X_t) / dH_safe
        # At DC (dH==0, sqrt==0), the score is degenerate; set to 0.
        S_freq = torch.where(dH.abs() < 1e-12, torch.zeros_like(S_freq), S_freq)
        return torch.fft.irfftn(S_freq, s=spatial_sizes, dim=axes)

    def _freq_noise_from_velocity(self, v, x_t, t):
        axes, spatial_sizes, K2, tau_max, H, sqrt_term = self._freq_arrays(x_t, t)
        V = torch.fft.rfftn(v, dim=axes)
        X_t = torch.fft.rfftn(x_t, dim=axes)
        dH = -self.nu * K2 * tau_max * H
        dsqrt = -H * dH / sqrt_term.clamp_min(1e-6)
        det = H * dsqrt - sqrt_term * dH
        det_safe = torch.where(det.abs() < 1e-12, torch.full_like(det, 1e-12), det)
        E_freq = (H * V - dH * X_t) / det_safe
        E_freq = torch.where(det.abs() < 1e-12, torch.zeros_like(E_freq), E_freq)
        return torch.fft.irfftn(E_freq, s=spatial_sizes, dim=axes)

    def _freq_velocity_from_score(self, score, x_t, t):
        axes, spatial_sizes, K2, tau_max, H, sqrt_term = self._freq_arrays(x_t, t)
        S = torch.fft.rfftn(score, dim=axes)
        X_t = torch.fft.rfftn(x_t, dim=axes)
        dH = -self.nu * K2 * tau_max * H
        # From score = (H*V - dH*X_t)/dH  ->  V = (score*dH + dH*X_t)/H = dH*(score+X_t)/H
        H_safe = H.clamp_min(1e-6)
        V_freq = dH * (S + X_t) / H_safe
        return torch.fft.irfftn(V_freq, s=spatial_sizes, dim=axes)

    # ---------------------------------------------------------------
    # ICPlan-compatible adapter used by the existing Transport
    # ---------------------------------------------------------------
    # The existing Transport / Sampler use the *opposite* time convention
    # (t=0 -> noise, t=1 -> data) and unpack (alpha, d_alpha) / (sigma,
    # d_sigma) tuples. The methods below adapt CouettePath so that
    # ``create_transport(path_type="Couette", ...)`` returns a Transport
    # whose path_sampler matches what the Transport code expects, without
    # changing the public test API above.
    def plan_transport(self, t: Tensor, x0: Tensor, x1: Tensor):
        """Adapter for Transport.training_losses.

        Transport convention: x0 = noise, x1 = data, t in [0, 1] with
        t=0 -> noise and t=1 -> data. We map to the prompt convention by
        s = 1 - t, call ``plan(s, x_data=x1, x_noise=x0)`` and negate the
        velocity (chain rule: dx/dt = -dx/ds).
        """
        s = 1.0 - t
        x_t, v_prompt = self.plan(s, x1, x0)
        return t, x_t, -v_prompt

    def _tuple_alpha(self, t: Tensor):
        """ICPlan-style (alpha, d_alpha) under Transport's convention."""
        s = 1.0 - t
        # alpha_Transport(t) := alpha_prompt(1 - t); chain rule flips sign.
        alpha = self._alpha_time(s)
        d_alpha = -self._d_alpha_time(s)
        return alpha, d_alpha

    def _tuple_sigma(self, t: Tensor):
        s = 1.0 - t
        sigma = self._sigma_time(s)
        d_sigma = -self._d_sigma_time(s)
        return sigma, d_sigma
