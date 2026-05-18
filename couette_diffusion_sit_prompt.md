Please make the modifications strictly within the boundaries of Section 3 of the document. First, implement the §4 test, and then write the main code (Test-Driven Development).
# Coding Prompt — CouetteDiffusion path for the SiT transport framework

## 0. Task

You are extending an existing **SiT (Scalable Interpolant Transformer)** codebase. The codebase already exposes stochastic-interpolant paths through

```python
create_transport(path_type, prediction, loss_weight, train_eps, sample_eps)
```

and a `Sampler` wrapper for ODE/SDE generation (see `sample.py`). The transformer backbone (`models.py`, class `SiT`) accepts a continuous time `t` via `TimestepEmbedder` and is **not modified** by this task.

**Goal**: add a new path type `"Couette"` whose forward (corruption) process is derived from the **non-stationary Couette flow** (Stokes' first problem) instead of the conventional Linear / GVP / VP interpolants. Requirements:

1. The new path must plug into `create_transport(path_type="Couette", prediction=..., ...)` with **no changes** to `models.py` or `sample.py`.
2. Support all three SiT prediction targets, configurable via the existing `prediction` argument: `"velocity"`, `"score"`, `"noise"`.
3. Provide two variants, selected by a sub-flag `couette_mode ∈ {"time", "freq"}`:
   * `"time"`: scalar interpolant with an `erfc`-based α(t), σ(t).
   * `"freq"`: anisotropic per-Fourier-mode heat-kernel decay applied through FFT.
4. Pass the existing transport sanity tests (boundary, monotonicity, derivative consistency) and the new tests documented in §4.

The deliverable is code only — no paper, no training run.

## 1. Physical background (brief)

The 1D heat equation
$$\partial_t u = \nu\, \partial_y^2 u$$
with moving-wall BC $u(0,t) = U$, $u(y>0, 0) = 0$, has the closed-form solution
$$u(y,t) = U\,\mathrm{erfc}\!\left(\frac{y}{2\sqrt{\nu t}}\right),\qquad \delta(t)=\sqrt{\nu t}.$$

The Cauchy version (initial-value, infinite domain) decays in Fourier space as
$$\hat u(k,t) = \hat u(k,0)\,\exp(-\nu k^2 t).$$

CouetteDiffusion uses (a) the **spatial `erfc` profile** as a scalar SiT interpolant schedule, and (b) the **Fourier-mode decay** as a frequency-domain forward process. The boundary-layer thickness $\delta(t)=\sqrt{\nu t}$ is the natural time variable.

## 2. Mathematical specification

Use the **SiT convention** matching the existing `LinearPath`: $t \in [0,1]$ with $t=0$ being **clean data** and $t=1$ being **pure noise**. Verify by inspecting the existing `LinearPath` (where $\alpha(t)=1-t$, $\sigma(t)=t$). If your local fork uses the opposite convention, apply $t \leftarrow 1-t$ inside `CouettePath` only.

### 2.1 Time-domain interpolant (erfc-mixing)

One hyperparameter $\eta_{\max} > 0$ (default `3.0`) controls how deep the boundary layer extends:

$$
\boxed{\ \alpha(t) = \mathrm{erfc}(\eta_{\max}\, t),\quad \sigma(t) = \sqrt{1 - \alpha(t)^2}\ }
$$

Forward marginal:
$$x_t = \alpha(t)\, x_{\text{data}} + \sigma(t)\, \varepsilon,\qquad \varepsilon \sim \mathcal{N}(0, I).$$

Boundary conditions:
* $\alpha(0)=1,\ \sigma(0)=0$ (clean data).
* $\alpha(1)=\mathrm{erfc}(\eta_{\max})\approx 0,\ \sigma(1)\approx 1$. At $\eta_{\max}=3$: $\alpha(1)\approx 2.21\times 10^{-5}$.

Closed-form derivatives:
$$
\alpha'(t) = -\frac{2\eta_{\max}}{\sqrt{\pi}}\exp\!\big(-\eta_{\max}^2 t^2\big),
\qquad
\sigma'(t) = -\frac{\alpha(t)\,\alpha'(t)}{\sigma(t)}\ \ \text{for } \sigma(t)>0.
$$

**Numerical edge**: $\sigma'(t)\to +\infty$ as $t\to 0^+$ (same singularity class as VP / GVP paths). Honor the existing `train_eps` and `sample_eps` clamps; in `compute_d_sigma_t` add `sigma = sigma.clamp_min(1e-6)` before the division.

### 2.2 Frequency-domain interpolant (heat kernel)

A configurable axis set `freq_axes` (default `(-2, -1)` for image tensors `(B, C, H, W)`; pass `(-1,)` for 1D trajectory tensors). For each Fourier coefficient indexed by wavevector $k=(k_1,\ldots,k_d)$ over `freq_axes`:

$$
K^2 := \sum_i \left(\frac{k_i}{N_i}\right)^2,\qquad
H(k,t) = \exp\!\big(-\nu\, K^2\, \tau(t)\big).
$$

Per-mode forward:
$$
\hat X_t(k) = H(k,t)\,\hat X_{\text{data}}(k) + \sqrt{1 - H(k,t)^2}\,\hat\eta(k),
\qquad
x_t = \mathrm{iFFT}\big[\hat X_t\big].
$$

Time-scaling: pick $\tau(t)=t\,\tau_{\max}$ where $\tau_{\max}$ is set so the **maximum present wavenumber** is fully decayed at $t=1$:
$$\tau_{\max} = -\frac{\log\alpha_{\min}}{\nu\, K^2_{\max}}, \quad \alpha_{\min}=10^{-4}\ \text{(default)}.$$
This makes $K^2_{\max}\cdot \nu \cdot \tau_{\max} = -\log\alpha_{\min}$, i.e., the bandwidth-limited mode reaches $\alpha_{\min}$ at $t=1$.

This variant is **anisotropic**: high-frequency components (oscillations) corrupt first, low-frequency structure persists longest — directly analogous to image blurring or trajectory smoothing.

### 2.3 Velocity / score / noise conversion

For the time mode, the velocity field needed for `prediction="velocity"` is
$$v(x_t, t) = \alpha'(t)\, x_{\text{data}} + \sigma'(t)\, \varepsilon.$$

For the freq mode, $\alpha$ and $\sigma$ become **frequency-dependent**: replace by $H(k,t)$ and $\sqrt{1-H(k,t)^2}$ respectively, and compute the per-mode velocity in FFT space then iFFT.

Conversion among the three prediction modes uses the standard identities
$$
\varepsilon = \frac{x_t - \alpha(t)\,x_0}{\sigma(t)},\qquad
\text{score} = -\frac{\varepsilon}{\sigma(t)},
$$
which must work mode-by-mode in the freq variant.

## 3. Concrete implementation

### 3.1 New file: `transport/couette.py`

Create a `CouettePath` class whose method names **match the existing path classes** (`LinearPath`, `GVPPath`, `VPPath` in `transport/path.py`) so `Transport` and `Sampler` need no changes.

Required public API:

```python
class CouettePath:
    def __init__(
        self,
        mode: str = "time",          # {"time", "freq"}
        # --- time-mode params ---
        eta_max: float = 3.0,
        # --- freq-mode params ---
        nu: float = 1.0,
        tau_max: float | None = None,        # auto if None
        alpha_min: float = 1e-4,             # used to auto-set tau_max
        freq_axes: tuple[int, ...] = (-2, -1),
    ): ...

    # ----- scalar API; identical signatures to LinearPath -----
    def compute_alpha_t(self, t: Tensor) -> Tensor: ...
    def compute_sigma_t(self, t: Tensor) -> Tensor: ...
    def compute_d_alpha_t(self, t: Tensor) -> Tensor: ...
    def compute_d_sigma_t(self, t: Tensor) -> Tensor: ...

    # ----- the plan() entry used by Transport.training_losses -----
    def plan(self, t: Tensor, x_data: Tensor, x_noise: Tensor) -> tuple[Tensor, Tensor]:
        """Return (x_t, velocity_target)."""

    # ----- freq-mode-only helpers (raise NotImplementedError in time mode) -----
    def freq_kernel(self, x_shape: tuple, t: Tensor, device, dtype) -> Tensor:
        """Return H(k, t) broadcastable to rfftn(x_data) over freq_axes."""

    def q_sample(self, x_data: Tensor, t: Tensor, eps: Tensor) -> Tensor:
        """Generic forward marginal; dispatches on self.mode."""

    # ----- conversions consumed by Transport / Sampler -----
    def get_score_from_velocity(self, v, x_t, t) -> Tensor: ...
    def get_noise_from_velocity(self, v, x_t, t) -> Tensor: ...
    def get_velocity_from_score(self, score, x_t, t) -> Tensor: ...
```

Implementation notes:

* Use `torch.special.erfc`. Avoid `scipy`.
* Time arg `t` may have shape `(B,)`; broadcast to `x` via `t.view(-1, *([1] * (x.dim() - 1)))`. Provide a helper `_align(t, x)`.
* For `freq_axes`, compute the wavenumber grid once in `freq_kernel`; cache by `(x_shape, device, dtype)` on the instance.
* Use `torch.fft.rfftn(..., dim=freq_axes)` (real-input FFT) for memory; reconstruct via `torch.fft.irfftn(..., s=spatial_sizes, dim=freq_axes)`.
* The noise term in freq mode must be **circularly-symmetric complex Gaussian** with variance matching the iFFT normalization. The simplest correct approach: sample $\eta$ in **real space** as $\mathcal{N}(0, I)$, FFT it, multiply by $\sqrt{1-H^2}$, then iFFT. This is mathematically equivalent and avoids manual complex-noise bookkeeping.
* In the freq mode, `compute_alpha_t` and `compute_sigma_t` return scalar effective values (root-mean-square over modes) — used only for `train_eps`/`sample_eps` schedule clamping; the *actual* corruption is applied per-mode inside `q_sample` and `plan`.

### 3.2 Modify: `transport/transport.py` — registration

```python
def create_transport(
    path_type, prediction,
    loss_weight=None, train_eps=None, sample_eps=None,
    # --- Couette ---
    couette_mode="time",
    couette_eta_max=3.0,
    couette_nu=1.0,
    couette_tau_max=None,
    couette_alpha_min=1e-4,
    couette_freq_axes=(-2, -1),
):
    if path_type == "Linear":
        path = LinearPath()
    elif path_type == "GVP":
        path = GVPPath()
    elif path_type == "VP":
        path = VPPath()
    elif path_type == "Couette":
        from .couette import CouettePath
        path = CouettePath(
            mode=couette_mode,
            eta_max=couette_eta_max,
            nu=couette_nu,
            tau_max=couette_tau_max,
            alpha_min=couette_alpha_min,
            freq_axes=tuple(couette_freq_axes),
        )
    else:
        raise ValueError(f"Unknown path_type={path_type}")

    return Transport(
        path=path,
        prediction=prediction,
        loss_weight=loss_weight,
        train_eps=train_eps,
        sample_eps=sample_eps,
    )
```

If `Transport.training_losses` currently assumes scalar $\alpha,\sigma$ (it likely does for the existing three paths), refactor only the forward call so that:

```python
x_t, v_target = self.path.plan(t, x_data, x_noise)
```

is the single source of truth. All existing paths already (or trivially can) implement `plan()` — keep their behavior bit-for-bit identical and have the freq mode of `CouettePath` plug in here.

### 3.3 Modify: `train_utils.parse_transport_args`

Add CLI flags (do **not** remove existing ones):

```text
--path-type {Linear,GVP,VP,Couette}     # extend the choices list
--couette-mode {time,freq}              default: time
--couette-eta-max FLOAT                 default: 3.0
--couette-nu FLOAT                      default: 1.0
--couette-tau-max FLOAT                 default: None
--couette-alpha-min FLOAT               default: 1e-4
--couette-freq-axes "INT,INT,..."       default: "-2,-1"
```

Parse `--couette-freq-axes` with `lambda s: tuple(int(x) for x in s.split(","))`.

No changes to `sample.py` or `models.py`.

## 4. Test cases — `tests/test_couette.py`

```python
import torch
from transport.couette import CouettePath

def test_boundary_conditions_time():
    p = CouettePath(mode="time", eta_max=3.0)
    assert torch.allclose(p.compute_alpha_t(torch.tensor([0.0])), torch.tensor([1.0]))
    assert p.compute_alpha_t(torch.tensor([1.0])).item() < 1e-4
    assert torch.allclose(p.compute_sigma_t(torch.tensor([0.0])), torch.tensor([0.0]),
                          atol=1e-6)
    assert abs(p.compute_sigma_t(torch.tensor([1.0])).item() - 1.0) < 1e-4

def test_variance_preservation_time():
    p = CouettePath(mode="time", eta_max=3.0)
    t = torch.linspace(0.0, 1.0, 200)
    a, s = p.compute_alpha_t(t), p.compute_sigma_t(t)
    assert torch.allclose(a**2 + s**2, torch.ones_like(t), atol=1e-5)

def test_derivative_consistency_time():
    p = CouettePath(mode="time", eta_max=3.0)
    t = torch.linspace(0.05, 0.95, 50, requires_grad=True)
    a = p.compute_alpha_t(t); a.sum().backward()
    a_num = t.grad.clone(); t.grad = None
    a_ana = p.compute_d_alpha_t(t.detach())
    assert torch.allclose(a_num, a_ana, atol=1e-4)
    s = p.compute_sigma_t(t); s.sum().backward()
    s_num = t.grad.clone()
    s_ana = p.compute_d_sigma_t(t.detach())
    assert torch.allclose(s_num, s_ana, atol=1e-3)

def test_qsample_t0_is_identity_freq():
    p = CouettePath(mode="freq", nu=1.0, alpha_min=1e-4, freq_axes=(-2, -1))
    x = torch.randn(2, 3, 16, 16)
    x_t = p.q_sample(x, torch.zeros(2), eps=torch.randn_like(x))
    assert torch.allclose(x_t, x, atol=1e-5)

def test_qsample_t1_is_noise_freq():
    p = CouettePath(mode="freq", nu=1.0, alpha_min=1e-4, freq_axes=(-2, -1))
    x = torch.randn(2, 3, 16, 16)
    eps = torch.randn_like(x)
    x_t = p.q_sample(x, torch.ones(2), eps=eps)
    # signal should be dominated by eps; correlation with x_data should be ~0
    c = (x_t * x).flatten(1).sum(1) / (x_t.flatten(1).norm(dim=1) * x.flatten(1).norm(dim=1))
    assert c.abs().mean().item() < 0.05

def test_velocity_score_noise_roundtrip_time():
    p = CouettePath(mode="time", eta_max=3.0)
    t = torch.tensor([0.3, 0.7])
    x0 = torch.randn(2, 3, 16, 16)
    eps = torch.randn_like(x0)
    x_t, v = p.plan(t, x0, eps)
    score = p.get_score_from_velocity(v, x_t, t)
    v_back = p.get_velocity_from_score(score, x_t, t)
    assert torch.allclose(v, v_back, atol=1e-4)

def test_dropin_with_existing_transport():
    from transport import create_transport
    tr = create_transport(path_type="Couette", prediction="velocity")
    # mimic one training step
    x = torch.randn(4, 3, 32, 32)
    # ...assert tr.training_losses(...) returns finite scalar
```

## 5. Hyperparameter defaults and ablations

**Defaults that should reproduce LinearPath-like behavior on first run**:
* `couette_mode="time"`, `eta_max=3.0`, `prediction="velocity"`, `loss_weight=None`.

**Ablation grid for evaluation** (do not run these as part of this task — just make them trivially reachable from the CLI):

1. `eta_max ∈ {1.5, 2.0, 2.5, 3.0, 3.5}` — controls schedule steepness near $t=1$.
2. `couette_mode ∈ {"time", "freq"}` — scalar interpolant vs anisotropic spectral decay.
3. `nu ∈ {0.5, 1.0, 2.0, 4.0}` (freq mode only) — per-mode decay rate (equivalent to rescaling $\tau_{\max}$).
4. `prediction ∈ {"velocity", "score", "noise"}` — numerical stability comparison.
5. vs baselines `path_type ∈ {Linear, GVP, VP}` at matched compute.

## 6. Acceptance criteria

Implementation is accepted when:

* All tests in §4 pass.
* `python sample.py ODE --path-type Couette --couette-mode time --prediction velocity --num-sampling-steps 50 --ckpt <path>` produces non-NaN images.
* The same command with `--couette-mode freq` works and produces non-NaN images.
* Existing `--path-type Linear/GVP/VP` runs are bit-for-bit unchanged (regression-test by hashing one sampled batch with fixed seed before and after the patch).
* Training loop runs for ≥100 steps without NaN with `--path-type Couette --couette-mode time` and `--couette-mode freq`.

## 7. Out of scope

* Architectural changes to `SiT` or any block in `models.py`.
* Replacing `TimestepEmbedder` or modifying $t$-conditioning.
* Touching `Sampler` integrators (Euler / Heun / Dopri etc.) — they should work unchanged through the `Transport` interface.
* Classifier-free guidance handling (`forward_with_cfg` is unchanged).
* Any wall-clock or FID benchmarking — reserved for downstream paper experiments.

## Appendix A — Derivations

### A.1 $\alpha'(t)$

$$\alpha(t)=\mathrm{erfc}(\eta_{\max} t)=\frac{2}{\sqrt{\pi}}\int_{\eta_{\max} t}^{\infty} e^{-u^2}\,du.$$

$$\alpha'(t) = \frac{2}{\sqrt{\pi}}\cdot\big(-e^{-(\eta_{\max} t)^2}\big)\cdot \eta_{\max} = -\frac{2\eta_{\max}}{\sqrt{\pi}} e^{-\eta_{\max}^2 t^2}.$$

### A.2 $\sigma'(t)$

From $\sigma^2 = 1-\alpha^2$:
$$2\sigma\sigma' = -2\alpha\alpha' \implies \sigma'(t) = -\frac{\alpha(t)\,\alpha'(t)}{\sigma(t)}.$$

Since $\alpha\in(0,1]$ and $\alpha'<0$, $\sigma'>0$ on $(0,1)$. As $t\to 0^+$: $\alpha\to 1$, $\sigma\sim 2\sqrt{\eta_{\max} t/\sqrt\pi}$, so $\sigma'\sim (\eta_{\max}/\sqrt\pi)^{1/2}/\sqrt{t}\to\infty$. Train/sample $\epsilon$-clamping handles this.

### A.3 $\tau_{\max}$ from $\alpha_{\min}$

Want the worst-case mode satisfying $\exp(-\nu K^2_{\max}\tau_{\max})=\alpha_{\min}$. For `rfftn` over axes with spatial sizes $N_i$, the worst case is $k_i = N_i/2$ for all $i$, giving $K^2_{\max} = d/4$ where $d = |\texttt{freq\_axes}|$. Therefore:
$$\tau_{\max} = -\frac{4\log\alpha_{\min}}{\nu\,d}.$$

Implement this when `tau_max is None`.

---

End of prompt.
