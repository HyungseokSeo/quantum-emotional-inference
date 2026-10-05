"""
Liouvillian (GKSL generator) spectral analysis.

Implements the spectral machinery of dissertation §2.5 ("Spectral Structure of
Affective Relaxation") on top of the existing ``LindbladDynamics`` objects.

Given a GKSL generator

    dρ/dt = -i[H, ρ] + Σₖ γₖ (Lₖ ρ Lₖ† - ½{Lₖ†Lₖ, ρ})

the generator is a linear map on operators.  Vectorising ρ by stacking its
columns, vec(A ρ B) = (Bᵀ ⊗ A) vec(ρ), gives the n² × n² matrix

    L̂ = -i(I ⊗ H - Hᵀ ⊗ I)
        + Σₖ γₖ ( L̄ₖ ⊗ Lₖ - ½ I ⊗ Lₖ†Lₖ - ½ (Lₖ†Lₖ)ᵀ ⊗ I ).

Its eigenvalues λⱼ (Re λⱼ ≤ 0) are the relaxation modes of the affective
state, the Liouvillian gap Δ = min_{j≥1} |Re λⱼ| is the slowest nonzero rate,
and the right eigenoperators Rⱼ are the density-matrix patterns that decay at
each rate.  Everything here is NumPy/SciPy; torch tensors are accepted and
converted.

References:
    Gorini, Kossakowski & Sudarshan (1976); Lindblad (1976).
    Wolf, Eisert, Cubitt & Cirac, PRL 101, 150402 (2008)  — GKSL test for log Φ.
    Macieszczak, Guţă, Lesanovsky & Garrahan, PRL 116, 240404 (2016) — metastability.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import scipy.linalg as sla

ArrayLike = Union[np.ndarray, "torch.Tensor"]  # noqa: F821


# --------------------------------------------------------------------------- #
# Conversions
# --------------------------------------------------------------------------- #

def _np(x: ArrayLike) -> np.ndarray:
    """Convert a NumPy array or torch tensor to a complex128 NumPy array."""
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.complex128)


def vec(rho: ArrayLike) -> np.ndarray:
    """Column-stacking vectorisation, vec(ρ)."""
    return _np(rho).reshape(-1, order="F")


def unvec(v: ArrayLike, n: Optional[int] = None) -> np.ndarray:
    """Inverse of :func:`vec`."""
    v = _np(v)
    if n is None:
        n = int(round(np.sqrt(v.size)))
    return v.reshape((n, n), order="F")


# --------------------------------------------------------------------------- #
# Building the generator
# --------------------------------------------------------------------------- #

def liouvillian(
    H: ArrayLike,
    jump_ops: Sequence[ArrayLike] = (),
    rates: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """
    Build the n² × n² Liouvillian matrix of a GKSL generator.

    Args:
        H: Hermitian n × n Hamiltonian.
        jump_ops: jump operators Lₖ (n × n each).
        rates: γₖ ≥ 0; defaults to 1 for every operator (rates absorbed in Lₖ).

    Returns:
        L̂ acting on column-stacked vec(ρ).
    """
    H = _np(H)
    n = H.shape[0]
    I = np.eye(n, dtype=np.complex128)
    if not np.allclose(H, H.conj().T, atol=1e-8):
        raise ValueError("Hamiltonian must be Hermitian")

    Lhat = -1j * (np.kron(I, H) - np.kron(H.T, I))

    if rates is None:
        rates = [1.0] * len(jump_ops)
    if len(rates) != len(jump_ops):
        raise ValueError("rates and jump_ops must have the same length")

    for L, g in zip(jump_ops, rates):
        if g < 0:
            raise ValueError("GKSL rates must be non-negative")
        L = _np(L)
        LdL = L.conj().T @ L
        Lhat += g * (np.kron(L.conj(), L) - 0.5 * np.kron(I, LdL) - 0.5 * np.kron(LdL.T, I))
    return Lhat


def liouvillian_from_dynamics(dynamics) -> np.ndarray:
    """Build the Liouvillian of an existing ``LindbladDynamics`` object."""
    return liouvillian(
        dynamics.H,
        [op.operator for op in dynamics.lindblad_ops],
        [op.rate for op in dynamics.lindblad_ops],
    )


def apply(Lhat: np.ndarray, rho: ArrayLike) -> np.ndarray:
    """Apply a vectorised superoperator to a density matrix."""
    return unvec(Lhat @ vec(rho))


# --------------------------------------------------------------------------- #
# Spectrum
# --------------------------------------------------------------------------- #

def offdiag_weight(R: np.ndarray) -> float:
    """Fraction of the Hilbert–Schmidt norm of R carried by off-diagonal entries."""
    R = _np(R)
    total = np.sum(np.abs(R) ** 2)
    if total == 0:
        return 0.0
    diag = np.sum(np.abs(np.diag(R)) ** 2)
    return float(1.0 - diag / total)


@dataclass
class Band:
    """A cluster of relaxation rates (one 'timescale layer')."""
    indices: List[int]
    rate_min: float
    rate_max: float

    @property
    def rate_geomean(self) -> float:
        return float(np.sqrt(self.rate_min * self.rate_max))

    @property
    def timescale(self) -> float:
        return 1.0 / self.rate_geomean


@dataclass
class Spectrum:
    """Result of :func:`spectrum`. Modes are sorted by |Re λ| ascending."""
    eigenvalues: np.ndarray                 # λⱼ, sorted
    right: List[np.ndarray]                 # Rⱼ (n × n), unit Hilbert–Schmidt norm
    left: List[np.ndarray]                  # Lⱼ, biorthogonal: Tr(Lⱼ† Rₖ) = δⱼₖ
    steady_states: List[np.ndarray]         # Hermitian, trace-1 basis of the kernel
    offdiag: np.ndarray                     # off-diagonal weight of each Rⱼ
    gap: float                              # min |Re λⱼ| over non-stationary modes
    n_stationary: int                       # dimension of the kernel
    bands: List[Band] = field(default_factory=list)

    @property
    def rates(self) -> np.ndarray:
        return -self.eigenvalues.real

    @property
    def frequencies(self) -> np.ndarray:
        return self.eigenvalues.imag

    def nonstationary(self) -> np.ndarray:
        return np.arange(self.n_stationary, len(self.eigenvalues))

    def gap_carrier(self) -> Tuple[int, np.ndarray, float]:
        """Index, eigenoperator and off-diagonal weight of the slowest mode."""
        j = int(self.n_stationary)
        return j, self.right[j], float(self.offdiag[j])

    def coefficients(self, rho0: ArrayLike) -> np.ndarray:
        """Mode amplitudes cⱼ = Tr(Lⱼ† ρ₀) for the expansion ρ(t) = Σ cⱼ e^{λⱼt} Rⱼ."""
        rho0 = _np(rho0)
        return np.array([np.trace(Lj.conj().T @ rho0) for Lj in self.left])

    def evolve(self, rho0: ArrayLike, t: float) -> np.ndarray:
        """ρ(t) from the spectral decomposition."""
        c = self.coefficients(rho0)
        out = np.zeros_like(self.right[0])
        for cj, lj, Rj in zip(c, self.eigenvalues, self.right):
            out += cj * np.exp(lj * t) * Rj
        return out


def _hermitize(R: np.ndarray) -> np.ndarray:
    return 0.5 * (R + R.conj().T)


def spectrum(
    Lhat: np.ndarray,
    stationary_tol: float = 1e-9,
    min_gap_decades: float = 0.5,
) -> Spectrum:
    """
    Eigen-decompose a Liouvillian and organise the result.

    Args:
        Lhat: n² × n² generator from :func:`liouvillian`.
        stationary_tol: |Re λ| below this counts as stationary (λ = 0).
        min_gap_decades: separation in log10(rate) that starts a new band.
    """
    n = int(round(np.sqrt(Lhat.shape[0])))
    evals, V = np.linalg.eig(Lhat)
    order = np.argsort(np.abs(evals.real) + 1e-12 * np.abs(evals.imag))
    evals, V = evals[order], V[:, order]

    # Left eigenvectors: W = (V⁻¹)†, so that Wⱼ† Vₖ = δⱼₖ.
    W = np.linalg.inv(V).conj().T

    right, left = [], []
    for j in range(V.shape[1]):
        Rj = unvec(V[:, j], n)
        norm = np.linalg.norm(Rj)
        right.append(Rj / norm)
        left.append(unvec(W[:, j], n) * norm)

    n_stat = int(np.sum(np.abs(evals.real) < stationary_tol))
    n_stat = max(n_stat, 1)

    # Stationary states: Hermitian, trace-one representatives of the kernel.
    steady = []
    for j in range(n_stat):
        Rj = _hermitize(right[j])
        tr = np.trace(Rj).real
        if abs(tr) > 1e-10:
            steady.append(Rj / tr)
        else:
            steady.append(Rj)  # traceless stationary coherence (dark subspace)

    offd = np.array([offdiag_weight(R) for R in right])

    if n_stat < len(evals):
        gap = float(np.min(np.abs(evals.real[n_stat:])))
    else:
        gap = 0.0

    spec = Spectrum(evals, right, left, steady, offd, gap, n_stat)
    spec.bands = bands(spec, min_gap_decades)
    return spec


def bands(spec: Spectrum, min_gap_decades: float = 0.5) -> List[Band]:
    """
    Cluster the non-stationary rates into bands by gaps in log10(rate).

    Consecutive sorted rates that differ by more than ``min_gap_decades``
    decades start a new band.  Degenerate rates are grouped together.
    """
    idx = spec.nonstationary()
    if idx.size == 0:
        return []
    r = spec.rates[idx]
    logs = np.log10(np.maximum(r, 1e-300))
    out, cur = [], [int(idx[0])]
    for k in range(1, len(idx)):
        if logs[k] - logs[k - 1] > min_gap_decades:
            out.append(cur)
            cur = []
        cur.append(int(idx[k]))
    out.append(cur)
    return [Band(c, float(spec.rates[c].min()), float(spec.rates[c].max())) for c in out]


# --------------------------------------------------------------------------- #
# Dark states and stationary structure
# --------------------------------------------------------------------------- #

def _null_space(A: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    """Orthonormal basis of the null space of A (columns)."""
    if A.shape[0] == 0:
        return np.eye(A.shape[1], dtype=np.complex128)
    _, s, Vh = np.linalg.svd(A)
    rank = int(np.sum(s > tol * max(1.0, s[0] if s.size else 1.0)))
    return Vh[rank:].conj().T


def dark_states(
    H: ArrayLike,
    jump_ops: Sequence[ArrayLike],
    tol: float = 1e-9,
) -> np.ndarray:
    """
    Orthonormal basis (columns) of the dark subspace: vectors |D⟩ with
    Lₖ|D⟩ = 0 for all k that also span an H-invariant subspace
    (dissertation eq. 2.5.7).  Returns an n × 0 array if there is none.
    """
    H = _np(H)
    n = H.shape[0]
    Ls = [_np(L) for L in jump_ops]
    stacked = np.vstack(Ls) if Ls else np.zeros((0, n), dtype=np.complex128)
    N = _null_space(stacked, tol)
    # Shrink to the largest H-invariant subspace of ker(L): iterate
    # K ← {v ∈ K : H v ∈ K} until stable.
    while N.shape[1] > 0:
        P = N @ N.conj().T
        resid = (np.eye(n) - P) @ H @ N          # components of H N leaving K
        if np.linalg.norm(resid) < tol:
            break
        # restrict: new K = K ∩ ker((I-P) H)
        M = _null_space(np.vstack([stacked, (np.eye(n) - P) @ H]), tol)
        if M.shape[1] == N.shape[1]:
            break
        N = M
    if N.shape[1] == 0:
        return N
    # Diagonalise H within the dark subspace so each column is an H eigenvector.
    h = N.conj().T @ H @ N
    _, U = np.linalg.eigh(_hermitize(h))
    return N @ U


def is_dark_state(psi: ArrayLike, H: ArrayLike, jump_ops: Sequence[ArrayLike], tol=1e-8) -> bool:
    """Check eq. 2.5.7 for a single vector."""
    psi = _np(psi).reshape(-1)
    psi = psi / np.linalg.norm(psi)
    H = _np(H)
    for L in jump_ops:
        if np.linalg.norm(_np(L) @ psi) > tol:
            return False
    Hpsi = H @ psi
    E = np.vdot(psi, Hpsi)
    return bool(np.linalg.norm(Hpsi - E * psi) < tol)


def pump_operators(target: ArrayLike, n: Optional[int] = None) -> List[np.ndarray]:
    """
    Jump operators Lₖ = |D⟩⟨Bₖ| that pump every bright state Bₖ ⟂ D into the
    target |D⟩ (dissertation eq. 2.5.8).  |D⟩ is then a dark state of the set.
    """
    d = _np(target).reshape(-1)
    d = d / np.linalg.norm(d)
    n = d.size if n is None else n
    # Orthonormal complement of d via QR on [d | I].
    Q, _ = np.linalg.qr(np.column_stack([d, np.eye(n)]))
    bright = Q[:, 1:n]
    return [np.outer(d, b.conj()) for b in bright.T]


# --------------------------------------------------------------------------- #
# Generator from a fitted CPTP map (dissertation §2.5.7)
# --------------------------------------------------------------------------- #

def choi_from_superoperator(S: np.ndarray) -> np.ndarray:
    """Choi matrix C = Σᵢⱼ S(|i⟩⟨j|) ⊗ |i⟩⟨j| of a vectorised superoperator."""
    n = int(round(np.sqrt(S.shape[0])))
    C = np.zeros((n * n, n * n), dtype=np.complex128)
    for i in range(n):
        for j in range(n):
            E = np.zeros((n, n), dtype=np.complex128)
            E[i, j] = 1.0
            C += np.kron(apply(S, E), E)
    return C


@dataclass
class GKSLCheck:
    hermiticity_preserving: float   # ‖S(X†) - S(X)†‖ summed over a basis
    trace_annihilating: float       # ‖vec(I)† S‖
    conditional_cp_min_eig: float   # min eig of ω⊥ C_S ω⊥ (≥ 0 for GKSL)
    negative_channel_eigs: int      # # of Φ eigenvalues on the negative real axis

    @property
    def is_gksl(self) -> bool:
        return (self.hermiticity_preserving < 1e-7
                and self.trace_annihilating < 1e-7
                and self.conditional_cp_min_eig > -1e-7)


def gksl_check(S: np.ndarray, channel_eigs: Optional[np.ndarray] = None) -> GKSLCheck:
    """
    Test whether a superoperator S is a valid GKSL generator
    (Wolf, Eisert, Cubitt & Cirac 2008, Thm. 1): Hermiticity preserving,
    trace annihilating, and conditionally completely positive.
    """
    n = int(round(np.sqrt(S.shape[0])))
    herm = 0.0
    for i in range(n):
        for j in range(n):
            E = np.zeros((n, n), dtype=np.complex128)
            E[i, j] = 1.0
            herm += np.linalg.norm(apply(S, E.conj().T) - apply(S, E).conj().T)
    tr = float(np.linalg.norm(vec(np.eye(n)).conj() @ S))
    C = choi_from_superoperator(S)
    omega = vec(np.eye(n)) / np.sqrt(n)           # |Ω⟩ = Σᵢ |i,i⟩/√n
    Pperp = np.eye(n * n) - np.outer(omega, omega.conj())
    M = Pperp @ C @ Pperp
    ccp = float(np.min(np.linalg.eigvalsh(_hermitize(M))))
    neg = 0
    if channel_eigs is not None:
        neg = int(np.sum((channel_eigs.real < 0) & (np.abs(channel_eigs.imag) < 1e-9)))
    return GKSLCheck(float(herm), tr, ccp, neg)


def generator_from_channel(
    Phi: np.ndarray,
    tau: float,
) -> Tuple[np.ndarray, GKSLCheck]:
    """
    Recover a generator from a fitted one-step map Φ_τ (vectorised,
    column-stacking) via the principal logarithm, L̂ = log(Φ_τ)/τ
    (dissertation eq. 2.5.11), and report the GKSL diagnostics.

    The branch is ambiguous when Φ_τ has eigenvalues on the negative real
    axis; ``GKSLCheck.negative_channel_eigs`` counts them.
    """
    Phi = _np(Phi)
    ev = np.linalg.eigvals(Phi)
    Lhat = sla.logm(Phi) / tau
    Lhat = np.asarray(Lhat, dtype=np.complex128)
    return Lhat, gksl_check(Lhat, ev)


def channel_from_generator(Lhat: np.ndarray, tau: float) -> np.ndarray:
    """Φ_τ = exp(τ L̂)."""
    return sla.expm(tau * Lhat)


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #

def describe(spec: Spectrum, labels: Optional[Sequence[str]] = None, top: int = 6) -> str:
    """Human-readable summary of a spectrum."""
    lines = []
    lines.append(f"modes: {len(spec.eigenvalues)}   stationary: {spec.n_stationary}   "
                 f"gap Δ = {spec.gap:.4g}   1/Δ = {1/spec.gap if spec.gap else float('inf'):.4g}")
    j, R, w = spec.gap_carrier()
    lines.append(f"gap carrier: mode {j}, off-diagonal weight {w:.2f} "
                 f"({'coherence' if w > 0.5 else 'population'})")
    if labels is not None:
        # dominant entry of the gap carrier
        a, b = np.unravel_index(np.argmax(np.abs(R)), R.shape)
        lines.append(f"  dominant entry: ({labels[a]}, {labels[b]})")
    for k, b in enumerate(spec.bands, 1):
        offw = spec.offdiag[b.indices].mean()
        lines.append(f"band {k}: {len(b.indices):3d} modes  rate ∈ [{b.rate_min:.3g}, {b.rate_max:.3g}]  "
                     f"τ ≈ {b.timescale:.3g}  mean off-diag weight {offw:.2f}")
    lines.append("slowest modes:")
    for j in range(spec.n_stationary, min(len(spec.eigenvalues), spec.n_stationary + top)):
        lam = spec.eigenvalues[j]
        lines.append(f"  λ = {lam.real:+.4g} {lam.imag:+.4g}i   off-diag {spec.offdiag[j]:.2f}")
    return "\n".join(lines)


__all__ = [
    "vec", "unvec", "liouvillian", "liouvillian_from_dynamics", "apply",
    "offdiag_weight", "Band", "Spectrum", "spectrum", "bands",
    "dark_states", "is_dark_state", "pump_operators",
    "choi_from_superoperator", "GKSLCheck", "gksl_check",
    "generator_from_channel", "channel_from_generator", "describe",
]
