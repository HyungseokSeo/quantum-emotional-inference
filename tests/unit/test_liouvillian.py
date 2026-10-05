"""Tests for the Liouvillian spectral analysis (dissertation §2.5)."""
import numpy as np
import pytest
import torch

from src.quantum.dynamics.lindblad import LindbladDynamics
from src.quantum.dynamics import liouvillian as lv

SZ = np.diag([1.0, -1.0]).astype(complex)
SM = np.array([[0, 1], [0, 0]], dtype=complex)


def _random_rho(n, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    rho = A @ A.conj().T
    return rho / np.trace(rho)


class TestConstruction:
    def test_matches_lindblad_drho_dt(self):
        H = torch.tensor([[0, 0.4, 0], [0.4, 1, 0.2], [0, 0.2, 2]], dtype=torch.complex64)
        dyn = LindbladDynamics.amplitude_damping(3, [0.7, 1.3], hamiltonian=H)
        rho = torch.tensor(_random_rho(3), dtype=torch.complex64)
        a = dyn.drho_dt(rho).numpy()
        b = lv.apply(lv.liouvillian_from_dynamics(dyn), rho)
        assert np.abs(a - b).max() < 1e-6

    def test_trace_preserving(self):
        Lhat = lv.liouvillian(SZ, [SM, SZ], [0.8, 0.2])
        assert np.linalg.norm(lv.vec(np.eye(2)).conj() @ Lhat) < 1e-12

    def test_negative_rate_rejected(self):
        with pytest.raises(ValueError):
            lv.liouvillian(SZ, [SM], [-1.0])

    def test_vec_roundtrip(self):
        rho = _random_rho(4)
        assert np.allclose(lv.unvec(lv.vec(rho)), rho)


class TestSpectrum:
    def test_two_level_closed_form(self):
        """Eq. 2.5.6: spectrum {0, -γ, -γ/2-2γφ ± iΔE}."""
        g, gp, dE = 1.0, 0.3, 2.0
        s = lv.spectrum(lv.liouvillian(dE / 2 * SZ, [SM, SZ], [g, gp]))
        expected = np.array([0, -g, -g / 2 - 2 * gp + 1j * dE, -g / 2 - 2 * gp - 1j * dE])
        assert np.allclose(np.sort_complex(s.eigenvalues), np.sort_complex(expected), atol=1e-9)

    def test_gap_carrier_crossover(self):
        """Below γφ = γ/4 the gap is a coherence; above it, a population."""
        g, dE = 1.0, 2.0
        lo = lv.spectrum(lv.liouvillian(dE / 2 * SZ, [SM, SZ], [g, 0.1]))
        hi = lv.spectrum(lv.liouvillian(dE / 2 * SZ, [SM, SZ], [g, 0.4]))
        assert lo.offdiag[lo.n_stationary] > 0.99 and np.isclose(lo.gap, g / 2 + 0.2)
        assert hi.offdiag[hi.n_stationary] < 0.01 and np.isclose(hi.gap, g)

    def test_projector_dephasing_gap_and_kernel(self):
        """Repo dephasing: coherences decay at γ, all diagonal states stationary."""
        n, g = 5, 0.7
        ops = [np.diag([1.0 if i == k else 0.0 for i in range(n)]).astype(complex) for k in range(n)]
        s = lv.spectrum(lv.liouvillian(np.zeros((n, n)), ops, [g] * n))
        assert s.n_stationary == n
        assert np.isclose(s.gap, g)
        assert len(s.bands) == 1

    def test_spectral_evolution_matches_expm(self):
        H = np.array([[0, 0.3, 0], [0.3, 0.5, 0.1], [0, 0.1, 1.0]], dtype=complex)
        L = np.zeros((3, 3), dtype=complex); L[0, 2] = 1.0
        Lhat = lv.liouvillian(H, [L], [0.6])
        s = lv.spectrum(Lhat)
        rho0 = _random_rho(3, 1)
        t = 1.7
        exact = lv.apply(lv.channel_from_generator(Lhat, t), rho0)
        assert np.allclose(s.evolve(rho0, t), exact, atol=1e-9)

    def test_bands_split_on_decades(self):
        """Three channel families at rates 10, 1, 0.1 give three bands."""
        n = 3
        H = np.zeros((n, n))
        deph = [np.diag([1.0 if i == k else 0.0 for i in range(n)]).astype(complex) for k in range(n)]
        L01 = np.zeros((n, n), dtype=complex); L01[0, 1] = 1
        L20 = np.zeros((n, n), dtype=complex); L20[2, 0] = 1
        s = lv.spectrum(lv.liouvillian(H, deph + [L01, L20], [10.0] * n + [1.0, 0.1]))
        assert len(s.bands) == 3
        ts = [b.timescale for b in s.bands]
        assert ts == sorted(ts, reverse=True)  # bands ordered slowest first


class TestChain:
    def test_population_rate_eq_2_5_9(self):
        """A –Ω– B –γ→ D: population of A leaves at γ/2 - sqrt(γ²/4 - 4Ω²) for γ>4Ω."""
        Om = 1.0
        H = np.zeros((3, 3), dtype=complex); H[0, 1] = H[1, 0] = Om
        L = np.zeros((3, 3), dtype=complex); L[2, 1] = 1.0
        rho0 = np.zeros((3, 3), dtype=complex); rho0[0, 0] = 1.0
        for g in (0.5, 2.0, 10.0, 50.0):
            s = lv.spectrum(lv.liouvillian(H, [L], [g]))
            c = s.coefficients(rho0)
            wts = np.array([abs(c[j] * s.right[j][0, 0]) for j in range(len(c))])
            active = [j for j in s.nonstationary() if wts[j] > 1e-8]
            rate = s.rates[active].min()
            pred = g / 2 if g <= 4 * Om else g / 2 - np.sqrt(g**2 / 4 - 4 * Om**2)
            assert np.isclose(rate, pred, rtol=1e-6), (g, rate, pred)

    def test_gap_is_non_monotone(self):
        Om = 1.0
        H = np.zeros((3, 3), dtype=complex); H[0, 1] = H[1, 0] = Om
        L = np.zeros((3, 3), dtype=complex); L[2, 1] = 1.0
        gaps = [lv.spectrum(lv.liouvillian(H, [L], [g])).gap for g in (0.5, 4.0, 100.0)]
        assert gaps[1] > gaps[0] and gaps[1] > gaps[2]


class TestDarkStates:
    def test_pump_makes_target_unique_coherent_fixed_point(self):
        target = np.array([1, 1, 0], dtype=complex) / np.sqrt(2)
        ops = lv.pump_operators(target)
        H = np.zeros((3, 3))
        assert all(lv.is_dark_state(target, H, [op]) for op in ops)
        s = lv.spectrum(lv.liouvillian(H, ops, [1.0] * 2))
        assert s.n_stationary == 1
        rhoD = s.steady_states[0]
        assert np.isclose(np.real(target.conj() @ rhoD @ target), 1.0)
        assert abs(rhoD[0, 1]) > 0.49  # coherent, not a mixture
        D = lv.dark_states(H, ops)
        assert D.shape[1] == 1

    def test_hamiltonian_leaking_out_of_kernel_removes_dark_state(self):
        """|D⟩ in ker L but H|D⟩ ∉ span{|D⟩}: no dark state (eq. 2.5.7 fails)."""
        L = np.zeros((2, 2), dtype=complex); L[0, 1] = 1.0   # ker L = span{|0⟩}
        H = np.array([[0, 1], [1, 0]], dtype=complex)        # H|0⟩ = |1⟩
        assert lv.dark_states(H, [L]).shape[1] == 0
        assert not lv.is_dark_state([1, 0], H, [L])
        assert lv.is_dark_state([1, 0], np.zeros((2, 2)), [L])


class TestChannelLog:
    def test_log_recovers_generator_and_passes_gksl(self):
        H = np.array([[0, 0.3], [0.3, 0.8]], dtype=complex)
        Lhat = lv.liouvillian(H, [SM, SZ], [0.5, 0.1])
        Phi = lv.channel_from_generator(Lhat, 0.7)
        Lrec, chk = lv.generator_from_channel(Phi, 0.7)
        assert np.allclose(Lrec, Lhat, atol=1e-8)
        assert chk.is_gksl

    def test_non_markovian_map_fails_ccp(self):
        """A unitary-like map with a non-CP log: swap-ish Φ whose log is not GKSL."""
        # Φ = transpose map on 2x2 (positive but not CP) composed with mild contraction
        n = 2
        T = np.zeros((n * n, n * n), dtype=complex)
        for i in range(n):
            for j in range(n):
                E = np.zeros((n, n)); E[i, j] = 1
                T[:, i + n * j] = lv.vec(E.T)
        Phi = 0.5 * T + 0.5 * np.eye(n * n)
        _, chk = lv.generator_from_channel(Phi, 1.0)
        assert not chk.is_gksl
