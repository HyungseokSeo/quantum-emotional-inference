"""
Experiment 04: Liouvillian spectrum of the affective generator.

Computational protocol of dissertation §2.5.7.  Three modes:

  levels     spectrum of each repo level generator (reactive / adaptive /
             reflective, rates from experiments/configs/base.yaml) and of a
             composite generator that carries all three channel families.
             Tests Prediction S1 (three bands) and S2 (what carries the gap).

  demo       numerical check of the closed-form results quoted in §2.5:
             eq. 2.5.6 (two-level gap crossover) and eq. 2.5.9 (non-monotone
             gap of the A -H- B -L-> D chain, Prediction S3), plus a dark-state
             pump built with eq. 2.5.8.

  channel    load a fitted one-step CPTP map Φ_τ (e.g. the linear part of the
             Level-2 dynamical map), take L̂ = log(Φ_τ)/τ, run the GKSL test of
             Wolf et al. (2008), and analyse the spectrum.

Usage:
    python experiments/04_hierarchy_validation/liouvillian_spectrum.py levels
    python experiments/04_hierarchy_validation/liouvillian_spectrum.py demo
    python experiments/04_hierarchy_validation/liouvillian_spectrum.py channel \
        --phi path/to/phi.npy --tau 1.0 [--labels happy,sad,...]

    A generator saved as .npz with arrays H, L (k×n×n), gamma (k,) can be
    analysed with:  ... generator --npz path/to/gen.npz

Figures and a JSON summary go to experiments/04_hierarchy_validation/outputs/.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import yaml

warnings.filterwarnings("ignore", category=UserWarning)
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.quantum.dynamics import liouvillian as lv  # noqa: E402
from src.quantum.states.emotional_basis import EKMAN_EMOTIONS  # noqa: E402

OUT = Path(__file__).resolve().parent / "outputs"
CONFIG = ROOT / "experiments" / "configs" / "base.yaml"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def load_config() -> dict:
    with open(CONFIG) as f:
        return yaml.safe_load(f)


def projector_dephasing(n: int):
    """The repo's dephasing channel: one projector |i><i| per emotion."""
    ops = []
    for i in range(n):
        P = np.zeros((n, n), dtype=complex)
        P[i, i] = 1.0
        ops.append(P)
    return ops


def lowering_to(n: int, target: int):
    """|target><i| for every i != target (relaxation into one category)."""
    ops = []
    for i in range(n):
        if i == target:
            continue
        L = np.zeros((n, n), dtype=complex)
        L[target, i] = 1.0
        ops.append(L)
    return ops


def appraisal_hamiltonian(n: int, coupling: float, seed: int = 0) -> np.ndarray:
    """Random Hermitian H with off-diagonal couplings of scale ``coupling``."""
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    H = coupling * (A + A.conj().T) / 2
    np.fill_diagonal(H, 0.0)
    return H


def plot_spectrum(spec: lv.Spectrum, title: str, path: Path) -> None:
    """Rates on a log axis, coloured by off-diagonal weight, bands shaded."""
    idx = spec.nonstationary()
    rates = spec.rates[idx]
    offd = spec.offdiag[idx]
    freqs = np.abs(spec.frequencies[idx])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [2, 1]})
    ax = axes[0]
    for k, b in enumerate(spec.bands):
        lo, hi = np.log10(b.rate_min) - 0.1, np.log10(b.rate_max) + 0.1
        ax.axvspan(10 ** lo, 10 ** hi, color="0.92" if k % 2 else "0.85", zorder=0)
        ax.text(b.rate_geomean, 1.02, f"band {k+1}\nτ≈{b.timescale:.2g}s",
                ha="center", va="bottom", fontsize=8, transform=ax.get_xaxis_transform())
    sc = ax.scatter(rates, np.arange(len(rates)), c=offd, cmap="viridis", vmin=0, vmax=1,
                    s=28, edgecolor="k", linewidth=0.3, zorder=3)
    ax.set_xscale("log")
    ax.set_xlabel(r"relaxation rate $|\mathrm{Re}\,\lambda_j|$  [s$^{-1}$]")
    ax.set_ylabel("mode index (sorted)")
    ax.axvline(spec.gap, color="C3", ls="--", lw=1, label=rf"gap $\Delta$={spec.gap:.3g}")
    ax.legend(loc="lower right", fontsize=8)
    cb = fig.colorbar(sc, ax=ax, pad=0.01)
    cb.set_label("off-diagonal weight of $R_j$")

    ax = axes[1]
    ax.scatter(spec.eigenvalues.real, spec.eigenvalues.imag, c=spec.offdiag, cmap="viridis",
               vmin=0, vmax=1, s=22, edgecolor="k", linewidth=0.3)
    ax.axvline(0, color="0.5", lw=0.8)
    ax.set_xlabel(r"$\mathrm{Re}\,\lambda$")
    ax.set_ylabel(r"$\mathrm{Im}\,\lambda$")
    ax.set_title("complex spectrum", fontsize=10)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def summary_dict(spec: lv.Spectrum) -> dict:
    return {
        "n_modes": int(len(spec.eigenvalues)),
        "n_stationary": int(spec.n_stationary),
        "gap": float(spec.gap),
        "gap_time": float(1 / spec.gap) if spec.gap > 0 else None,
        "gap_carrier_offdiag_weight": float(spec.offdiag[spec.n_stationary]),
        "bands": [
            {"n_modes": len(b.indices), "rate_min": b.rate_min, "rate_max": b.rate_max,
             "timescale": b.timescale,
             "mean_offdiag_weight": float(spec.offdiag[b.indices].mean())}
            for b in spec.bands
        ],
    }


# --------------------------------------------------------------------------- #
# Mode: levels
# --------------------------------------------------------------------------- #

def run_levels(args) -> dict:
    cfg = load_config()
    n = cfg["quantum"]["n_emotions"]
    labels = EKMAN_EMOTIONS[:n]
    hier = cfg["hierarchy"]
    gammas = {k: hier[f"gamma_{k}"] for k in ("reactive", "adaptive", "reflective")}
    taus = {k: hier[f"{k}_timescale"] for k in ("reactive", "adaptive", "reflective")}
    H = appraisal_hamiltonian(n, args.coupling, cfg["experiment"]["seed"])
    results = {}

    print("=" * 72)
    print("Per-level generators (repo defaults: projector dephasing, rate γ_ℓ)")
    print("=" * 72)
    deph = projector_dephasing(n)
    for name in ("reactive", "adaptive", "reflective"):
        g = gammas[name]
        Lhat = lv.liouvillian(H, deph, [g] * n)
        spec = lv.spectrum(Lhat)
        print(f"\n--- {name}: γ = {g}, nominal τ = {taus[name]} s ---")
        print(lv.describe(spec, labels, top=3))
        print(f"  check: γ·τ (decoherence per pass) = {g * taus[name]:.3g}; "
              f"kernel dimension {spec.n_stationary} "
              f"({'no unique steady state' if spec.n_stationary > 1 else 'unique steady state'})")
        plot_spectrum(spec, f"{name} level generator (γ={g}, coupling={args.coupling})",
                      OUT / f"spectrum_level_{name}.png")
        results[name] = summary_dict(spec)

    print("\n" + "=" * 72)
    print("Composite generator: all three channel families on one state")
    print("=" * 72)
    # Channel families, one per layer, each at its configured rate:
    #   reactive   : dephasing in the category basis         (kills coherence)
    #   adaptive   : relaxation of arousal-type categories    (population flow)
    #   reflective : relaxation of everything into 'neutral'  (settling)
    ops, rates, names = [], [], []
    for P in deph:
        ops.append(P); rates.append(gammas["reactive"]); names.append("dephase")
    # adaptive: pairwise flow between the first two contested categories
    L01 = np.zeros((n, n), dtype=complex); L01[0, 1] = 1.0
    L10 = L01.T.copy()
    ops += [L01, L10]; rates += [gammas["adaptive"]] * 2; names += ["flow", "flow"]
    neutral = labels.index("neutral") if "neutral" in labels else n - 1
    for L in lowering_to(n, neutral):
        ops.append(L); rates.append(gammas["reflective"]); names.append("settle")
    # NB: the repo's γ ordering is reactive < adaptive < reflective ("integrated
    # effect"), so here the *slowest* band is the dephasing one.  The band
    # structure is what matters for S1; which family is slowest is reported.
    Lhat = lv.liouvillian(H, ops, rates)
    spec = lv.spectrum(Lhat, min_gap_decades=args.min_gap)
    print(lv.describe(spec, labels, top=6))
    print(f"\nPrediction S1: {len(spec.bands)} band(s) found "
          f"(threshold {args.min_gap} decades). "
          + ("three-band structure PRESENT" if len(spec.bands) == 3 else "three-band structure ABSENT"))
    j, R, w = spec.gap_carrier()
    print(f"Prediction S2: gap carried by {'coherence' if w > 0.5 else 'population'} "
          f"(off-diagonal weight {w:.2f})")
    plot_spectrum(spec, f"composite affective generator (coupling={args.coupling})",
                  OUT / "spectrum_composite.png")
    results["composite"] = summary_dict(spec)
    results["composite"]["channels"] = [{"type": t, "rate": r} for t, r in zip(names, rates)]

    # Dark states of the composite set
    D = lv.dark_states(H, ops)
    print(f"dark subspace dimension: {D.shape[1]}")
    results["composite"]["dark_dim"] = int(D.shape[1])
    return results


# --------------------------------------------------------------------------- #
# Mode: demo (closed forms of §2.5.3 and §2.5.5)
# --------------------------------------------------------------------------- #

def run_demo(args) -> dict:
    out = {}
    sz = np.diag([1.0, -1.0]).astype(complex)
    sm = np.array([[0, 1], [0, 0]], dtype=complex)

    # ---- eq. 2.5.6: gap crossover at γ_φ = γ/4 -----------------------------
    print("=" * 72)
    print("§2.5.3  two-level gap crossover (eq. 2.5.6)")
    print("=" * 72)
    g, dE = 1.0, 2.0
    H = dE / 2 * sz
    gphis = np.linspace(0, 0.6, 61)
    gaps, carriers = [], []
    for gp in gphis:
        s = lv.spectrum(lv.liouvillian(H, [sm, sz], [g, gp]))
        gaps.append(s.gap); carriers.append(s.offdiag[s.n_stationary])
    gaps, carriers = np.array(gaps), np.array(carriers)
    pred = np.minimum(g, g / 2 + 2 * gphis)
    err = np.abs(gaps - pred).max()
    cross = gphis[np.argmax(carriers < 0.5)]
    print(f"max |numerical gap − min(γ, γ/2+2γφ)| = {err:.2e}")
    print(f"gap carrier switches coherence→population at γφ ≈ {cross:.3f}  (predicted γ/4 = {g/4})")
    out["two_level"] = {"max_err": float(err), "crossover": float(cross), "predicted": g / 4}

    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.plot(gphis, gaps, "k", label=r"numerical gap $\Delta$")
    ax.plot(gphis, pred, "C3--", label=r"$\min(\gamma,\ \gamma/2+2\gamma_\phi)$")
    ax.axvline(g / 4, color="0.6", ls=":", label=r"$\gamma_\phi=\gamma/4$")
    sc = ax.scatter(gphis, gaps, c=carriers, cmap="viridis", vmin=0, vmax=1, s=12, zorder=3)
    fig.colorbar(sc, ax=ax, label="off-diag weight of gap carrier")
    ax.set_xlabel(r"dephasing rate $\gamma_\phi$"); ax.set_ylabel("gap")
    ax.set_title("eq. 2.5.6: which mode carries the gap"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(OUT / "demo_two_level_gap.png", dpi=150); plt.close(fig)

    # ---- eq. 2.5.9: chain A -H- B -L-> D ------------------------------------
    print("\n" + "=" * 72)
    print("§2.5.5  chain A –Ω– B –γ→ D : gap vs γ (eq. 2.5.9, Prediction S3)")
    print("=" * 72)
    Om = 1.0
    A, B, D = 0, 1, 2
    H = np.zeros((3, 3), dtype=complex); H[A, B] = H[B, A] = Om
    L = np.zeros((3, 3), dtype=complex); L[D, B] = 1.0
    gs = np.logspace(-1, 2, 121)
    gap_num, pop_rate, carrier = [], [], []
    rho0 = np.zeros((3, 3), dtype=complex); rho0[A, A] = 1.0
    for g in gs:
        s = lv.spectrum(lv.liouvillian(H, [L], [g]))
        gap_num.append(s.gap); carrier.append(s.offdiag[s.n_stationary])
        # population-of-A decay rate: slowest mode with weight in ρ_AA
        c = s.coefficients(rho0)
        wts = np.array([abs(c[j] * s.right[j][A, A]) for j in range(len(c))])
        active = [j for j in s.nonstationary() if wts[j] > 1e-8]
        pop_rate.append(s.rates[active].min() if active else np.nan)
    gap_num, pop_rate, carrier = map(np.array, (gap_num, pop_rate, carrier))
    pred_pop = np.where(gs <= 4 * Om, gs / 2, gs / 2 - np.sqrt(np.maximum(gs**2 / 4 - 4 * Om**2, 0)))
    err_pop = np.nanmax(np.abs(pop_rate - pred_pop))
    k = int(np.argmax(gap_num))
    print(f"max |population rate − eq. 2.5.9| = {err_pop:.2e}")
    print(f"Liouvillian gap peaks at γ ≈ {gs[k]:.3g} (critical damping predicted at 4Ω = {4*Om})")
    print(f"gap / population-rate ratio at large γ: {gap_num[-1]/pop_rate[-1]:.3f}  "
          f"(gap carried by {'A–D coherence' if carrier[-1] > 0.5 else 'population'})")
    print(f"non-monotone: gap(γ=0.1)={gap_num[0]:.3g}, max={gap_num[k]:.3g}, gap(γ=100)={gap_num[-1]:.3g}")
    out["chain"] = {"max_err_population_rate": float(err_pop), "gamma_at_max_gap": float(gs[k]),
                    "gap_over_pop_rate_large_gamma": float(gap_num[-1] / pop_rate[-1])}

    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.loglog(gs, pop_rate, "k", label=r"population rate $\Gamma_{\rm eff}$ (num.)")
    ax.loglog(gs, pred_pop, "C3--", label="eq. 2.5.9")
    ax.loglog(gs, gap_num, "C0", label=r"Liouvillian gap $\Delta$")
    ax.axvline(4 * Om, color="0.6", ls=":", label=r"$\gamma=4\Omega$")
    ax.set_xlabel(r"channel strength $\gamma$"); ax.set_ylabel("rate")
    ax.set_title("eq. 2.5.9: regulation is non-monotone"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(OUT / "demo_chain_gap.png", dpi=150); plt.close(fig)

    # ---- eq. 2.5.8: dark-state pump ------------------------------------------
    print("\n" + "=" * 72)
    print("§2.5.5  dark-state pump (eq. 2.5.8): coherent target as unique fixed point")
    print("=" * 72)
    n = 3
    target = np.array([1, 1, 0], dtype=complex) / np.sqrt(2)      # ambivalent 'equanimity'
    ops = lv.pump_operators(target)
    H = np.outer(target, target.conj()) * 0.0                       # H commutes with ρ_D
    spec = lv.spectrum(lv.liouvillian(H, ops, [1.0] * len(ops)))
    rhoD = spec.steady_states[0]
    fid = np.real(target.conj() @ rhoD @ target)
    print(f"stationary states: {spec.n_stationary}; fidelity of ρ∞ with |D⟩: {fid:.6f}")
    print(f"ρ∞ off-diagonal (0,1): {rhoD[0,1].real:+.3f}  (coherent fixed point, not a mixture)")
    print(f"dark subspace found: dim {lv.dark_states(H, ops).shape[1]}; gap = {spec.gap:.3g}")
    out["pump"] = {"fidelity": float(fid), "gap": float(spec.gap), "n_stationary": int(spec.n_stationary)}
    return out


# --------------------------------------------------------------------------- #
# Mode: channel / generator
# --------------------------------------------------------------------------- #

def run_channel(args) -> dict:
    Phi = np.load(args.phi)
    if Phi.ndim != 2 or Phi.shape[0] != Phi.shape[1]:
        raise SystemExit("Φ must be a square n²×n² superoperator (column-stacking convention)")
    Lhat, chk = lv.generator_from_channel(Phi, args.tau)
    print(f"GKSL check: hermiticity-preserving residual {chk.hermiticity_preserving:.2e}, "
          f"trace-annihilating residual {chk.trace_annihilating:.2e}, "
          f"conditional-CP min eig {chk.conditional_cp_min_eig:+.2e}, "
          f"negative Φ eigenvalues {chk.negative_channel_eigs}")
    print(f"→ {'valid GKSL generator' if chk.is_gksl else 'NOT a GKSL generator (see §2.5.8 caveats)'}")
    return analyse_generator(Lhat, args, chk)


def run_generator(args) -> dict:
    z = np.load(args.npz)
    Lhat = lv.liouvillian(z["H"], list(z["L"]), list(z["gamma"]))
    return analyse_generator(Lhat, args, None)


def analyse_generator(Lhat, args, chk) -> dict:
    labels = args.labels.split(",") if args.labels else None
    spec = lv.spectrum(Lhat, min_gap_decades=args.min_gap)
    print(lv.describe(spec, labels))
    plot_spectrum(spec, Path(args.phi or args.npz).name, OUT / "spectrum_fitted.png")
    res = summary_dict(spec)
    if chk is not None:
        res["gksl_check"] = chk.__dict__ | {"is_gksl": chk.is_gksl}
    return res


# --------------------------------------------------------------------------- #

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=["levels", "demo", "channel", "generator"])
    p.add_argument("--coupling", type=float, default=0.5,
                   help="scale of off-diagonal appraisal couplings in H (levels mode)")
    p.add_argument("--min-gap", type=float, default=0.5,
                   help="decades of separation that start a new band")
    p.add_argument("--phi", type=str, default=None, help=".npy n²×n² CPTP map (channel mode)")
    p.add_argument("--tau", type=float, default=1.0, help="time step of Φ_τ in seconds")
    p.add_argument("--npz", type=str, default=None, help=".npz with H, L, gamma (generator mode)")
    p.add_argument("--labels", type=str, default=None, help="comma-separated category labels")
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    runner = {"levels": run_levels, "demo": run_demo, "channel": run_channel,
              "generator": run_generator}[args.mode]
    if args.mode == "channel" and not args.phi:
        p.error("channel mode needs --phi")
    if args.mode == "generator" and not args.npz:
        p.error("generator mode needs --npz")
    res = runner(args)
    with open(OUT / f"summary_{args.mode}.json", "w") as f:
        json.dump(res, f, indent=2, default=float)
    print(f"\nfigures and summary written to {OUT.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
