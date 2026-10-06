# Chapter Outline — Hierarchical Decoherence: A Computational Realization in a Multi-Timescale Perception Architecture

*Integrative chapter draft outline for* **Quantum Emotional Inference and the Collapse of Affect**. *StackEdit-ready: Markdown with `$...$` LaTeX. Position: after the theoretical-foundations chapter (density-matrix affect, quantum-cognition lineage), before the general discussion.*

---

## §1. Introduction: From Structural Claim to Measurable Dynamics

- Restate the dissertation's structural claim: affect decoheres hierarchically across three nested timescales — reactive ($\sim$100 ms), adaptive ($\sim$1 s), reflective ($\sim$10 s). Ambivalence (coherent superposition, off-diagonal $\rho_{ij} \neq 0$) is a property of the *fast* layers; the reflective layer delivers near-classical, near-diagonal states ("the collapse of affect").
- The problem this chapter solves: as stated so far, hierarchical decoherence is a *structural* postulate. This chapter operationalizes it in a trainable multi-timescale architecture, converting the postulate into quantitative, falsifiable predictions about coherence decay across levels.
- Chapter thesis in one sentence: **each slower processing level acts as an environment for the level below; estimated affective coherence must therefore decay monotonically up the hierarchy on ambivalent stimuli, and only there.**
- Contribution list: (i) formal three-level open-system model; (ii) architecture whose levels output density-matrix parameters rather than labels; (iii) the coherence-decay prediction and its empirical test; (iv) nested free-energy (valence/arousal) signals from inter-level prediction error.

## §2. Formal Model: Nested Open Systems and Inter-Level Measurement

### §2.1 Three-level state space
- Affective state at level $\ell \in \{1,2,3\}$: density operator $\rho^{(\ell)}$ on emotion space $\mathcal{H}_e$ ($\dim = n$ emotion categories), evolving at characteristic timescale $\tau_\ell \in \{10^{-1}, 10^{0}, 10^{1}\}$ s.
- Composite state on $\mathcal{H}_e \otimes \mathcal{H}_{c}^{(\ell+1)}$: fast affect entangled with the slower level's *context* degrees of freedom.

### §2.2 Slower levels as environments (the key move)
- Reduced fast-level dynamics as a partial trace over the slower context:
$$\rho^{(\ell)}(t) = \operatorname{Tr}_{c}\!\left[\, U_t \left(\rho^{(\ell)} \otimes \rho^{(\ell+1)}_{c}\right) U_t^\dagger \,\right]$$
- Consequence: coherences of $\rho^{(\ell)}$ decay at rates set by the entangling interaction with level $\ell+1$; effective Lindblad form at each timescale, with decoherence rates $\gamma_\ell$ ordered $\gamma_1 < \gamma_2 < \gamma_3$ in integrated effect (slower levels have "measured more").
- State the ordering prediction that anchors §5:
$$C\big(\rho^{(1)}\big) \;\ge\; C\big(\rho^{(2)}\big) \;\ge\; C\big(\rho^{(3)}\big), \qquad C(\rho) = \sum_{i \ne j} |\rho_{ij}| \ \text{(} \ell_1 \text{ coherence)}$$

### §2.3 Inter-level readout as nested Naimark dilation
- Each level's readout of the level below is a POVM $\{F_i^{(\ell)}\}$; its Naimark dilation is a sharp (projective) measurement on affect $\otimes$ context at level $\ell+1$. Unsharpness at level $\ell$ = the shadow of the traced-out context.
- Connects the chapter to the measurement-theoretic backbone of the dissertation: the reflective label is the (approximately) projective end of a dilation chain whose restrictions downward are progressively fuzzier.
- Contrast with single-level quantum-cognition models (Aerts lineage): one Hilbert space, one measurement; here, *measurement itself is stratified in time*.

### §2.4 Ambivalence vs. uncertainty under the hierarchy
- Ambivalent input: $\rho^{(1)}$ has large off-diagonals → decoherence has something to destroy → strict coherence decay upward.
- Uncertain input: $\rho^{(1)}$ already a classical mixture (diagonal) → nothing to decohere → coherence flat (≈0) across levels.
- This asymmetry is what no classical hierarchical classifier predicts, and is the discriminating signature of the framework.

## §3. Computational Realization: The Three-Level Architecture

### §3.1 Level-to-layer mapping
| Theory layer | Timescale | Architecture level | Backbone |
|---|---|---|---|
| Reactive | $\sim$100 ms | Level 1: frame | CNN + TCN |
| Adaptive | $\sim$1 s | Level 2: sequence | Transformer (expression evolution) |
| Reflective | $\sim$10 s | Level 3: episode | Hierarchical transformer (emotional context) |

### §3.2 Density-matrix heads
- Each level outputs parameters of $\rho^{(\ell)}$, not a softmax: Cholesky parametrization $\rho = LL^\dagger / \operatorname{Tr}[LL^\dagger]$ guarantees positivity and unit trace.
- Off-diagonals must be *earned*, not decorative: coherence estimated from cross-channel/cross-modal disagreement (per the Seo & Kim analytic construction), so that $\rho_{ij}$ carries the interference structure of conflicting evidence rather than a reparametrized softmax.
- Diagonal-only ablation defined here (used in §5 as the "classical control" model).

### §3.3 Inter-level coupling
- Level 2 as learned dynamical map: predicts $\rho^{(1)}(t + \Delta)$ from the frame history — an empirically fitted CPTP/Lindblad generator at the 1 s scale.
- Level 3 as context state: episode embedding plays $\rho_c$ in §2.2; its coupling strength to Level 1/2 estimates is the learned analogue of the decoherence rate $\gamma$.
- Training objectives: per-level supervision (annotator distributions, not hard labels), dynamical consistency loss between Level 2 predictions and Level 1 estimates, and a *no coherence penalty/bonus* — the coherence ordering must emerge, not be imposed (critical for §5's evidential value).

## §4. Free-Energy Dynamics Across Levels

- Implement the dissertation's identification valence $= dF/dt$, arousal $= d^2F/dt^2$ with $F^{(\ell)}$ = level-$\ell$ prediction error over level $\ell-1$ (predictive-coding reading).
- Yields three nested valence/arousal signals — reactive, adaptive, reflective affect can diverge on the same stimulus. Cross-timescale divergence is itself an operationalization of ambivalence, dual to the off-diagonal signature.
- Derive the relation between coherence decay ($\S2.2$) and free-energy descent: decoherence as the resolution of inter-level prediction error; "collapse of affect" = joint minimization.

## §5. Predictions and Experimental Design

### §5.1 Hypotheses
- **H1 (coherence decay):** on ambivalent stimuli, $C(\rho^{(1)}) > C(\rho^{(2)}) > C(\rho^{(3)})$; effect absent on uncertain stimuli (§2.4).
- **H2 (dissociation):** ambivalence vs. uncertainty are dissociable in the model exactly where human annotators dissociate them (contested-pair vs. low-agreement-diffuse items).
- **H3 (behavioral correlate):** items with high Level-1 coherence but low Level-3 coherence predict annotator response phenomena that diagonal models cannot: bimodal label distributions, longer annotation latencies, order effects.

### §5.2 Materials
- IEMOCAP contested emotion pairs (continuity with the companion paper); video corpus for the full 10 s episodic scale (e.g., Aff-Wild2 or equivalent), partitioned into *ambivalent* (systematic annotator split) vs. *uncertain* (diffuse disagreement) subsets — operational criteria stated here.

### §5.3 Analyses
- Primary: per-item coherence trajectory across levels; ordered-hypothesis test for H1; interaction (stimulus type × level) as the critical statistic.
- Controls/ablations: diagonal-only heads (classical control); shuffled inter-level coupling (breaks the environment structure); single-timescale model with matched parameters (shows hierarchy, not capacity, carries the effect).
- Robustness: coherence measure choice ($\ell_1$ vs. relative entropy of coherence), basis dependence addressed (coherence reported in the annotator-category basis, with basis-choice discussion).

## §6. Results

*(Structure now; populate after experiments.)*
- 6.1 Coherence-decay profiles by stimulus class (H1)
- 6.2 Ambivalence/uncertainty dissociation vs. human annotator structure (H2)
- 6.3 Behavioral correlates and comparison to diagonal/classical controls (H3)
- 6.4 Learned decoherence rates $\hat\gamma_\ell$ vs. the postulated timescale ordering

## §7. Discussion

- What is established: hierarchical decoherence promoted from postulate to measured dynamics; the quantum formalism does non-decorative explanatory work (the H1 asymmetry).
- Relation to the lineage: extends Aerts-style single-measurement models to temporally stratified measurement; connects two-mode (emergent vs. logical) thought to the reactive/reflective distinction — fast layers live in the emergent sector, the reflective layer approximates the logical one.
- Objections addressed head-on: (i) "the network is just a classifier in quantum clothes" → answered by ablations and by the emergent (untrained) coherence ordering; (ii) basis-dependence of coherence; (iii) analogy vs. mechanism — the claim is structural/representational, not that neurons are quantum.
- Limitations: dataset scale at the 10 s level; identifiability of $\gamma_\ell$; POVM elements learned, not derived.
- Bridge to the final chapter: decoherence hierarchy as the precondition for "existential stakes" — a system whose reflective layer irreversibly collapses its own affective superpositions.

## Appendix (chapter-local)

- A.1 Architecture and training details (backbones, losses, hyperparameters) — kept out of the main text deliberately.
- A.2 Cholesky parametrization and coherence-measure definitions.
- A.3 Dataset partition criteria and annotator-statistics preprocessing.
