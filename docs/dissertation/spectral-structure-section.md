# §2.5 Spectral Structure of Affective Relaxation

*Section draft for the hierarchical-decoherence chapter of* **Quantum Emotional Inference and the Collapse of Affect**. *StackEdit-ready: Markdown with `$...$` LaTeX. Position: after §2.2 (slower levels as environments) and before §2.3 (nested Naimark dilation). Purpose: restate the three-timescale hierarchy as a property of the spectrum of the affective generator, so that it becomes computable from a fitted model rather than postulated.*

---

## §2.5.1 The affective generator as a matrix

§2.2 established that, after tracing out the slower context, the affective state at level $\ell$ obeys an effective master equation of Gorini–Kossakowski–Sudarshan–Lindblad (GKSL) form,

$$\dot\rho^{(\ell)} = \mathcal L_\ell\big[\rho^{(\ell)}\big] = -i\big[H_\ell,\rho^{(\ell)}\big] + \sum_k \gamma_{\ell k}\Big(L_{\ell k}\,\rho^{(\ell)} L_{\ell k}^\dagger - \tfrac12\big\{L_{\ell k}^\dagger L_{\ell k},\rho^{(\ell)}\big\}\Big). \tag{2.5.1}$$

The generator $\mathcal L_\ell$ is a linear map on operators. Writing $\rho$ as a vector $|\rho\rangle\!\rangle \in \mathbb C^{n^2}$ by stacking its columns, $\mathcal L_\ell$ becomes an $n^2 \times n^2$ matrix,

$$\hat{\mathcal L}_\ell = -i\big(H_\ell\otimes I - I\otimes H_\ell^{T}\big) + \sum_k \gamma_{\ell k}\Big(L_{\ell k}\otimes \bar L_{\ell k} - \tfrac12\,L_{\ell k}^\dagger L_{\ell k}\otimes I - \tfrac12\, I\otimes L_{\ell k}^{T}\bar L_{\ell k}\Big). \tag{2.5.2}$$

Its eigenvalues $\{\lambda_j\}_{j=0}^{n^2-1}$ satisfy $\operatorname{Re}\lambda_j \le 0$, with $\lambda_0 = 0$ corresponding to the stationary state $\rho_\infty$. When $\hat{\mathcal L}_\ell$ is diagonalizable, the solution of (2.5.1) is

$$\rho^{(\ell)}(t) = \rho_\infty + \sum_{j\ge 1} c_j\, e^{\lambda_j t}\, R_j, \qquad c_j = \operatorname{Tr}\big[L_j^\dagger \rho^{(\ell)}(0)\big], \tag{2.5.3}$$

where $R_j$ and $L_j$ are the right and left eigenoperators, biorthogonal under the Hilbert–Schmidt inner product. Each term is one **relaxation mode**: $-\operatorname{Re}\lambda_j$ is its decay rate, $\operatorname{Im}\lambda_j$ its oscillation frequency, and $R_j$ the pattern in the density matrix that decays. The **Liouvillian gap**

$$\Delta_\ell = \min_{j\ge 1}\,\big|\operatorname{Re}\lambda_j\big| \tag{2.5.4}$$

is the slowest nonzero rate, so $1/\Delta_\ell$ is the time for level $\ell$ to reach its stationary state from a generic initial condition.

The spectrum therefore gives three things the trajectory alone does not: *how many* distinct timescales the dynamics has, *what* decays on each, and *where* the dynamics ends.

## §2.5.2 The hierarchy as a spectral statement

The structural claim of this dissertation is that affect relaxes on three nested timescales, $\tau_\ell \in \{10^{-1}, 10^{0}, 10^{1}\}$ s. In the language of §2.5.1 this is the claim that the eigenvalues of the full affective generator cluster into **three bands**,

$$\big|\operatorname{Re}\lambda_j\big| \in \mathcal B_1 \cup \mathcal B_2 \cup \mathcal B_3, \qquad \mathcal B_1 \sim 10\ \mathrm{s^{-1}},\quad \mathcal B_2 \sim 1\ \mathrm{s^{-1}},\quad \mathcal B_3 \sim 0.1\ \mathrm{s^{-1}}, \tag{2.5.5}$$

separated by gaps of roughly one decade. Two consequences follow immediately.

**The reflective layer is the Liouvillian gap.** The slowest band $\mathcal B_3$ contains $\Delta$. The collapse of affect is complete on the timescale $1/\Delta \approx 10$ s, which is why the reflective label is the one that behaves classically.

**The ordering of §2.2 becomes a statement about eigenoperators.** The prediction $C(\rho^{(1)}) \ge C(\rho^{(2)}) \ge C(\rho^{(3)})$ holds when the eigenoperators $R_j$ in the fast bands $\mathcal B_1, \mathcal B_2$ have most of their weight on off-diagonal entries of $\rho$ in the emotion-category basis, while those in $\mathcal B_3$ are predominantly diagonal. The coherence-decay profile of §5 is then a projection of the spectral structure onto one scalar.

This reformulation converts a postulate into a test. Given any fitted generator, whether constructed analytically or extracted from the learned Level-2 dynamical map of §3.3, one computes the spectrum of $\hat{\mathcal L}$ and inspects $\operatorname{Re}\lambda_j$ on a logarithmic axis. Three clusters support the hierarchy; a continuum or a different number of clusters falsifies it at the level of that model.

> **Prediction S1 (spectral bands).** The eigenvalues of the affective generator fitted to naturalistic emotional episodes cluster into three bands separated by roughly one decade, with the slowest band carrying predominantly diagonal eigenoperators.

## §2.5.3 Which part of emotion relaxes last

The gap identifies not only how slow the slowest process is but *which quantity* carries it. A two-category example makes this explicit. Let $\mathcal H_e = \operatorname{span}\{|0\rangle, |1\rangle\}$, $H = \tfrac{\Delta_E}{2}\sigma_z$, and consider two channels: a relaxation channel $L_1 = \sigma_- = |0\rangle\langle 1|$ with rate $\gamma$, and a dephasing channel $L_2 = \sigma_z$ with rate $\gamma_\phi$. Writing $\rho = \begin{pmatrix} p_0 & c \\ \bar c & p_1\end{pmatrix}$, equation (2.5.1) decouples into

$$\dot p_1 = -\gamma\, p_1, \qquad \dot c = -\big(\tfrac{\gamma}{2} + 2\gamma_\phi + i\Delta_E\big)\, c. \tag{2.5.6}$$

The spectrum is $\{0,\ -\gamma,\ -\tfrac{\gamma}{2} - 2\gamma_\phi \pm i\Delta_E\}$. Hence

| Regime | Gap $\Delta$ | Slowest mode | Reading |
|---|---|---|---|
| $\gamma_\phi < \gamma/4$ | $\tfrac{\gamma}{2} + 2\gamma_\phi$ | coherence $c$ | ambivalence outlives uncertainty |
| $\gamma_\phi > \gamma/4$ | $\gamma$ | population $p_1$ | uncertainty outlives ambivalence |

In the pure-relaxation case, the off-diagonal entry is the *last* thing to disappear. In the dephasing-dominated case, the categorical uncertainty is. This is a qualitative distinction with a direct empirical counterpart on contested stimuli: whether the cross-modal disagreement that generates coherence (Seo & Kim) persists longer than the drift of the category probabilities themselves.

> **Prediction S2 (carrier of the gap).** On ambivalent items, the slowest eigenoperator has its dominant weight on the off-diagonal pair corresponding to the contested categories; on uncertain items, it is diagonal. Equivalently, the ratio of effective dephasing to effective relaxation, $\gamma_\phi/\gamma$, is smaller for ambivalent than for uncertain stimuli.

## §2.5.4 Metastability: mood as a slow manifold

When the spectrum has a large internal separation, so that a few eigenvalues lie close to zero and the rest far from it, relaxation proceeds in two stages. On the fast timescale the state collapses onto the **metastable manifold** spanned by $\rho_\infty$ and the slow eigenoperators; on the slow timescale it moves within that manifold toward $\rho_\infty$ (Macieszczak, Guţă, Lesanovsky and Garrahan, 2016). The metastable manifold is itself a convex set of approximately stationary states, which in the simplest case is a simplex whose vertices are the *metastable phases*.

The affective reading is immediate. **Emotion** is the fast transient onto the manifold; **mood** is the position within it; the **gap** is the mood lifetime. A mood is not a separate construct bolted onto the emotion model, it is the slow sector of the same generator. Mood transitions appear as motion between vertices of the metastable simplex, and their rates are the eigenvalues in $\mathcal B_3$.

This also sharpens the hierarchy of §2.2. The three bands of (2.5.5) correspond to a *nested* sequence of metastable manifolds: the reactive dynamics collapses onto the adaptive manifold within $\sim 10^{-1}$ s, the adaptive dynamics onto the reflective manifold within $\sim 10^{0}$ s, and the reflective dynamics onto $\rho_\infty$ within $\sim 10^{1}$ s.

## §2.5.5 Dark states and the design of regulatory channels

The spectrum describes relaxation. The complementary question is constructive: which stationary states can a given set of channels produce, and how can channels be chosen to make a *desired* state stationary? This is the question of emotion regulation posed in operator language.

A pure state $|D\rangle \in \mathcal H_e$ is a **dark state** of (2.5.1) when

$$L_k |D\rangle = 0 \ \ \text{for all } k, \qquad H|D\rangle = E\,|D\rangle. \tag{2.5.7}$$

Then $\rho_D = |D\rangle\langle D|$ satisfies $\mathcal L[\rho_D] = 0$. Three properties make dark states the right formalization of a regulated affective state.

**A dark state is pure, so it can be coherent.** $\rho_D$ generally has nonzero off-diagonals in the emotion-category basis. A "quiet" dark state is therefore not a classical mixture and not an absence of affect; it is a *stabilized superposition*, an ambivalence that no longer decays. This separates two notions that the single-timescale literature conflates: *decoherence*, which destroys off-diagonals, and *regulation*, which stabilizes a target state whether or not it carries off-diagonals. Equanimity in this framework is a coherent fixed point, not a decohered one.

**A dark state is reached dissipatively.** Choose jump operators of the form

$$L_k = |D\rangle\langle B_k|, \qquad \{|B_k\rangle\} \text{ an orthonormal basis of } |D\rangle^\perp, \tag{2.5.8}$$

so that every channel pumps a *bright* state $|B_k\rangle$ into $|D\rangle$. If in addition no proper subspace of $|D\rangle^\perp$ is invariant under $H$ and all $L_k$ jointly, then $\rho_D$ is the unique stationary state and every initial condition converges to it. This is dissipative state engineering in the sense of Diehl et al. (2008), Kraus et al. (2008) and Verstraete, Wolf and Cirac (2009): the environment, here the slower contextual level, is designed so that its fixed point is the target. In cognitive terms, regulation is the choice of which transitions the context makes irreversible.

**The approach rate is set by the gap, and it is not monotone in the channel strength.** Consider a three-category chain in which an initial state $|A\rangle$ is coupled coherently to a transit state $|B\rangle$ by $H = \Omega\big(|A\rangle\langle B| + |B\rangle\langle A|\big)$, and $|B\rangle$ decays into the dark state through $L = \sqrt\gamma\,|D\rangle\langle B|$. The population of $|A\rangle$ leaves at the effective rate

$$\Gamma_{\mathrm{eff}}(\gamma) = \begin{cases} \gamma/2, & \gamma \le 4\Omega, \\[4pt] \dfrac{\gamma}{2} - \sqrt{\dfrac{\gamma^2}{4} - 4\Omega^2}\ \xrightarrow{\ \gamma \gg \Omega\ }\ \dfrac{4\Omega^2}{\gamma}, & \gamma > 4\Omega. \end{cases} \tag{2.5.9}$$

The rate rises with $\gamma$ up to critical damping at $\gamma = 4\Omega$ and then *falls* as $4\Omega^2/\gamma$. Strong dissipation on the transit state freezes the coherent step that feeds it, a dissipative Zeno effect. The Liouvillian gap itself is carried by the coherence between $|A\rangle$ and $|D\rangle$ and equals $\Gamma_{\mathrm{eff}}/2$ for every $\gamma$, so it peaks at the same $\gamma = 4\Omega$: the last trace of the unregulated state to vanish is its ambivalence with the regulated one, and it vanishes at half the rate of the population transfer. Both curves are reproduced numerically in `experiments/04_hierarchy_validation/liouvillian_spectrum.py demo`. The practical content for regulation is that a channel which acts too forcefully on an intermediate affective state *slows* settling into the target. Coordinated design of $\{L_k, \gamma_k\}$ therefore means satisfying two conditions at once: (i) uniqueness of $\rho_D$ via (2.5.7)–(2.5.8), and (ii) maximization of the gap, which places $\gamma$ near $4\Omega$ rather than as large as possible.

> **Prediction S3 (non-monotone regulation).** When a regulatory influence is operationalized as the strength of a contextual channel acting on an intermediate affective state, the time to reach the regulated state is U-shaped in that strength, with a minimum where the dissipative rate matches the coherent coupling.

## §2.5.6 Spectral reading of valence and arousal

The free-energy identification of §4, valence $= dF/dt$ and arousal $= d^2F/dt^2$, lives on trajectories. The spectrum lifts it to the generator. Expanding $F$ to quadratic order about $\rho_\infty$ and using (2.5.3), each mode contributes to leading order

$$F_j(t) \propto |c_j|^2\, e^{2\operatorname{Re}\lambda_j\, t}, \qquad \frac{dF_j}{dt} = 2\operatorname{Re}\lambda_j\, F_j, \qquad \frac{d^2F_j}{dt^2} = 4\big(\operatorname{Re}\lambda_j\big)^2 F_j, \tag{2.5.10}$$

neglecting cross terms between non-orthogonal modes. Along mode $j$, valence magnitude scales with $|\operatorname{Re}\lambda_j|$ and arousal with $(\operatorname{Re}\lambda_j)^2$. The spectral bands of (2.5.5) are therefore also bands of arousal: modes in $\mathcal B_1$ are high-arousal by construction, modes in $\mathcal B_3$ are low-arousal. The nested valence and arousal signals of §4 are the projections of free-energy descent onto the three bands. This is the formal link between the open-system and the predictive-coding halves of the dissertation.

## §2.5.7 Estimating the spectrum from data

The learned Level-2 dynamical map of §3.3 predicts $\rho^{(1)}(t + \tau)$ from the frame history. Restricted to its linear part it is a completely positive trace-preserving (CPTP) map $\Phi_\tau$, and the generator follows from

$$\hat{\mathcal L} = \tfrac{1}{\tau}\log \hat\Phi_\tau, \tag{2.5.11}$$

whenever the principal logarithm exists and is of GKSL form (Wolf, Eisert, Cubitt and Cirac, 2008). The protocol is:

1. Fit $\Phi_\tau$ at the Level-2 step $\tau \approx 1$ s, and the analogous maps at the Level-1 and Level-3 steps.
2. Take the matrix logarithm, verify that the result is Hermiticity- and trace-preserving, and project onto GKSL form if necessary.
3. Compute the eigenvalues $\lambda_j$ and eigenoperators $R_j$; plot $|\operatorname{Re}\lambda_j|$ on a logarithmic axis (Prediction S1).
4. For each $R_j$, compute the fraction of its Hilbert–Schmidt norm on off-diagonal entries in the annotator-category basis; test whether this fraction decreases across bands and differs between ambivalent and uncertain item sets (Prediction S2).
5. Identify the metastable manifold from the eigenoperators in $\mathcal B_3$ and compare its vertices with annotator mood descriptors where available (§2.5.4).

At $n \le 8$ emotion categories, $\hat{\mathcal L}$ is at most $64 \times 64$ and the computation is immediate. The protocol is implemented in `src/quantum/dynamics/liouvillian.py` and driven by `experiments/04_hierarchy_validation/liouvillian_spectrum.py`, whose `channel` mode takes a saved $\hat\Phi_\tau$ and performs steps 2–4; the Wolf et al. criterion (Hermiticity preservation, trace annihilation, conditional complete positivity) is reported as a pass/fail diagnostic before any spectral claim is made.

## §2.5.8 Caveats

- **Basis dependence.** Jump operators and coherence are both defined relative to a basis. All spectral statements here are made in the annotator-category basis, consistent with §5.3. The dark-state construction (2.5.8) presupposes that the target $|D\rangle$ is specified in that basis.
- **Markovianity.** GKSL generators have no memory. Context effects in which an earlier episode resurfaces after an intervening one lie outside the present form and would require a time-nonlocal or embedded-Markov extension.
- **Non-normality.** $\hat{\mathcal L}$ is in general non-normal, so eigenvalues alone do not bound transient behavior; short-time coherence can grow before it decays. Where this matters, the pseudospectrum rather than the spectrum is the relevant object.
- **Identifiability.** The map $\Phi_\tau \mapsto \hat{\mathcal L}$ is not unique when $\Phi_\tau$ has negative or complex eigenvalues, and a given spectrum is compatible with many $\{H, L_k, \gamma_k\}$. The predictions S1–S3 are stated at the level of the spectrum and eigenoperators, which are identifiable, rather than at the level of individual jump operators, which are not.

---

### References to add to the chapter bibliography

- Gorini, V., Kossakowski, A., & Sudarshan, E. C. G. (1976). Completely positive dynamical semigroups of N-level systems. *J. Math. Phys.*, 17, 821.
- Lindblad, G. (1976). On the generators of quantum dynamical semigroups. *Commun. Math. Phys.*, 48, 119.
- Breuer, H.-P., & Petruccione, F. (2002). *The Theory of Open Quantum Systems*. Oxford University Press.
- Macieszczak, K., Guţă, M., Lesanovsky, I., & Garrahan, J. P. (2016). Towards a theory of metastability in open quantum dynamics. *Phys. Rev. Lett.*, 116, 240404.
- Diehl, S., Micheli, A., Kantian, A., Kraus, B., Büchler, H. P., & Zoller, P. (2008). Quantum states and phases in driven open quantum systems with cold atoms. *Nature Physics*, 4, 878.
- Kraus, B., Büchler, H. P., Diehl, S., Kantian, A., Micheli, A., & Zoller, P. (2008). Preparation of entangled states by quantum Markov processes. *Phys. Rev. A*, 78, 042307.
- Verstraete, F., Wolf, M. M., & Cirac, J. I. (2009). Quantum computation and quantum-state engineering driven by dissipation. *Nature Physics*, 5, 633.
- Albert, V. V., & Jiang, L. (2014). Symmetries and conserved quantities in Lindblad master equations. *Phys. Rev. A*, 89, 022118.
- Wolf, M. M., Eisert, J., Cubitt, T. S., & Cirac, J. I. (2008). Assessing non-Markovian quantum dynamics. *Phys. Rev. Lett.*, 101, 150402.
- Asano, M., Basieva, I., Khrennikov, A., Ohya, M., & Tanaka, Y. (2012). Quantum-like dynamics of decision-making. *Physica A*, 391, 2083.
