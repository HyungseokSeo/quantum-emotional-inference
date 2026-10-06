# From Concept Disjunction to Affective Superposition: The Quantum-Cognition Lineage

*Draft section for the theoretical-foundations / related-work chapter of* **Quantum Emotional Inference and the Collapse of Affect**. *StackEdit-ready: Markdown with `$...$` LaTeX.*

---

## 1. The founding anomaly: $\mu(A \text{ or } B)$

The quantum-cognition program begins with an empirical anomaly in human concept combination. Hampton (1988a, 1988b) asked subjects to rate the membership of items in concepts $A$, $B$, and their conjunction "$A$ and $B$" or disjunction "$A$ or $B$", obtaining membership weights $\mu(A)$, $\mu(B)$, $\mu(A \text{ and } B)$, $\mu(A \text{ or } B)$. The data systematically violate every classical (fuzzy-set or Kolmogorovian) model of the connectives: subjects judge *Pencil Eraser* a better member of *Instruments or Tools* than a bare average would allow, yet with $\mu(A \text{ or } B) = 0.45$ **below** both $\mu(A) = 0.4$-compatible bounds and $\mu(B) = 0.7$ — "underextension" that no measure satisfying $\mu(A \text{ or } B) \geq \max(\mu(A), \mu(B))$ can produce. The dual phenomenon for conjunction is the classic "guppy effect" of overextension.

Aerts (2009) showed these deviations are not noise but *structure*: they are exactly the signature of quantum probability. Modeling the disjunction as the superposition $\tfrac{1}{\sqrt{2}}(|A\rangle + |B\rangle)$ and membership as a projection measurement $M$ yields

$$\mu(A \text{ or } B) = \frac{\mu(A) + \mu(B)}{2} + \Re\langle A|M|B\rangle,$$

where $\Re\langle A|M|B\rangle$ is an **interference term**. Fitting Hampton's data amounts to measuring cognitive interference: the deviation of human judgment from the classical average is carried entirely by the off-diagonal matrix element between the component concepts.

## 2. The Fock-space turn: why quantum *field* theory

At the close of this analysis Aerts (2009, §1.6) identifies a structural obstruction. In the no-interference limit the quantum formula reduces to $\tfrac{1}{2}(\mu(A) + \mu(B))$ — the behavior of a classical *particle* at a double slit, not of classical *logic*, which would demand $\mu(A) + \mu(B) - \mu(A)\mu(B)$. A single Hilbert space therefore cannot contain both the interference-bearing mode of judgment *and* genuinely logical combination as limiting cases. His resolution is the state space of quantum field theory: **Fock space**,

$$\mathcal{F} = \mathcal{H} \oplus (\mathcal{H} \otimes \mathcal{H}),$$

in which the number of conceptual entities is variable. In the two-entity sector $\mathcal{H} \otimes \mathcal{H}$, the subject holds $A$ and $B$ as *two* concepts and combines the outcomes logically. In the one-entity sector $\mathcal{H}$, the phrase "$A$ or $B$" is a *newly created* concept — a single emergent entity in the superposition state $\tfrac{1}{\sqrt{2}}(|A\rangle + |B\rangle)$. A general cognitive state weights the two sectors, giving

$$\mu(A \text{ or } B) = m^2\big(\mu(A) + \mu(B) - \mu(A)\mu(B)\big) + n^2\Big(\tfrac{\mu(A)+\mu(B)}{2} + \Re\langle A|M|B\rangle\Big), \qquad m^2 + n^2 = 1.$$

The field-theoretic apparatus is thus not decorative. Concept combination is *entity creation*: forming "$A$ or $B$" is the cognitive analogue of particle creation, and only a formalism with variable particle number can host it. Aerts reads the two sectors as **two modes of human thought** — a logical mode and an emergent (conceptual) mode — with actual judgment a superposition of both.

## 3. Development of the program, 2009–present

Three strands of subsequent work matter for this dissertation.

**(a) The two-mode model was confirmed and quantified.** Extensions to conjunction, negation, and borderline contradictions (Sozzo, 2014; Aerts, Sozzo & Veloz, 2015a, 2015b) fit the sector weights across large datasets and found the emergent sector *dominant*: human combination judgments load primarily on the one-entity (superposition) sector, with the logical sector a minority contribution. Concept combinations were further shown to violate the CHSH inequality (Aerts & Sozzo, 2011), establishing entanglement — not merely interference — in compositional cognition.

**(b) Quantum cognition became a mature field.** Busemeyer & Bruza (2012) systematized quantum probability as a general framework for judgment and decision; Wang et al. (2014) derived and confirmed a parameter-free prediction (the QQ equality) for question-order effects across 72 national surveys; reviews in *Trends in Cognitive Sciences* (Bruza, Wang & Busemeyer, 2015) and the *Annual Review of Psychology* (Pothos & Busemeyer, 2022) consolidated interference, contextuality, and order effects as standard modeling tools.

**(c) The QFT analogy became literal.** Aerts & Beltran (2020, 2022) took the final step implicit in the Fock-space construction: if concepts are quantum entities, identical concepts should be *indistinguishable* and obey quantum statistics. They showed that word frequencies in human texts follow a **Bose–Einstein** rather than Maxwell–Boltzmann distribution, modeling language as a boson gas of entangled words ("cognitons") and linking Zipf's law to Bose–Einstein condensation. What began as a borrowed state space is now a statistical field theory of cognition.

## 4. Position of this dissertation in the lineage

The present work transposes this program from *concept membership* to *affect*, and from human judgment data to machine representation. The correspondence is point-for-point:

| Aerts program (concepts) | This dissertation (affect) |
|---|---|
| Concepts $A$, $B$ | Emotion categories $e_i$, $e_j$ |
| Membership weight $\mu(A)$ of item $X$ | Class probability $p_i$ for a stimulus |
| Interference term $\Re\langle A|M|B\rangle$ | Off-diagonal coherence $\rho_{ij}$ derived from cross-modal disagreement |
| Hampton's over-/underextension data | IEMOCAP annotator disagreement; contested emotion pairs |
| One-entity sector: emergent concept, superposition | Ambivalence: coherent superposition of emotions (reactive layer) |
| Two-entity sector: logical combination | Uncertainty: classical mixture; definite reflective label |
| Sector weights $m^2, n^2$ | Position along the hierarchical decoherence schedule |

Three aspects of this transposition deserve emphasis.

**Ambivalence versus uncertainty is the sector distinction.** Aerts' emergent mode — one entity in superposition — is precisely what this dissertation calls *ambivalence*: multiple emotions coherently co-present before expression, represented by off-diagonal terms of a density matrix. His logical mode — two entities combined by classical rules — corresponds to *uncertainty*: a classical mixture over which single emotion obtains. Where scalar probability outputs collapse the distinction, the density matrix preserves it, exactly as the interference term $\Re\langle A|M|B\rangle$ preserves what the average $\tfrac{1}{2}(\mu(A)+\mu(B))$ discards.

**Annotator disagreement plays the role of Hampton's data.** In the companion work (Seo & Kim), coherence is not learned but derived analytically from inter-modal deviation, and the contested emotion pair is statistically recoverable from the off-diagonal pattern. This is the affective analogue of Aerts' central empirical claim: the non-classical residue in human responses is *structured*, localized on specific pairs, and recoverable from off-diagonal elements — not compressible to a scalar conflict measure, just as Hampton's deviations are not noise around a classical mean.

**Hierarchical decoherence dynamizes the two modes.** Aerts' model is static: $m^2$ and $n^2$ weight two coexisting modes at the moment of judgment. The present framework's reactive $\to$ adaptive $\to$ reflective hierarchy supplies what the Fock-space picture lacks — a *temporal mechanism* by which cognition passes from the emergent sector (high coherence, $\sim$100 ms, pre-reflective superposition) to the logical sector (decohered, $\sim$10 s, conscious classical limit). The "collapse of affect" is, in this reading, sector transfer under environmental coupling: decoherence as the dynamical bridge between Aerts' two modes of thought.

**Outlook.** The lineage suggests concrete extensions. Mixed emotions such as *bittersweet* invite a two-sector Fock model of affective blends — emergent blend versus logical co-occurrence — with sector weights estimable from contested-pair statistics, mirroring the $\mu(A \text{ or } B)$ methodology with emotion-pair judgments. And the Bose–Einstein statistics found in general language (Aerts & Beltran, 2020) invite a test on affective speech corpora: whether emotional expression exhibits the same quantum-statistical signature, which would ground the density-matrix representation not merely as a convenient formalism but as the native statistics of expressed affect.

## References

- Aerts, D. (2009). Quantum structure in cognition. *Journal of Mathematical Psychology*, 53(5), 314–348.
- Aerts, D., & Beltran, L. (2020). Quantum structure in cognition: Human language as a Boson gas of entangled words. *Foundations of Science*, 25, 755–802.
- Aerts, D., & Beltran, L. (2022). Are words the quanta of human language? Extending the domain of quantum cognition. *Entropy*, 24(1), 6.
- Aerts, D., & Gabora, L. (2005). A theory of concepts and their combinations I–II. *Kybernetes*, 34(1/2), 167–221.
- Aerts, D., Gabora, L., & Sozzo, S. (2013). Concepts and their dynamics: A quantum-theoretic modeling of human thought. *Topics in Cognitive Science*, 5(4), 737–772.
- Aerts, D., & Sozzo, S. (2011). Quantum structure in cognition: Why and how concepts are entangled. In *Quantum Interaction 2011*, LNCS 7052 (pp. 116–127). Springer.
- Aerts, D., Sozzo, S., & Veloz, T. (2015a). Quantum structure of negation and conjunction in human thought. *Frontiers in Psychology*, 6, 1447.
- Aerts, D., Sozzo, S., & Veloz, T. (2015b). New fundamental evidence of non-classical structure in the combination of natural concepts. *Philosophical Transactions of the Royal Society A*, 374, 20150095.
- Bruza, P. D., Wang, Z., & Busemeyer, J. R. (2015). Quantum cognition: A new theoretical approach to psychology. *Trends in Cognitive Sciences*, 19(7), 383–393.
- Busemeyer, J. R., & Bruza, P. D. (2012). *Quantum Models of Cognition and Decision*. Cambridge University Press.
- Hampton, J. A. (1988a). Overextension of conjunctive concepts: Evidence for a unitary model of concept typicality and class inclusion. *Journal of Experimental Psychology: Learning, Memory, and Cognition*, 14(1), 12–32.
- Hampton, J. A. (1988b). Disjunction of natural concepts. *Memory & Cognition*, 16, 579–591.
- Pothos, E. M., & Busemeyer, J. R. (2022). Quantum cognition. *Annual Review of Psychology*, 73, 749–778.
- Sozzo, S. (2014). A quantum probability explanation in Fock space for borderline contradictions. *Journal of Mathematical Psychology*, 58, 1–12.
- Wang, Z., Solloway, T., Shiffrin, R. M., & Busemeyer, J. R. (2014). Context effects produced by question orders reveal quantum nature of human judgments. *Proceedings of the National Academy of Sciences*, 111(26), 9431–9436.
