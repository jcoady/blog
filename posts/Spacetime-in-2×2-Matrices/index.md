---
title: "Spacetime in 2×2 Matrices over a Finite Field"
subtitle: "How a Galois extension sidesteps a Witt-index obstruction, fixes a composition anomaly, and gives a small, checkable model of Lorentz holonomy"
author: "John D. Coady"
date: "2026-10-04"
categories: [pure-math, geometry, hyperbolic-geometry, mathematical-physics]
description: "A plain-language tour of a Zenodo preprint on a 2×2 Hermitian-matrix model of (1+3)D spacetime over F_p, with exact finite-field checks."
doi: 10.5281/zenodo.23138782
format:
  html:
    toc: true
    toc-depth: 2
    code-fold: true
    number-sections: false
---

::: {.callout-note}
**Paper and code.** *A Projective 2×2 Rotor Architecture for (1+3)D Discrete Spacetime: Holonomies, Ghost Parity, and the Double-Cover Anchor Map.* Zenodo preprint, CC BY 4.0, [doi:10.5281/zenodo.23138782](https://doi.org/10.5281/zenodo.23138782). The record includes the PDF, a README, the LaTeX/supplementary archive, and the verification script `verify_rotor_model.py`.
:::

## The one-paragraph version

Over a finite field $\mathbb{F}_p$ with $p \equiv 3 \pmod 4$, you cannot write four-dimensional Lorentzian spacetime as $2\times 2$ matrices over $\mathbb{F}_p$ itself. But if you let the entries live in the quadratic extension $\mathbb{K} = \mathbb{F}_p[i] \cong \mathbb{F}_{p^2}$, the **Hermitian** $2\times 2$ matrices form a four-dimensional $\mathbb{F}_p$-space whose determinant *is* the Lorentz quadratic form. Everything else in the paper follows from that: a compact description of $SO(1,3;\mathbb{F}_p)$, a formula that reads off a rotation's generator, a simple explanation of an earlier numerical glitch, and a small model of loop holonomies that you can enumerate exhaustively over $\mathbb{F}_7$.

## Why bother with finite fields?

Discrete and finitist approaches to spacetime want geometry that is exact: no limits, no transcendental functions, no floating-point drift. Wildberger's rational trigonometry and Universal Hyperbolic Geometry replace angles and exponentials with rational operations, and his recent "anchor and cap" construction replaces $\exp$ for $SL(2)$ by a rational map. Working over $\mathbb{F}_p$ pushes that idea to its limit: every statement is a finite, checkable fact.

The cost is that finite fields have no ordering. "Positive" has no meaning. So this is not a claim about physical probabilities. It is an algebraic toy model in which structures from the continuum, such as indefinite inner products, a $\mathbb{Z}_2$ "ghost parity" grading, and holonomies, can be tested exhaustively. The ghost-parity idea is borrowed from recent work of Bateman and Turok on higher-derivative quantum field theory, where a hidden discrete symmetry on a Krein space keeps probabilities well-defined. The finite-field version here is an analogy, not a derivation of their result.

## The obstruction, and why it is not one

An earlier paper in this series proved a **Witt-index obstruction**. For $p \equiv 3 \pmod 4$ we have $-1$ a non-square, so the form $x_0^2 - x_1^2 - x_2^2 - x_3^2$ has Witt index 1 (elliptic type). The determinant on $M_2(\mathbb{F}_p)$ has Witt index 2 (split). No linear isometry can connect them, so that paper built a $4\times 4$ Clifford-algebra engine instead.

The obstruction only forbids embeddings into $M_2(\mathbb{F}_p)$. Extend scalars to $\mathbb{K}$ and the picture changes:

$$
X(x) = \begin{pmatrix} x_0 + x_3 & x_1 - i x_2 \\ x_1 + i x_2 & x_0 - x_3 \end{pmatrix},
\qquad \det X(x) = x_0^2 - x_1^2 - x_2^2 - x_3^2 .
$$

The matrices $X(x)$ with $x \in \mathbb{F}_p^4$ are exactly the Hermitian ones, and the determinant is $\mathbb{F}_p$-valued on them. As a quadratic space this is a hyperbolic plane plus an anisotropic plane, so it has Witt index 1, exactly matching the Lorentz form. Nothing is "bypassed" in the sense of breaking Witt's theorem. We are simply looking at a different quadratic space than the one the theorem excluded. (The earlier paper even remarked that extension of scalars was not ruled out. This paper carries it out.)

## Rotors: all of $SO(1,3;\mathbb{F}_p)$ from eight numbers

Write a general element of $M_2(\mathbb{K})$ as $G = xI + \mathbf{z}\cdot\boldsymbol{\sigma}$ with $x = a + ip$ and $\mathbf{z} = \boldsymbol{\kappa} + i\boldsymbol{\rho}$. Its determinant is $\mathbb{F}_p$-valued exactly when

$$
ap = \boldsymbol{\kappa}\cdot\boldsymbol{\rho}.
$$

That is a quadric in $\mathbb{P}^7(\mathbb{F}_p)$ (a "rotor"). Each non-null rotor acts on Hermitian matrices by $X \mapsto G X G^\dagger / N$, where $N = a^2 - p^2 - \boldsymbol{\kappa}^2 + \boldsymbol{\rho}^2$, and this gives a group isomorphism with $SO(1,3;\mathbb{F}_p)$. The Lorentz matrix has entries that are quadratic polynomials in the eight rotor coordinates divided by the single number $N$. This isomorphism comes from the companion paper on rational rotations of four-dimensional relativistic space; the present paper builds on it.

Two counts follow, and both were checked by brute force over $\mathbb{F}_7$:

- $|SO(1,3;\mathbb{F}_7)| = 7^2(7^4 - 1) = 117{,}600$,
- exactly $p^4 + p^2 = 2{,}450$ of them have trace zero. These are the half-turns, which have no anchor (see below).

## Reading off a generator: the anchor formula

The Lie algebra $\mathfrak{so}(1,3)$ is six-dimensional and can be encoded as a vector $\boldsymbol{\zeta} \in \mathbb{K}^3$, the **anchor**. For any Lorentz transformation $\Lambda$ with $\operatorname{tr}\Lambda \neq 0$,

$$
A(\Lambda) = \frac{2\,\Lambda_{\mathrm{as}}}{\operatorname{tr}\Lambda},
\qquad
\Lambda_{\mathrm{as}} = \tfrac12\left(\Lambda - \eta\Lambda^{T}\eta\right).
$$

No matrix inversion, no logarithm, and it works for rotations with two invariant planes, which was the sticking point for the earlier $4\times 4$ approach. A fair caveat: this is a *chart statement*. The anchor is the unique parameter for which $\Lambda$ is a "cap" of it, not the generator of a one-parameter subgroup through $\Lambda$. If your element is already given as a $2\times 2$ spinor $xI + \mathbf{z}\cdot\boldsymbol{\sigma}$, the anchor is just $\mathbf{z}/x$ and the $4\times 4$ formula adds nothing. The formula earns its keep when you are working in the vector representation.

The caps themselves have a small catch worth knowing. An anchor yields an $\mathbb{F}_p$-rational Lorentz transformation only when $m^2 = (1-\alpha)^2 + \beta^2$ is a non-zero square in $\mathbb{F}_p$, where $\zeta\cdot\zeta = \alpha + i\beta$. Enumerating all $7^6 = 117{,}649$ anchors over $\mathbb{F}_7$:

| Anchors with … | Count |
|---|---|
| rational cap ($m^2$ a non-zero square) | 57,575 |
| no rational cap ($m^2$ a non-square) | 57,624 |
| $m^2 = 0$ (the locus $\zeta\cdot\zeta = 1$) | 2,450 |

The first number is exactly $\tfrac12(p^6 - p^4 - 2p^2)$, half the 115,150 rotations with non-zero trace (each anchor gives two caps, $\pm$). So roughly half of all anchors have no rational cap. That is a genuine finite-field phenomenon.

## The "anchor doubling" law and the 11% mystery

A previous preprint in the series tried to compose $(1+2)$-dimensional edge anchors using Wildberger's anchor product, and found that the predicted holonomy trace matched the real $2\times 2$ computation only about 11% of the time. The explanation turns out to be small and structural.

The spinor Cayley map $\mathrm{ca}(\zeta) = (I + \zeta\cdot\sigma)(I - \zeta\cdot\sigma)^{-1}$ lives in $SL(2,\mathbb{K})$. The Lorentz rotation it induces by conjugation has anchor

$$
\boldsymbol{\zeta}' = \frac{2\boldsymbol{\zeta}}{1 + \boldsymbol{\zeta}\cdot\boldsymbol{\zeta}},
$$

so the spin chart and the vector chart are related by a nonlinear *doubling*, the same phenomenon as angle-doubling in $SU(2)\to SO(3)$. A neat corollary: $1 - \boldsymbol{\zeta}'\cdot\boldsymbol{\zeta}' = \big((1-\boldsymbol{\zeta}\cdot\boldsymbol{\zeta})/(1+\boldsymbol{\zeta}\cdot\boldsymbol{\zeta})\big)^2$ is a perfect square, so doubled anchors automatically satisfy the descent condition above.

There is a second ingredient. $SL(2) \to SO$ is two-to-one, so the half-trace $T$ is only defined up to sign. The sign-free quantity is $T^2$, and in $(1+2)$ dimensions it satisfies $T^2 = 1 - p$, where $p$ is the half spread of the rotation. (In $(1+3)$ dimensions the analogues are $w^2 = 1 - \pi$ and $N_{\mathbb{K}/\mathbb{F}_p}(w) = 1 - s$.)

The check is to compose two spinor caps, take the trace of the product, square it, and compare with $1 - p$ computed from the composed doubled anchors. Over primes 7, 11, 13 and 17, every one of about 1.7 million valid test pairs agrees exactly. For contrast, the older, wrong target $(1-2p)^2$ agrees only 70%, 38%, 35% and 24% of the time at those primes: accidental agreement that shrinks roughly like $1/p$.

An honest note on what this shows: it demonstrates that the corrected comparison is exact and that the original failure was of this type. It does not recover the original 11% figure, which came from a different script, so "explains the origin" is the right wording rather than "reproduces the anomaly."

## Edge transports and holonomy

To get a connection on a tetrahedron's vertices, take two non-null points $u, v$ and form the reflection-pair product

$$
X(u)X(v)^{\vee} = B(u,v)\,I + \mathbf{z}_{uv}\cdot\boldsymbol{\sigma}.
$$

The edge anchor is $\boldsymbol{\zeta}_{uv} = \mathbf{z}_{uv}/B(u,v)$, which is unchanged if you rescale $u$ or $v$, and the transport is its spinor cap:

$$
U_{uv} = (1 - 2s_{uv})\,I + \frac{2B(u,v)}{Q(u)Q(v)}\,\mathbf{z}_{uv}\cdot\boldsymbol{\sigma}
= \frac{\big(X(u)X(v)^\vee\big)^2}{Q(u)Q(v)},
\qquad s_{uv} = 1 - \frac{B(u,v)^2}{Q(u)Q(v)}.
$$

Note the square. The *unsquared* reflection pairs telescope: multiplying them around a closed loop gives a central scalar, which acts as the identity rotation, so the connection is flat. Squaring is a modeling choice that makes the holonomy non-trivial. The paper says so plainly and calls it an ansatz. It rotates by twice the reflection-pair angle and is not a transport that carries $[u]$ to $[v]$, so it should not be read as a derivation of curvature.

Over $\mathbb{F}_7$ there are 350 non-null projective points and 68,600 admissible ordered edges. Enumerating every ordered admissible 4-cycle with the first vertex fixed to one representative of each square class (the isometry group acts transitively on each class, an assumption the paper states explicitly), each class gives 4,597,908 cycles. Of these, 9,492 have holonomy $+I$ and 1,680 have $-I$, so **99.757% have non-trivial holonomy**. A random cross-check (42,972 cycles) gives 0.223% trivial, consistent with the exhaustive 0.243%, and the trace of the holonomy is invariant under cyclic shifts and reversal in every sampled case.

The 0.243% may consist of degenerate (coplanar) vertex configurations. In a random sample of 9,097 non-coplanar cycles, none had trivial holonomy. That is evidence, not proof; an exhaustive non-coplanar count is left to future work.

## Ghost parity and the quadrume

The $\mathbb{Z}_2$ grading comes from vertex quadrances: $\varepsilon(\gamma) = \prod_i \chi_p(Q(v_i))$, where $\chi_p$ is the Legendre symbol. The paper realizes this as the spinor-norm character of the open reflection chain $R_\gamma = X_1 X_2^\vee X_3 X_4^\vee$, whose determinant is $\prod Q(v_i)$, and ties it to the projective quadrume $\mathcal{V} = \det G_4 / \prod Q(v_i)$ by $\chi_p(\mathcal{V}) = -\varepsilon$, which comes from $\det\eta = -1$ and $\chi_p(-1) = -1$.

Two honest remarks. First, this is a consistency identity, close to a definition. $\det R_\gamma = \prod Q(v_i)$ by construction. Second, the grading enters the dynamics only through the mutation rule, which only connects configurations of the same parity. So "ghost sectors decouple" holds by construction here. In Bateman and Turok's setting, the analogous statement is a nontrivial theorem about an interacting theory. This model does not reproduce that.

## Unitarity without positivity

The dynamics are a Cayley transform $U = (I - iH)(I + iH)^{-1}$ of a symmetric $\mathbb{F}_p$-valued Hamiltonian $H = -A + gW$. Because $H$ has entries in $\mathbb{F}_p$, $U^\dagger U = I$ exactly, and if $H$ commutes with the parity grading $\mathcal{G}$, so does $U$. That gives $U^\dagger \mathcal{G} U = \mathcal{G}$ and conserved field-norm weights within each parity sector. These are exact identities in a field with no notion of probability, and they follow quickly from the block structure. The paper defers the actual configuration space, the matrix $A$ and the spectra of $H$ to companion work. Nothing in this paper computes physical dynamics.

## What was actually verified

All numerical claims are produced by a single script, `verify_rotor_model.py`, run under Python 3.13 with NumPy 2.1.3. It uses exact finite-field arithmetic, with no floating-point tolerances in the algebra. Its four parts:

1. the $(1+2)$ composition test with the corrected target $T^2 = 1 - p$,
2. the $\mathbb{F}_7$ holonomy enumeration, the non-coplanar sample, and the $D_4$ check,
3. the exhaustive anchor census and the group-order counts,
4. a test that builds $\Lambda_R$ from the actual sandwich $X \mapsto GXG^\dagger/N$ and checks $\Lambda^T\eta\Lambda = \eta$, $\det\Lambda = 1$, $\operatorname{tr}\Lambda = 4(a^2+p^2)/N$, and a characteristic-polynomial coefficient, over 2,499 random rotors.

Some of the paper's identities (for example $\chi_p(\mathcal{V}) = -\varepsilon$) are tautological once the definitions are fixed, so they are not part of the verification.

## What this does and does not do

**It does:** give a compact, exact $2\times 2$ description of the finite-field Lorentz group; explain a specific numerical anomaly as a chart mismatch plus a double-cover sign; provide closed-form edge transports and an exhaustively checked holonomy statistic; and ship code that reproduces every figure in the paper.

**It does not:** derive curvature from the holonomy, justify the squared transport beyond calling it an ansatz, reproduce Bateman and Turok's positivity theorem, or touch continuum limits. The paper lists the quadrume–deficit relation and the large-$p$ limit toward Regge calculus as open problems.

## Try it

Download the record from [doi:10.5281/zenodo.23138782](https://doi.org/10.5281/zenodo.23138782) and run:

```bash
pip install numpy
python verify_rotor_model.py
```

You should see 100.00% agreement in Part 1 at every prime, 99.757% non-trivial holonomy in Part 2, and the counts 57,575 / 57,624 / 2,450 in Part 3. The random-sample figures depend on the NumPy generator, so exact digits in those lines can shift across major NumPy versions. Substitute your own prime in the small functions if you want to explore, but note that the paper's counts are specific to $p = 7$ (and $p = 7, 11, 13, 17$ for the $(1+2)$ test).

## References

- J. D. Coady, *A Projective 2×2 Rotor Architecture for (1+3)D Discrete Spacetime*, Zenodo (2026), [doi:10.5281/zenodo.23138782](https://doi.org/10.5281/zenodo.23138782).
- J. D. Coady, *Rational Rotations of Four Dimensional Relativistic Space: Projective Rotors, Anchors and Caps over a General Field* (preprint, Oct 2026).
- J. D. Coady, *A Unified Finitist Indefinite-Hermitian Framework for 3+1D Spacetime* (preprint, Sept 2026).
- S. Bateman and N. Turok, *Escape from Ostrogradsky via Hidden Ghost Parity*, arXiv:2607.00096.
- N. J. Wildberger, *The Anchor to Cap Map Replaces Exp for SL(2)* and *Rational Rotations of Three Dimensional Relativistic Space* (draft manuscripts); *Universal Hyperbolic Geometry I: Trigonometry*, Geom. Dedicata 163 (2013).

*Code and data were developed with AI assistance (Claude, Anthropic), as disclosed in the paper.*
