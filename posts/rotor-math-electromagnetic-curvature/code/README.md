# Code for "Is the Rotor Anchor the Electromagnetic Field Strength?"

Scripts that reproduce numbers quoted in the blog post
`posts/rotor-math-electromagnetic-curvature/`.

| File | What it checks | Needs | Runtime |
|:---|:---|:---|:---:|
| `cayley_u1_lattice.py` | U(1) lattice built from Cayley phases `(1 - i a)/(1 + i a)`. Exact Stokes, Bianchi and gauge invariance on a 4D torus over Q and F_p. Wilson vs Cayley F² action in the continuum regime and at strong coupling. | numpy, scipy | about 1 s |
| `cayley_su2_lattice.py` | The non-abelian version with the spin Cayley cap. Path-ordered Stokes for rectangles, cube Bianchi identity (the valid face orderings are found by search), and the scaling of the Wilson action versus the half-trace potential `(1 - w)^2`. | numpy | about 25 s |

The flux and duality checks (Calculations 8 and 9) and the Lorentz-force checks
(Calculations 1 to 7) are embedded as code blocks in the post itself.

## Running

Locally:

```
python cayley_u1_lattice.py
python cayley_su2_lattice.py
```

In Google Colab, in a cell (replace `main` with the branch or tag you want):

```
!wget -q https://raw.githubusercontent.com/jcoady/blog/main/posts/rotor-math-electromagnetic-curvature/code/cayley_u1_lattice.py
%run cayley_u1_lattice.py
```

Neither script takes command-line arguments, so they run unchanged in a notebook.

## Scope

These scripts check exact algebraic identities and compare two lattice actions.
They do not derive field equations, and the continuum comparisons are done over the
rationals and reals only. Over F_p there is no ordering, so only the exact identities
carry over.
