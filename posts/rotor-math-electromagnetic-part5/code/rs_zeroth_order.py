"""
Calculation 19 (Part 5): is a zeroth-order term excluded?

Equation with a possible zeroth-order term:   d_t zeta = c curl zeta + d zeta ,   div zeta = 0 .
Boost generator  K zeta = (x d_t + t d_x) zeta + mu e_x x zeta .       L zeta = d_t zeta - c curl zeta - d zeta .
  (1) the operator identity  [L, K] zeta = d_x zeta - c e_x x d_t zeta - c mu ( e_x div zeta - d_x zeta - e_x x curl zeta )
      and  div(K zeta) = d_t zeta_x + x d_t div zeta + t d_x div zeta - mu (curl zeta)_x   are verified symbolically for generic functions
  (2) reduction on solutions and the resulting conditions on (c, d, mu)
  (3) exact plane-wave check over Q(i):  boost closure holds if and only if d = 0
Needs sympy.
Tested with Python 3.12.3, SymPy 1.14.0, NumPy 2.4.4, SciPy 1.17.1 (one sandbox environment, not a lockfile-built clean environment).
"""
import sympy as sp
t, x, y, z = sp.symbols('t x y z'); I = sp.I
c, d, mu = sp.symbols('c d mu')
def curl(v): return [sp.diff(v[2], y) - sp.diff(v[1], z), sp.diff(v[0], z) - sp.diff(v[2], x), sp.diff(v[1], x) - sp.diff(v[0], y)]
def div(v): return sp.diff(v[0], x) + sp.diff(v[1], y) + sp.diff(v[2], z)
def cross_ex(v): return [0, -v[2], v[1]]                       # e_x x v
ex = [1, 0, 0]

# ---- (1) operator identities for generic functions ----
f = [sp.Function(f'f{a}')(t, x, y, z) for a in range(3)]
L = lambda v: [sp.diff(v[a], t) - c * curl(v)[a] - d * v[a] for a in range(3)]
K = lambda v: [x * sp.diff(v[a], t) + t * sp.diff(v[a], x) + mu * cross_ex(v)[a] for a in range(3)]
comm = [sp.expand(L(K(f))[a] - K(L(f))[a]) for a in range(3)]
dxf = [sp.diff(f[a], x) for a in range(3)]; dtf = [sp.diff(f[a], t) for a in range(3)]; cf = curl(f)
rhs = [dxf[a] - c * cross_ex(dtf)[a] - c * mu * (ex[a] * div(f) - dxf[a] - cross_ex(cf)[a]) for a in range(3)]
print("(1) [L,K] f = d_x f - c e_x x d_t f - c mu (e_x div f - d_x f - e_x x curl f), for generic f:", all(sp.expand(comm[a] - rhs[a]) == 0 for a in range(3)))
Kf = K(f)
divK = sp.expand(div(Kf)); rhs2 = sp.expand(dtf[0] + x * sp.diff(div(f), t) + t * sp.diff(div(f), x) - mu * cf[0])
print("    div(K f) = d_t f_x + x d_t div f + t d_x div f - mu (curl f)_x:", sp.expand(divK - rhs2) == 0)

# ---- (2) reduction on solutions: d_t zeta = c curl zeta + d zeta, div zeta = 0 ----
vx = sp.symbols('vx1:4'); vc = sp.symbols('vc1:4'); vz = sp.symbols('vz1:4')       # d_x zeta, curl zeta, zeta (independent vectors)
dt_sol = [c * vc[a] + d * vz[a] for a in range(3)]
comm_sol = [sp.expand(vx[a] - c * cross_ex(dt_sol)[a] - c * mu * (-vx[a] - cross_ex(vc)[a])) for a in range(3)]
cex = cross_ex(vc); czx = cross_ex(vz)
coef_vx = sp.expand(comm_sol[0].coeff(vx[0]))                                    # coefficient of d_x zeta (x-component)
print("(2) on solutions:  [L,K] zeta = (1 + c mu) d_x zeta - c (c - mu) e_x x curl zeta - c d e_x x zeta ;  div(K zeta) = (c - mu)(curl zeta)_x + d zeta_x")
print("    coefficients of d_x zeta, e_x x curl zeta, e_x x zeta:", sp.factor(coef_vx), ",", sp.factor(comm_sol[1].coeff(vc[2])*(-1)) , ",", sp.factor(comm_sol[1].coeff(vz[2])*(-1)))
conds = [1 + c * mu, c * (c - mu), c * d, c - mu, d]
print("    conditions:", conds, "->", sp.solve(conds, [c, d, mu], dict=True))

# ---- (3) exact plane-wave check over Q(i):  zeta = eps exp(i k.x + lam t) ----
kx, ky = 3, 4; kv = sp.Matrix([kx, ky, 0]); kn = 5                                     # |k| = 5, rational
def kcross(v): return sp.Matrix(kv).cross(sp.Matrix(v))
Kmat = sp.Matrix([[0, 0, ky], [0, 0, -kx], [-ky, kx, 0]])                               # k x
rows = []
for s in (1, -1):
    ns = (Kmat - s * kn * I * sp.eye(3)).nullspace()
    eps = ns[0]
    for cval in (-I, I):
        for dval in (0, sp.Rational(1, 2), I):
            lam = -kn * cval * s + dval                                                  # from lam eps = c i (k x eps) + d eps
            phase = sp.exp(I * (kx * x + ky * y) + lam * t)
            zeta = [eps[a] * phase for a in range(3)]
            eq_res = [sp.simplify((sp.diff(zeta[a], t) - cval * curl(zeta)[a] - dval * zeta[a]) / phase) for a in range(3)]
            assert all(r == 0 for r in eq_res) and sp.simplify(div(zeta) / phase) == 0
            mval = cval                                                                  # mu = c is the only candidate
            Kz = [x * sp.diff(zeta[a], t) + t * sp.diff(zeta[a], x) + mval * cross_ex(zeta)[a] for a in range(3)]
            res = [sp.simplify((sp.diff(Kz[a], t) - cval * curl(Kz)[a] - dval * Kz[a]) / phase) for a in range(3)]
            dres = sp.simplify(div(Kz) / phase)
            closed = all(sp.simplify(r) == 0 for r in res) and sp.simplify(dres) == 0
            rows.append((s, cval, dval, closed))
print("(3) plane waves (k = (3,4,0), |k| = 5), mu = c:  boost closure of {equation, div = 0}:")
for s, cval, dval, closed in rows:
    print(f"      helicity s = {s:+d}, c = {str(cval):>2}, d = {str(dval):>3}:  closed under the boost: {closed}")
