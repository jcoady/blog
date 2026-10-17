"""
Calculation 20 (Part 5): the rotor-to-link map for polynomial solutions of the free field equation.

  Input:  a polynomial solution zeta = E + iB of  d_t zeta = -i curl zeta,  div zeta = 0  (zeta in K^3, K = Q(i); E, B have Q coefficients)
  (a) vector potential in temporal gauge, exactly:   A(t,x) = int_0^1 s B(0,sx) x x ds  -  int_0^t E(tau,x) dtau ,   curl A = B ,  d_t A = -E
  (b) Cayley links   a_e = -(e/2) * (integral of A along the edge)   (exact rational numbers)
  (c) plaquettes (+) of the four links satisfy the six-face cube identity EXACTLY (realizable by construction)
  (d) FLUX-FIRST regularisation  a_f = -(e/2) Phi_f  (Phi_f = exact flux of B): the additive cube identity holds, the (+) identity fails at O(e^3)
  (e) gauge equivariance: A -> A + grad(lambda) changes the (+)-plaquettes at O(e^3) for the linear regularisation
  (f) exhaustive check over F_p, p = 3 (mod 4) only (the standing hypothesis -1 not a square): there is no non-trivial homomorphism
      (F_p, +) -> (P^1(F_p), (+));  for p = 1 (mod 4) the parameter 1 + a^2 can vanish and the group is a different one (not treated)
Needs sympy.
Tested with Python 3.12.3, SymPy 1.14.0, NumPy 2.4.4, SciPy 1.17.1 (one sandbox environment, not a lockfile-built clean environment).
"""
import itertools, random
from fractions import Fraction as Fr
import sympy as sp

t, x, y, z, s, tau = sp.symbols('t x y z s tau'); X = (x, y, z); I = sp.I
def curl(v): return [sp.diff(v[2], y) - sp.diff(v[1], z), sp.diff(v[0], z) - sp.diff(v[2], x), sp.diff(v[1], x) - sp.diff(v[0], y)]
def div(v): return sp.diff(v[0], x) + sp.diff(v[1], y) + sp.diff(v[2], z)
def cross(a, b): return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]

# ---------------- polynomial solutions of d_t zeta = -i curl zeta, div zeta = 0 (total degree <= maxdeg) ----------------
def solution_space(maxdeg, c=-I):
    mons = [t**a * x**b * y**cc * z**dd for a in range(maxdeg + 1) for b in range(maxdeg + 1) for cc in range(maxdeg + 1)
            for dd in range(maxdeg + 1) if a + b + cc + dd <= maxdeg]
    syms = [sp.Symbol(f'q{a}_{k}') for a in range(3) for k in range(len(mons))]
    zeta = [sum(syms[a * len(mons) + k] * mons[k] for k in range(len(mons))) for a in range(3)]
    eqs = [sp.diff(zeta[a], t) - c * curl(zeta)[a] for a in range(3)] + [div(zeta)]
    rows = []
    for e in eqs:
        e = sp.expand(e)
        if e != 0: rows += list(sp.Poly(e, t, x, y, z).coeffs())
    M, _ = sp.linear_eq_to_matrix(rows, syms)
    return [[sp.expand(sum(v[a * len(mons) + k] * mons[k] for k in range(len(mons)))) for a in range(3)] for v in M.nullspace()]

def split_EB(zeta):
    """zeta = E + iB with E, B polynomials with rational coefficients."""
    E, B = [], []
    for comp in zeta:
        P = sp.Poly(sp.expand(comp), t, x, y, z); Ep = 0; Bp = 0
        for mon, co in P.terms():
            m = t**mon[0] * x**mon[1] * y**mon[2] * z**mon[3]
            Ep += sp.re(co) * m; Bp += sp.im(co) * m
        E.append(sp.expand(Ep)); B.append(sp.expand(Bp))
    return E, B

def random_solution(sols, seed, n=5):
    rng = random.Random(seed); zeta = [0, 0, 0]
    for sol in rng.sample(sols, n):
        co = rng.randint(-2, 2) + I * rng.randint(-2, 2)
        zeta = [sp.expand(zeta[a] + co * sol[a]) for a in range(3)]
    return zeta

def potential(E, B):
    """A(t,x) = int_0^1 s B(0,sx) x x ds - int_0^t E(tau,x) dtau."""
    B0 = [bb.subs({t: 0, x: s * x, y: s * y, z: s * z}, simultaneous=True) for bb in B]
    A0 = [sp.integrate(sp.expand(s * c_), (s, 0, 1)) for c_ in cross(B0, [x, y, z])]
    return [sp.expand(A0[a] - sp.integrate(E[a].subs(t, tau), (tau, 0, t))) for a in range(3)]

# ---------------- Cayley arithmetic over Q ----------------
def kmul(u, v): return (u[0] * v[0] - u[1] * v[1], u[0] * v[1] + u[1] * v[0])
class Ring:
    def __init__(s_, mk, p=None): s_.mk, s_.p, s_.one, s_.zero = mk, p, mk(1), mk(0)
    def phase(s_, a):
        if a is None: return (-s_.one, s_.zero)
        dd = s_.one + a * a
        if dd == s_.zero: raise ZeroDivisionError("1 + a^2 = 0: not an admissible Cayley parameter (the hypothesis that -1 is not a square fails)")
        return ((s_.one - a * a) / dd, s_.mk(-2) * a / dd)
    def a_of(s_, u): return None if u[0] == -s_.one else -u[1] / (s_.one + u[0])
    def oplus(s_, a, b): return s_.a_of(kmul(s_.phase(a), s_.phase(b)))
    def neg(s_, a): return None if a is None else -a
    def ominus(s_, a, b): return s_.oplus(a, s_.neg(b))
RQ = Ring(Fr)
def make_Fp(p):
    class FpP:
        P = p
        def __init__(s_, v): s_.v = int(v) % FpP.P
        def __add__(s_, o): return FpP(s_.v + (o.v if hasattr(o, 'v') else int(o)))
        __radd__ = __add__
        def __sub__(s_, o): return FpP(s_.v - (o.v if hasattr(o, 'v') else int(o)))
        def __neg__(s_): return FpP(-s_.v)
        def __mul__(s_, o): return FpP(s_.v * (o.v if hasattr(o, 'v') else int(o)))
        __rmul__ = __mul__
        def __truediv__(s_, o): return FpP(s_.v * pow(o.v if hasattr(o, 'v') else int(o), -1, FpP.P))
        def __eq__(s_, o): return o is not None and s_.v == (o.v if hasattr(o, 'v') else int(o)) % FpP.P
        def __hash__(s_): return hash(s_.v)
    return FpP
def eq(a, b): return (a is None and b is None) or (a is not None and b is not None and a == b)

# ---------------- lattice plaquettes ----------------
def Pxy(R, a, i, j, k): return R.oplus(R.oplus(R.oplus(a[(i, j, k, 0)], a[(i + 1, j, k, 1)]), R.neg(a[(i, j + 1, k, 0)])), R.neg(a[(i, j, k, 1)]))
def Pyz(R, a, i, j, k): return R.oplus(R.oplus(R.oplus(a[(i, j, k, 1)], a[(i, j + 1, k, 2)]), R.neg(a[(i, j, k + 1, 1)])), R.neg(a[(i, j, k, 2)]))
def Pzx(R, a, i, j, k): return R.oplus(R.oplus(R.oplus(a[(i, j, k, 2)], a[(i, j, k + 1, 0)]), R.neg(a[(i + 1, j, k, 2)])), R.neg(a[(i, j, k, 0)]))
def face_ranges(N):
    return {'xy': [(i, j, k) for i in range(N) for j in range(N) for k in range(N + 1)],
            'yz': [(i, j, k) for i in range(N + 1) for j in range(N) for k in range(N)],
            'zx': [(i, j, k) for i in range(N) for j in range(N + 1) for k in range(N)]}
def faces_from_links(R, a, N):
    fr = face_ranges(N); f = {'xy': Pxy, 'yz': Pyz, 'zx': Pzx}
    return {n: {p: f[n](R, a, *p) for p in fr[n]} for n in fr}
def cube_residual(R, F, i, j, k):
    d_ = lambda f, p, q: R.ominus(F[f][p], F[f][q])
    return R.oplus(R.oplus(d_('xy', (i, j, k + 1), (i, j, k)), d_('yz', (i + 1, j, k), (i, j, k))), d_('zx', (i, j + 1, k), (i, j, k)))

# ---------------- exact line and surface integrals of polynomials on the lattice ----------------
def tofr(expr):
    r = sp.Rational(sp.expand(expr)); return Fr(int(r.p), int(r.q))

def edge_integrals(A, t0, N):
    """a-values:  int along the edge of A, for edges (i,j,k,m), m = 0,1,2, spacing 1."""
    At = [sp.expand(c_.subs(t, t0)) for c_ in A]; out = {}
    anti = [sp.integrate(At[0], x), sp.integrate(At[1], y), sp.integrate(At[2], z)]
    for i, j, k in itertools.product(range(N + 1), repeat=3):
        out[(i, j, k, 0)] = tofr(anti[0].subs({x: i + 1, y: j, z: k}) - anti[0].subs({x: i, y: j, z: k}))
        out[(i, j, k, 1)] = tofr(anti[1].subs({x: i, y: j + 1, z: k}) - anti[1].subs({x: i, y: j, z: k}))
        out[(i, j, k, 2)] = tofr(anti[2].subs({x: i, y: j, z: k + 1}) - anti[2].subs({x: i, y: j, z: k}))
    return out

def face_fluxes(B, t0, N):
    Bt = [sp.expand(c_.subs(t, t0)) for c_ in B]; fr = face_ranges(N); out = {'xy': {}, 'yz': {}, 'zx': {}}
    Iz = sp.integrate(sp.integrate(Bt[2], x), y); Ix = sp.integrate(sp.integrate(Bt[0], y), z); Iy = sp.integrate(sp.integrate(Bt[1], z), x)
    def box(F2, v1, v2, a1, a2, fixed):
        tot = 0
        for (p, q, sg) in ((a1 + 1, a2 + 1, 1), (a1, a2, 1), (a1 + 1, a2, -1), (a1, a2 + 1, -1)):
            tot += sg * F2.subs({v1: p, v2: q, **fixed})
        return tofr(tot)
    for (i, j, k) in fr['xy']: out['xy'][(i, j, k)] = box(Iz, x, y, i, j, {z: k})
    for (i, j, k) in fr['yz']: out['yz'][(i, j, k)] = box(Ix, y, z, j, k, {x: i})
    for (i, j, k) in fr['zx']: out['zx'][(i, j, k)] = box(Iy, z, x, k, i, {y: j})
    return out

def scaled(links, factor): return {k: factor * v for k, v in links.items()}
def maxabs_resid(R, F, N):
    m = Fr(0)
    for c_ in itertools.product(range(N), repeat=3):
        r = cube_residual(R, F, *c_)
        if r is None: return None
        m = max(m, abs(r))
    return m

def main():
    print("building polynomial Maxwell solutions (total degree <= 3), exactly over Q(i) ...")
    sols = solution_space(3); print("   dimension over Q(i):", len(sols))
    zeta = random_solution(sols, seed=11); E, B = split_EB(zeta)
    maxwell = all(sp.expand(sp.diff(E[a], t) - curl(B)[a]) == 0 and sp.expand(sp.diff(B[a], t) + curl(E)[a]) == 0 for a in range(3)) \
        and sp.expand(div(E)) == 0 and sp.expand(div(B)) == 0
    print("   E, B (rational coefficients) satisfy d_t E = curl B, d_t B = -curl E, div E = div B = 0:", maxwell)
    A = potential(E, B)
    ok_curl = all(sp.expand(curl(A)[a] - B[a]) == 0 for a in range(3)); ok_dt = all(sp.expand(sp.diff(A[a], t) + E[a]) == 0 for a in range(3))
    print("(a) exact potential in temporal gauge:  curl A = B:", ok_curl, ";  d_t A = -E:", ok_dt)

    N = 3; t0 = Fr(1, 2)
    lin = edge_integrals(A, sp.Rational(1, 2), N); flux = face_fluxes(B, sp.Rational(1, 2), N)
    print("(b)-(d) lattice N = 3, spacing 1, t0 = 1/2.  Columns: e, then max over cubes of the cube-identity residual")
    print("        e      links-first (+)-residual   flux-first (+)-residual   additive residual   max|a_P(links) - (-e Phi/2)|")
    prev = None
    for e in (Fr(1, 128), Fr(1, 256), Fr(1, 512), Fr(1, 1024)):
        links = scaled(lin, -e / 2)                                                  # a_e = -(e/2) * int_e A
        F_links = faces_from_links(RQ, links, N)
        F_flux = {n: {p: -e / 2 * v for p, v in flux[n].items()} for n in flux}      # a_f = -(e/2) Phi_f
        r_links = maxabs_resid(RQ, F_links, N); r_flux = maxabs_resid(RQ, F_flux, N)
        add = max(abs((flux['xy'][(i, j, k + 1)] - flux['xy'][(i, j, k)]) + (flux['yz'][(i + 1, j, k)] - flux['yz'][(i, j, k)]) +
                      (flux['zx'][(i, j + 1, k)] - flux['zx'][(i, j, k)])) for (i, j, k) in itertools.product(range(N), repeat=3))
        dif = max(abs(F_links[n][p] - F_flux[n][p]) for n in F_links for p in F_links[n])
        ratio = "" if prev is None else f"   (ratio to previous {float(prev / dif):.2f})"
        prev = dif
        print(f"        {str(e):>5}   {str(r_links):>22}   {float(r_flux):>22.3e}   {str(add):>17}   {float(dif):.3e}{ratio}")

    print("(e) gauge change A -> A + grad(lambda), lambda = x^2 y + 2 y z + x z^2 (time independent): plaquettes of the linear regularisation")
    lam = x**2 * y + 2 * y * z + x * z**2
    A2 = [sp.expand(A[0] + sp.diff(lam, x)), sp.expand(A[1] + sp.diff(lam, y)), sp.expand(A[2] + sp.diff(lam, z))]
    lin2 = edge_integrals(A2, sp.Rational(1, 2), N); prev = None
    for e in (Fr(1, 128), Fr(1, 256), Fr(1, 512)):
        F1 = faces_from_links(RQ, scaled(lin, -e / 2), N); F2 = faces_from_links(RQ, scaled(lin2, -e / 2), N)
        dif = max(abs(F1[n][p] - F2[n][p]) for n in F1 for p in F1[n])
        ratio = "" if prev is None else f"   (ratio to previous {float(prev / dif):.2f})"; prev = dif
        print(f"        e = {str(e):>5}: max change of a plaquette parameter = {float(dif):.3e}{ratio}")
    fl2 = face_fluxes(sp.Matrix(curl(A2)).T.tolist()[0], sp.Rational(1, 2), N)
    print("        the additive fluxes are unchanged by the gauge change:", all(fl2[n][p] == flux[n][p] for n in flux for p in flux[n]))

    print("(f) homomorphisms (F_p, +) -> (P^1(F_p), (+)): number of g in P^1(F_p) with p-fold g = 0  (only g = 0 means no non-trivial homomorphism)")
    for p in (3, 7, 11, 19, 23):
        assert p % 4 == 3, 'the Cayley group of order p+1 needs p = 3 (mod 4)'
        R = Ring(make_Fp(p), p); pts = [None] + [R.mk(v) for v in range(p)]; cnt = 0
        for g in pts:
            r = R.zero
            for _ in range(p): r = R.oplus(r, g)
            cnt += eq(r, R.zero)
        print(f"        p = {p:2d}: solutions {cnt}   (|P^1| = {p + 1}, gcd(p, p+1) = 1)")

main()
