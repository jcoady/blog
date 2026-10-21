"""
Calculations 22 and 23 (Part 7): a minimal coupled gauge-matter lattice model in exact arithmetic.

MODEL (2+1 D periodic L x L lattice, Hamiltonian form, temporal gauge; the temporal link is the Lagrange multiplier of the Gauss constraint)
  variables:  Cayley link parameters a_mu(x) in P^1(F),  electric fields E_mu(x) in F,  complex scalar phi(x) in K = F[i],  momentum pi(x) in K
  H = sum_{x,mu} E^2/2  +  sum_P (1 - cos theta_P)  +  sum_x pi pi*  +  sum_{x,mu} |U_mu(x) phi(x+mu) - phi(x)|^2  +  m^2 sum_x |phi|^2
  gauge transformation:  phi -> g phi,  pi -> g^-1 pi,  U_mu(x) -> g(x) U_mu(x) g(x+mu)^-1,  E unchanged
  Gauss generator:  G(x) = (div E)(x) + rho(x),   rho = -2 Im(pi phi),   (div E)(x) = sum_mu [E_mu(x) - E_mu(x-mu)]
  link current:     J_mu(x) = dH_hop/d theta_mu(x) = 2 Im( U_mu(x) phi(x+mu) phi(x)* )
CONVENTIONS: phi, phi*, pi, pi* are independent canonical variables with {phi, pi} = 1, {phi*, pi*} = 1, {theta, E} = 1; physical states have
  phi* = conj(phi), pi* = conj(pi).  rho = i (pi phi - pi* phi*) = -2 Im(pi phi) generates phi -> e^{i alpha} phi, pi -> e^{-i alpha} pi;
  alpha_x (div E)_x summed over x generates theta_mu(x) -> theta_mu(x) + alpha_x - alpha_{x+mu}.
SCHEME (symmetric kick-drift-kick):  step(h) = kick(h/2) drift(h) kick(h/2).
  The kicks are exact flows of the potential terms of H.  The matter drift is the exact flow of pi pi*.  The Cayley drift of the links is the
  exact flow of the MODIFIED kinetic energy T_h(E) with T_h'(E) = (2/h) arctan(hE/2), i.e. T_h(E) = E^2/2 - h^2 E^4/48 + O(h^4);
  it is NOT the exact flow of E^2/2.
  kick:   E <- E - h dH_pot/d theta,  pi <- pi - h dH_pot/d phi          (H_pot = plaquette + hopping + mass; depends on positions only)
  drift:  phi <- phi + h pi*,   a <- a (+) ( -(h/2) E )                  (Cayley drift)
Checks over Q (one full step: exact rational numbers grow about 14-fold in size per step), F_7 and F_11 (five full steps); random states, including a = infinity:
  (a) G is conserved exactly by every sub-step, for any h        (b) exact discrete continuity  Delta rho = h div J  for the kick
  (c) the step commutes with gauge transformations               (d) step(-h) is the exact inverse
  (b') continuity over a FULL step:  rho(n+1) - rho(n) = (h/2) div J(first kick) + (h/2) div J(second kick)
  (e) the space of real quadratic Lorentz invariants of (E, B) is 2-dimensional, spanned by E^2 - B^2 and E.B   (sympy)
  (f) floating point: the implemented kick forces equal the gradients of the stated potential Hamiltonian (finite differences)
Needs sympy for (e).  Tested with Python 3.12.3, SymPy 1.14.0.
"""
import itertools
import random
from fractions import Fraction as Fr


# ------------------------------------------------------------------ fields ---
def make_Fp(p):
    class FpP:
        P = p
        def __init__(s, v): s.v = int(v) % FpP.P
        def __add__(s, o): return FpP(s.v + _v(o))
        __radd__ = __add__
        def __sub__(s, o): return FpP(s.v - _v(o))
        def __rsub__(s, o): return FpP(_v(o) - s.v)
        def __neg__(s): return FpP(-s.v)
        def __mul__(s, o): return FpP(s.v * _v(o))
        __rmul__ = __mul__
        def __truediv__(s, o): return FpP(s.v * pow(_v(o), -1, FpP.P))
        def __eq__(s, o): return o is not None and s.v == _v(o) % FpP.P
        def __hash__(s): return hash(s.v)
        def __repr__(s): return str(s.v)
    return FpP
def _v(o): return o.v if hasattr(o, "v") else int(o)

def kadd(u, v): return (u[0] + v[0], u[1] + v[1])
def ksub(u, v): return (u[0] - v[0], u[1] - v[1])
def kmul(u, v): return (u[0] * v[0] - u[1] * v[1], u[0] * v[1] + u[1] * v[0])
def kconj(u): return (u[0], -u[1])
def ksc(s, u): return (s * u[0], s * u[1])

class Ring:
    """P^1(F) with the Cayley law; a = None is infinity.  Requires -1 not a square (p = 3 mod 4)."""
    def __init__(s, mk, p=None):
        if p is not None: assert p % 4 == 3
        s.mk, s.p, s.one, s.zero = mk, p, mk(1), mk(0)
    def phase(s, a):
        if a is None: return (-s.one, s.zero)
        d = s.one + a * a
        if d == s.zero: raise ZeroDivisionError("1 + a^2 = 0")
        return ((s.one - a * a) / d, s.mk(-2) * a / d)
    def a_of(s, u): return None if u[0] == -s.one else -u[1] / (s.one + u[0])
    def oplus(s, a, b): return s.a_of(kmul(s.phase(a), s.phase(b)))
    def neg(s, a): return None if a is None else -a
    def eq(s, a, b): return (a is None and b is None) or (a is not None and b is not None and a == b)
RQ = Ring(Fr)
def RF(p): return Ring(make_Fp(p), p)


# ------------------------------------------------------------------ lattice ---
L = 3
SITES = [(i, j) for i in range(L) for j in range(L)]
def sh(x, mu, s=1): return ((x[0] + s * (mu == 0)) % L, (x[1] + s * (mu == 1)) % L)

class State:
    def __init__(s, a, E, phi, pi): s.a, s.E, s.phi, s.pi = a, E, phi, pi
    def copy(s): return State(dict(s.a), dict(s.E), dict(s.phi), dict(s.pi))
    def same(s, R, o):
        return all(R.eq(s.a[k], o.a[k]) and s.E[k] == o.E[k] for k in s.a) and all(s.phi[x] == o.phi[x] and s.pi[x] == o.pi[x] for x in s.phi)

def random_state(R, rng, with_infinity=True):
    if R is RQ:
        num = lambda: Fr(rng.randint(-5, 5), rng.randint(1, 4))
        par = lambda: None if (with_infinity and rng.random() < 0.05) else num()
    else:
        num = lambda: R.mk(rng.randrange(R.p))
        par = lambda: None if (with_infinity and rng.randrange(R.p + 1) == R.p) else R.mk(rng.randrange(R.p))
    a = {(x, mu): par() for x in SITES for mu in (0, 1)}; E = {(x, mu): num() for x in SITES for mu in (0, 1)}
    phi = {x: (num(), num()) for x in SITES}; pi = {x: (num(), num()) for x in SITES}
    return State({(x, mu): a[(x, mu)] for x in SITES for mu in (0, 1)}, E, phi, pi)

def links(R, st): return {k: R.phase(v) for k, v in st.a.items()}

def forces(R, st, m2):
    """Returns (dE_kick, dpi_kick, J) such that a kick of size h does E -= h*dE, pi -= h*dpi.   J_mu(x) = dH_hop/d theta."""
    U = links(R, st); two = R.mk(2)
    # plaquette: xi = sin(theta_P) = Im U_P
    xi = {}
    for (i, j) in SITES:
        UP = kmul(kmul(U[((i, j), 0)], U[(sh((i, j), 0), 1)]), kmul(kconj(U[(sh((i, j), 1), 0)]), kconj(U[((i, j), 1)])))
        xi[(i, j)] = UP[1]
    dE = {}
    for (i, j) in SITES:
        dE[((i, j), 0)] = xi[(i, j)] - xi[(i, (j - 1) % L)]                      # E_x -= h (xi(i,j) - xi(i,j-1))
        dE[((i, j), 1)] = -(xi[(i, j)] - xi[((i - 1) % L, j)])                   # E_y += h (xi(i,j) - xi(i-1,j))
    # hopping: J = 2 Im( U phi(x+mu) phi(x)* ),  dH/dphi_x = sum_mu [ 2 phi*_x - U_mu(x-mu) phi*_{x-mu} - U_mu(x)* phi*_{x+mu} ]
    J = {}
    for x in SITES:
        for mu in (0, 1):
            z = kmul(kmul(U[(x, mu)], st.phi[sh(x, mu)]), kconj(st.phi[x]))
            J[(x, mu)] = two * z[1]
            dE[(x, mu)] = dE[(x, mu)] + J[(x, mu)]
    dpi = {}
    for x in SITES:
        d = (R.zero, R.zero)
        for mu in (0, 1):
            d = kadd(d, ksc(two, kconj(st.phi[x])))
            d = ksub(d, kmul(U[(sh(x, mu, -1), mu)], kconj(st.phi[sh(x, mu, -1)])))
            d = ksub(d, kmul(kconj(U[(x, mu)]), kconj(st.phi[sh(x, mu)])))
        dpi[x] = kadd(d, ksc(m2, kconj(st.phi[x])))                               # + m^2 phi*
    return dE, dpi, J

def kick(R, st, h, m2):
    dE, dpi, J = forces(R, st, m2); out = st.copy()
    for k in out.E: out.E[k] = st.E[k] - h * dE[k]
    for x in out.pi: out.pi[x] = ksub(st.pi[x], ksc(h, dpi[x]))
    return out

def drift(R, st, h):
    out = st.copy()
    for x in out.phi: out.phi[x] = kadd(st.phi[x], ksc(h, kconj(st.pi[x])))
    half = h / R.mk(2)
    for k in out.a: out.a[k] = R.oplus(st.a[k], -(half * st.E[k]))
    return out

def step(R, st, h, m2):
    h2 = h / R.mk(2)
    return kick(R, drift(R, kick(R, st, h2, m2), h), h2, m2)

def gauss(R, st):
    G = {}
    for x in SITES:
        div = R.zero
        for mu in (0, 1): div = div + st.E[(x, mu)] - st.E[(sh(x, mu, -1), mu)]
        rho = -R.mk(2) * kmul(st.pi[x], st.phi[x])[1]
        G[x] = div + rho
    return G

def gauge_transform(R, st, gam):
    g = {x: R.phase(gam[x]) for x in SITES}; out = st.copy()
    for (x, mu) in out.a: out.a[(x, mu)] = R.oplus(R.oplus(gam[x], st.a[(x, mu)]), R.neg(gam[sh(x, mu)]))
    for x in out.phi:
        out.phi[x] = kmul(g[x], st.phi[x]); out.pi[x] = kmul(kconj(g[x]), st.pi[x])
    return out


def invariants_test():
    import sympy as sp
    E = sp.symbols('E1:4'); B = sp.symbols('B1:4'); v = list(E) + list(B)
    S = sp.Matrix(6, 6, lambda i, j: sp.Symbol(f's{min(i, j)}{max(i, j)}'))
    unk = sorted(S.free_symbols, key=str)
    def gen(rot, boost):
        # delta E = rot x E - boost x B ,  delta B = rot x B + boost x E   (rotation by a complex angle: delta zeta = (rot + i boost) x zeta ... i.e. i (boost x zeta))
        Ev = sp.Matrix(E); Bv = sp.Matrix(B); r = sp.Matrix(rot); b = sp.Matrix(boost)
        dE = r.cross(Ev) - b.cross(Bv); dB = r.cross(Bv) + b.cross(Ev)
        return sp.Matrix(list(dE) + list(dB))
    eqs = []
    unit = [sp.Matrix([1, 0, 0]), sp.Matrix([0, 1, 0]), sp.Matrix([0, 0, 1])]
    for k in range(3):
        for (r, b) in ((unit[k], sp.zeros(3, 1)), (sp.zeros(3, 1), unit[k])):
            dv = gen(r, b); q = sp.Matrix(v).T * S * dv * 2
            eqs += list(sp.Poly(sp.expand(q[0]), *v).coeffs())
    M, _ = sp.linear_eq_to_matrix(eqs, unk); ns = M.nullspace()
    forms = []
    for n in ns:
        sub = {unk[i]: n[i] for i in range(len(unk))}; Sm = S.subs(sub)
        forms.append(sp.expand((sp.Matrix(v).T * Sm * sp.Matrix(v))[0]))
    return len(ns), forms


def forces_vs_hamiltonian(seed=3, m2=1.0, delta=1e-6):
    """Floating point: compare the implemented kick forces with finite-difference gradients of
    H_pot = sum_P (1 - cos theta_P) + sum |U phi(x+mu) - phi(x)|^2 + m2 sum |phi|^2   (theta = -2 arctan a)."""
    import math
    Rf = Ring(float); rng = random.Random(seed)
    th = {(x, mu): rng.uniform(-1.2, 1.2) for x in SITES for mu in (0, 1)}
    phi = {x: complex(rng.uniform(-1, 1), rng.uniform(-1, 1)) for x in SITES}
    def Hpot(th, phi):
        U = {k: complex(math.cos(v), math.sin(v)) for k, v in th.items()}; H = 0.0
        for (i, j) in SITES:
            UP = U[((i, j), 0)] * U[(sh((i, j), 0), 1)] * U[(sh((i, j), 1), 0)].conjugate() * U[((i, j), 1)].conjugate(); H += 1 - UP.real
        for x in SITES:
            for mu in (0, 1): H += abs(U[(x, mu)] * phi[sh(x, mu)] - phi[x]) ** 2
            H += m2 * abs(phi[x]) ** 2
        return H
    a = {k: -math.tan(v / 2) for k, v in th.items()}
    st = State(a, {k: 0.0 for k in a}, {x: (phi[x].real, phi[x].imag) for x in SITES}, {x: (0.0, 0.0) for x in SITES})
    dE, dpi, J = forces(Rf, st, m2)
    errE = errP = 0.0
    for k in th:
        up = dict(th); dn = dict(th); up[k] += delta; dn[k] -= delta
        errE = max(errE, abs((Hpot(up, phi) - Hpot(dn, phi)) / (2 * delta) - dE[k]))
    for x in SITES:
        pr = dict(phi); mr = dict(phi); pi_ = dict(phi); mi = dict(phi)
        pr[x] += delta; mr[x] -= delta; pi_[x] += 1j * delta; mi[x] -= 1j * delta
        dRe = (Hpot(th, pr) - Hpot(th, mr)) / (2 * delta); dIm = (Hpot(th, pi_) - Hpot(th, mi)) / (2 * delta)
        wirt = 0.5 * (dRe - 1j * dIm)                                              # dH/dphi (Wirtinger derivative)
        errP = max(errP, abs(wirt - complex(dpi[x][0], dpi[x][1])))
    return errE, errP

def run_ring(label, R, nsteps, ntrials):
    rng = random.Random(5); m2 = R.mk(1); h = R.mk(1) / R.mk(3)
    res = dict(subG=0, subN=0, stepG=0, cont=0, contN=0, full=0, fullN=0, gauge=0, rev=0)
    for trial in range(ntrials):
        st = random_state(R, rng); G0 = gauss(R, st)
        for op in (lambda s: kick(R, s, h, m2), lambda s: drift(R, s, h), lambda s: step(R, s, h, m2)):
            res['subN'] += 1; res['subG'] += gauss(R, op(st)) == G0
        s5 = st
        for _ in range(nsteps): s5 = step(R, s5, h, m2)
        res['stepG'] += gauss(R, s5) == G0
        dE, dpi, J = forces(R, st, m2); k1 = kick(R, st, h, m2)
        for x in SITES:
            rho0 = -R.mk(2) * kmul(st.pi[x], st.phi[x])[1]; rho1 = -R.mk(2) * kmul(k1.pi[x], k1.phi[x])[1]
            divJ = R.zero
            for mu in (0, 1): divJ = divJ + J[(x, mu)] - J[(sh(x, mu, -1), mu)]
            res['contN'] += 1; res['cont'] += (rho1 - rho0 == h * divJ)
        # (b') continuity over a full step with the time-centred current
        h2 = h / R.mk(2); sa = kick(R, st, h2, m2); J0 = forces(R, st, m2)[2]; sb = drift(R, sa, h); J1 = forces(R, sb, m2)[2]; sc = kick(R, sb, h2, m2)
        for x in SITES:
            rho0 = -R.mk(2) * kmul(st.pi[x], st.phi[x])[1]; rho1 = -R.mk(2) * kmul(sc.pi[x], sc.phi[x])[1]
            d0 = d1 = R.zero
            for mu in (0, 1):
                d0 = d0 + J0[(x, mu)] - J0[(sh(x, mu, -1), mu)]; d1 = d1 + J1[(x, mu)] - J1[(sh(x, mu, -1), mu)]
            res['fullN'] += 1; res['full'] += (rho1 - rho0 == h2 * d0 + h2 * d1)
        gam = {x: (None if (R is not RQ and rng.randrange(R.p + 1) == R.p) else (Fr(rng.randint(-4, 4), rng.randint(1, 3)) if R is RQ else R.mk(rng.randrange(R.p)))) for x in SITES}
        res['gauge'] += step(R, gauge_transform(R, st, gam), h, m2).same(R, gauge_transform(R, step(R, st, h, m2), gam))
        res['rev'] += step(R, step(R, st, h, m2), -h, m2).same(R, st)
    print(f"   {label:<5} G conserved by kick/drift/full step: {res['subG']}/{res['subN']};  by {nsteps} full step(s): {res['stepG']}/{ntrials};  "
          f"continuity, kick (per site): {res['cont']}/{res['contN']};  continuity, full step: {res['full']}/{res['fullN']};  gauge covariance of the step: {res['gauge']}/{ntrials};  step(-h) inverts step(h): {res['rev']}/{ntrials}", flush=True)

def main():
    import sys
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "Q"):
        print("22. minimal coupled model, 2+1 D torus L = 3 (9 sites, 18 links), exact arithmetic, h = 1/3, m^2 = 1", flush=True)
        run_ring("Q", RQ, 1, 2)
    if which in ("all", "F7"): run_ring("F_7", RF(7), 5, 6)
    if which in ("all", "F11"): run_ring("F_11", RF(11), 5, 6)
    if which in ("all", "float"):
        eE, eP = forces_vs_hamiltonian()
        print(f"   (f) floating point: max |dH_pot/d theta - implemented force| = {eE:.2e};  max |dH_pot/d phi (Wirtinger) - implemented force| = {eP:.2e}", flush=True)
    if which in ("all", "inv"):
        print("23. Lorentz-invariant quadratic forms in (E, B)")
        dim, forms = invariants_test()
        print(f"   dimension of the space of real quadratic forms invariant under rotations and boosts of zeta = E + iB: {dim}")
        for f in forms: print("   basis element:", f)

if __name__ == "__main__":
    main()
