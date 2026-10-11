"""
U(1) lattice gauge theory with the Cayley phase.

Link variable  U = (1 - i a)/(1 + i a) = exp(-2 i arctan a)      (a in P^1: a = infinity -> U = -1)
Group law      phase(a) phase(b) = phase(a (+) b),   a (+) b = (a + b)/(1 - a b)   (tangent addition)
The Cayley map is a group isomorphism  (P^1(F), (+)) -> {norm-1 elements of F[i]}.

  S1  Exact Stokes / Bianchi / gauge invariance on a 4D torus, in
        (a) exact rational arithmetic (Fractions)
        (b) the finite field F_p, p = 3 mod 4 (phases live in F_{p^2}, group of order p+1)
  S2  Smooth-field continuum comparison (reals): Wilson  sum(1-cos th)  vs  Cayley "F^2" sum a^2
  S3  Strong-field / strong-coupling comparison: single plaquette, <cos th> for both actions

Needs: numpy and scipy (both preinstalled in Google Colab).
Colab / terminal: paste into a cell or run `python cayley_u1_lattice.py`.  main() is called at the bottom.
"""
import itertools
import random
import time
from fractions import Fraction as Fr

import numpy as np
from scipy.special import ive

trap = getattr(np, 'trapezoid', None) or np.trapz      # NumPy 2.x renamed trapz

D = 4


# ------------------------------------------------------- scalar fields -----
class Fp:
    """Tiny prime-field scalar (p = 3 mod 4) supporting + - * /."""
    p = 7
    __slots__ = ("v",)

    def __init__(s, v): s.v = int(v) % Fp.p
    def __add__(s, o): return Fp(s.v + _v(o))
    __radd__ = __add__
    def __sub__(s, o): return Fp(s.v - _v(o))
    def __rsub__(s, o): return Fp(_v(o) - s.v)
    def __neg__(s): return Fp(-s.v)
    def __mul__(s, o): return Fp(s.v * _v(o))
    __rmul__ = __mul__
    def __truediv__(s, o): return Fp(s.v * pow(_v(o), -1, Fp.p))
    def __eq__(s, o): return s.v == _v(o) % Fp.p
    def __hash__(s): return hash(s.v)
    def __repr__(s): return f"{s.v}"


def _v(o):
    return o.v if isinstance(o, Fp) else int(o)


# ------------------------------------------------- Cayley U(1) group -------
def phase(a, mk):
    if a is None:                                   # a = infinity
        return (mk(-1), mk(0))
    d = mk(1) + a * a
    return ((mk(1) - a * a) / d, mk(-2) * a / d)


def gmul(u, v):
    return (u[0] * v[0] - u[1] * v[1], u[0] * v[1] + v[0] * u[1])


def ginv(u):
    return (u[0], -u[1])                             # norm-one: inverse = conjugate


def a_of(u, mk):
    if u[0] == mk(-1):
        return None
    return -u[1] / (mk(1) + u[0])


def neg(a):
    return None if a is None else -a


def oplus(a, b, mk):
    if a is None and b is None:
        return mk(0)
    if a is None:
        return None if b == 0 else -mk(1) / b
    if b is None:
        return None if a == 0 else -mk(1) / a
    den = mk(1) - a * b
    return None if den == 0 else (a + b) / den


# --------------------------------------------------------- lattice ---------
def shift(x, mu, L, s=1):
    y = list(x); y[mu] = (y[mu] + s) % L
    return tuple(y)


class Lattice:
    def __init__(self, L, mk, rand_a, seed):
        self.L, self.mk = L, mk
        self.rng = random.Random(seed)
        self.sites = list(itertools.product(range(L), repeat=D))
        self.a = {(x, m): rand_a(self.rng) for x in self.sites for m in range(D)}
        self.U = {k: phase(v, mk) for k, v in self.a.items()}

    def plaq(self, x, mu, nu, U=None):
        U, L = (U or self.U), self.L
        return gmul(gmul(U[(x, mu)], U[(shift(x, mu, L), nu)]),
                    gmul(ginv(U[(shift(x, nu, L), mu)]), ginv(U[(x, nu)])))

    def plaq_a(self, x, mu, nu):                     # same plaquette by tangent addition only
        L, mk, a = self.L, self.mk, self.a
        t = oplus(a[(x, mu)], a[(shift(x, mu, L), nu)], mk)
        t = oplus(t, neg(a[(shift(x, nu, L), mu)]), mk)
        return oplus(t, neg(a[(x, nu)]), mk)

    def one(self):
        return (self.mk(1), self.mk(0))


def stokes_suite(lat):
    L, mk = lat.L, lat.mk
    one = lat.one()
    planes = [(m, n) for m in range(D) for n in range(m + 1, D)]
    res = {}
    # (a) product of 4 phases == tangent-addition composition of the 4 link a's
    ok = sum(a_of(lat.plaq(x, m, n), mk) == lat.plaq_a(x, m, n) or
             (a_of(lat.plaq(x, m, n), mk) is lat.plaq_a(x, m, n))
             for x in lat.sites for m, n in planes)
    res["plaquette = tangent-addition of link a's"] = (ok, len(lat.sites) * len(planes))
    # (b) gauge invariance
    g = {x: phase(mk(lat.rng.randint(-4, 4)) / mk(lat.rng.randint(1, 3)), mk) for x in lat.sites}
    Ug = {(x, m): gmul(gmul(g[x], lat.U[(x, m)]), ginv(g[shift(x, m, L)])) for (x, m) in lat.U}
    ok = sum(lat.plaq(x, m, n, Ug) == lat.plaq(x, m, n) for x in lat.sites for m, n in planes)
    res["plaquette gauge invariant"] = (ok, len(lat.sites) * len(planes))
    # (c) Bianchi on every 3-cube
    tot = ok = 0
    for x in lat.sites:
        for m, n, r in itertools.combinations(range(D), 3):
            P = lat.plaq
            B = one
            for (p1, p2, off) in ((n, r, m), (r, m, n), (m, n, r)):
                B = gmul(B, gmul(P(x, p1, p2), ginv(P(shift(x, off, L), p1, p2))))
            tot += 1; ok += (B == one)
    res["Bianchi: boundary of every 3-cube = 1"] = (ok, tot)
    # (d) torus: product of all plaquettes of a plane (for each fixed transverse slice) = 1
    tot = ok = 0
    for m, n in planes:
        others = [i for i in range(D) if i not in (m, n)]
        for t0 in range(L):
            for t1 in range(L):
                B = one
                for i in range(L):
                    for j in range(L):
                        x = [0] * D; x[m], x[n], x[others[0]], x[others[1]] = i, j, t0, t1
                        B = gmul(B, lat.plaq(tuple(x), m, n))
                tot += 1; ok += (B == one)
    res["torus: product over a whole 2-plane = 1"] = (ok, tot)
    # (e) Stokes for m x n rectangles: boundary holonomy == product of enclosed plaquettes
    tot = ok = 0
    for (mu, nu) in planes:
        for x in lat.sites:
            for m in (1, 2):
                for n in (1, 2):
                    B = one; y = x
                    for _ in range(m): B = gmul(B, lat.U[(y, mu)]); y = shift(y, mu, L)
                    for _ in range(n): B = gmul(B, lat.U[(y, nu)]); y = shift(y, nu, L)
                    for _ in range(m): y = shift(y, mu, L, -1); B = gmul(B, ginv(lat.U[(y, mu)]))
                    for _ in range(n): y = shift(y, nu, L, -1); B = gmul(B, ginv(lat.U[(y, nu)]))
                    I = one
                    for i in range(m):
                        for j in range(n):
                            z = shift(shift(x, mu, L, i), nu, L, j)
                            I = gmul(I, lat.plaq(z, mu, nu))
                    tot += 1; ok += (B == I)
    res["Stokes: rectangle boundary = product of plaquettes"] = (ok, tot)
    return res


def section1():
    print("\n" + "-" * 76 + "\nS1  exact Stokes theorem for the Cayley U(1) lattice (4D torus, L=3)\n" + "-" * 76)
    # (a) rational
    def rq(rng):
        if rng.random() < 0.06:
            return None
        return Fr(rng.randint(-6, 6), rng.randint(1, 4))
    lat = Lattice(3, Fr, rq, seed=1)
    t = time.time()
    print("  (a) exact rational arithmetic (Fractions), random links incl. a = infinity:")
    for k, (ok, tot) in stokes_suite(lat).items():
        print(f"      {k:<52} {ok}/{tot}")
    print(f"      [{time.time()-t:.1f}s]")
    # (b) finite fields
    for p in (7, 11):
        Fp.p = p
        mk = Fp
        # group isomorphism check, exhaustively
        pts = [None] + [Fp(i) for i in range(p)]
        phases = {phase(a, mk) for a in pts}
        iso = all(gmul(phase(a, mk), phase(b, mk)) == phase(oplus(a, b, mk), mk)
                  for a in pts for b in pts)
        print(f"  (b) F_{p}: Cayley map P^1(F_p) -> norm-1 group of F_p[i]: {len(phases)} distinct phases"
              f" (= p+1 = {p+1}), homomorphism on all {(p+1)**2} pairs: {iso}")
        lat = Lattice(3, mk, lambda rng: None if rng.randrange(p + 1) == p else Fp(rng.randrange(p)), seed=2)
        for k, (ok, tot) in stokes_suite(lat).items():
            print(f"      {k:<52} {ok}/{tot}")
        # Wilson analogue N(1-U) = 4a^2/(1+a^2) in F_p
        good = tot = 0
        for x in lat.sites:
            for m in range(D):
                for n in range(m + 1, D):
                    U = lat.plaq(x, m, n); a = a_of(U, mk)
                    lhs = (mk(1) - U[0]) * (mk(1) - U[0]) + U[1] * U[1]
                    rhs = mk(4) if a is None else mk(4) * a * a / (mk(1) + a * a)
                    tot += 1; good += (lhs == rhs)
        print(f"      N(1-U_P) = 4a_P^2/(1+a_P^2) (Wilson plaquette term in F_p): {good}/{tot}")
    print("  Note: all of this is exact because the group is ABELIAN and the Cayley map is an")
    print("  isomorphism onto the circle group. This order-independent form of Stokes is not available")
    print("  for non-abelian holonomies (path ordering is needed); it was not tested here.")


# ---------------------------------------------------- S2 continuum ---------
def continuum(L, a=0.5, b=0.5):
    eps = 1.0 / L
    idx = np.indices((L,) * D).astype(float)

    def theta(mu):
        X = eps * idx
        X = X.copy(); X[mu] += 0.5 * eps          # midpoint rule for the link integral
        if mu == 0:
            A = a * np.cos(2 * np.pi * X[1])
        elif mu == 2:
            A = b * np.cos(2 * np.pi * X[3])
        else:
            A = np.zeros_like(X[0])
        return eps * A

    th = [theta(m) for m in range(D)]
    sh = lambda arr, mu: np.roll(arr, -1, axis=mu)
    W = C = T2 = comp = 0.0
    for mu in range(D):
        for nu in range(mu + 1, D):
            tp = th[mu] + sh(th[nu], mu) - sh(th[mu], nu) - th[nu]       # exact Stokes
            W += (1 - np.cos(tp)).sum()
            C += (np.tan(tp / 2) ** 2).sum()
            T2 += (tp ** 2).sum()
            al = np.tan(th[mu] / 2); be = np.tan(sh(th[nu], mu) / 2)
            ga = np.tan(sh(th[mu], nu) / 2); de = np.tan(th[nu] / 2)
            op = lambda u, v: (u + v) / (1 - u * v)
            chain = op(op(op(al, be), -ga), -de)
            comp = max(comp, abs(chain - np.tan(tp / 2)).max())
    return W, C, T2, comp


def section2():
    print("\n" + "-" * 76 + "\nS2  continuum regime: Wilson vs Cayley F^2 action on a smooth periodic field\n" + "-" * 76)
    a = b = 0.5
    E = (2 * np.pi) ** 2 * (a * a + b * b) / 2          # sum_{mu<nu} int F_{mu nu}^2
    print(f"  A_0 = {a} cos(2 pi x1), A_2 = {b} cos(2 pi x3);  continuum  sum int F^2 = {E:.5f}")
    print("  lattice: W = sum(1-cos th_P),  C = sum a_P^2,  a_P = tan(th_P/2) = tangent-addition of link a's")
    print(f"  expected limits: W -> E/2 = {E/2:.5f},  C -> E/4 = {E/4:.5f},  W/(2C) -> 1")
    print(f"  {'L':>3} {'W':>10} {'2C':>10} {'W/(2C)':>9} {'(W/(2C)-1) L^4':>15} {'(W-E/2)/(E/2)':>14} {'chain-vs-tan':>13}")
    for L in (4, 6, 8, 12, 16):
        W, C, T2, comp = continuum(L, a, b)
        r = W / (2 * C)
        print(f"  {L:>3} {W:>10.5f} {2*C:>10.5f} {r:>9.5f} {(r-1)*L**4:>15.4f} {(W-E/2)/(E/2):>14.2e} {comp:>13.1e}")
    print("  => both actions converge to the same continuum F^2 energy with O(eps^2) error;")
    print("     their ratio differs from 1 only at O(eps^4): per plaquette 2a^2/(1-cos th) = 1 + th^2/4 + ..., th = O(eps^2).")


# ------------------------------------------------ S3 strong regime ----------
def section3():
    print("\n" + "-" * 76 + "\nS3  where they differ: one plaquette at large angle / strong coupling\n" + "-" * 76)
    print("  Wilson term 1-cos th = 2a^2/(1+a^2) (exact, rational in a);  Cayley F^2 term 2a^2 (matched)")
    print(f"  {'theta':>7} {'a=tan(th/2)':>12} {'1-cos th':>10} {'2a^2':>10} {'ratio 2a^2/(1-cos)=1+a^2':>26}")
    for th in (0.01, 0.1, 0.5, 1.0, 2.0, 3.0, np.pi):
        a = np.tan(th / 2)
        w = 1 - np.cos(th)
        print(f"  {th:>7.3f} {a:>12.4g} {w:>10.5f} {2*a*a:>10.4g} {(1+a*a):>26.4g}")
    print("  (at theta = pi, a = infinity: Wilson term = 2, F^2 term diverges)")
    print("\n  <cos th_P> for one plaquette  [Wilson weight exp(-beta(1-cos th)) vs Cayley weight exp(-2 beta a^2)]")
    th = np.linspace(-np.pi, np.pi, 1_000_001)
    c = np.cos(th)
    t2 = np.tan(th / 2) ** 2
    print(f"  {'beta':>6} {'Wilson I1/I0':>14} {'Cayley F^2':>12} {'difference':>12}")
    for beta in (0.25, 0.5, 1, 2, 5, 10, 20, 50):
        wil = ive(1, beta) / ive(0, beta)
        w = np.exp(-2 * beta * t2)
        cay = trap(c * w, th) / trap(w, th)
        print(f"  {beta:>6} {wil:>14.6f} {cay:>12.6f} {cay-wil:>12.2e}")
    print("  => agree at weak coupling (beta large), differ at strong coupling: different lattice theories")
    print("     with the same continuum limit.")


def main():
    t0 = time.time()
    section1()
    section2()
    section3()
    print(f"\ndone in {time.time()-t0:.1f}s")


main()
