"""
Non-abelian lattice gauge theory with the spin Cayley cap.

Link:   U = cap(zeta) = (I + Z)(I - Z)^-1 = ((1 + zeta.zeta) I + 2 Z)/(1 - zeta.zeta),   Z = zeta.sigma
        zeta in K^3, K = Q[i] (exact) or F_p[i] (p = 3 mod 4)        -> gauge group SL(2,K)
        zeta = i w, w real/F_p-valued                                 -> gauge group SU(2)
Plaquette: P(x;mu,nu) = U_mu(x) U_nu(x+mu) U_mu(x+nu)^-1 U_nu(x)^-1

  S1  exact tests (Fractions and F_p), with path ordering
        (a) cap properties, gauge covariance of plaquettes
        (b) non-abelian Stokes: rectangle boundary = ordered product of transported plaquettes
        (c) cube Bianchi identity: the 6 transported face loops multiply to 1 ONLY in special
            orders; the valid orders are FOUND by search, then verified on every cube
  S2  continuum regime (SU(2), smooth non-commuting field):
        Wilson  sum(1 - tr P/2)   vs   the papers' potential  sum N(1 - w) = sum (1 - w)^2,  w = tr P/2

Needs: numpy (preinstalled in Google Colab).
Colab / terminal: paste into a cell or run `python cayley_su2_lattice.py`.  main() is called at the bottom.
"""
import itertools
import random
import time
from fractions import Fraction as Fr

import numpy as np

D = 4


# --------------------------------------------------------- scalars ---------
class Fp:
    p = 11
    __slots__ = ("v",)

    def __init__(s, v): s.v = int(v) % Fp.p
    def __add__(s, o): return Fp(s.v + _v(o))
    __radd__ = __add__
    def __sub__(s, o): return Fp(s.v - _v(o))
    def __neg__(s): return Fp(-s.v)
    def __mul__(s, o): return Fp(s.v * _v(o))
    __rmul__ = __mul__
    def __truediv__(s, o): return Fp(s.v * pow(_v(o), -1, Fp.p))
    def __eq__(s, o): return s.v == _v(o) % Fp.p
    def __hash__(s): return hash(s.v)


def _v(o):
    return o.v if isinstance(o, Fp) else int(o)


class Kel:
    """Element r + s*i of K = base[i]."""
    __slots__ = ("r", "i")

    def __init__(s, r, i): s.r, s.i = r, i
    def __add__(s, o): return Kel(s.r + o.r, s.i + o.i)
    def __sub__(s, o): return Kel(s.r - o.r, s.i - o.i)
    def __neg__(s): return Kel(-s.r, -s.i)
    def __mul__(s, o): return Kel(s.r * o.r - s.i * o.i, s.r * o.i + s.i * o.r)
    def conj(s): return Kel(s.r, -s.i)
    def inv(s):
        n = s.r * s.r + s.i * s.i
        return Kel(s.r / n, -s.i / n)
    def __truediv__(s, o): return s * o.inv()
    def __eq__(s, o): return s.r == o.r and s.i == o.i


class Ctx:
    def __init__(self, mk, rand_scalar, name):
        self.mk, self.rs, self.name = mk, rand_scalar, name
        self.zero = Kel(mk(0), mk(0)); self.one = Kel(mk(1), mk(0)); self.two = Kel(mk(2), mk(0))
        self.I = Kel(mk(0), mk(1))
        self.Id = [[self.one, self.zero], [self.zero, self.one]]

    def rand_K(self, rng):
        return Kel(self.rs(rng), self.rs(rng))


def mm(A, B):
    return [[A[0][0] * B[0][0] + A[0][1] * B[1][0], A[0][0] * B[0][1] + A[0][1] * B[1][1]],
            [A[1][0] * B[0][0] + A[1][1] * B[1][0], A[1][0] * B[0][1] + A[1][1] * B[1][1]]]


def minv(A):                                       # det = 1
    return [[A[1][1], -A[0][1]], [-A[1][0], A[0][0]]]


def mdag(A):
    return [[A[0][0].conj(), A[1][0].conj()], [A[0][1].conj(), A[1][1].conj()]]


def meq(A, B):
    return all(A[i][j] == B[i][j] for i in range(2) for j in range(2))


def det(A):
    return A[0][0] * A[1][1] - A[0][1] * A[1][0]


def cap(z, C):
    zz = z[0] * z[0] + z[1] * z[1] + z[2] * z[2]
    d = C.one - zz
    Z = [[z[2], z[0] - C.I * z[1]], [z[0] + C.I * z[1], -z[2]]]
    num = C.one + zz
    return [[(num + C.two * Z[0][0]) / d, (C.two * Z[0][1]) / d],
            [(C.two * Z[1][0]) / d, (num + C.two * Z[1][1]) / d]]


def rand_anchor(rng, C, su2):
    while True:
        if su2:
            z = [Kel(C.mk(0), C.rs(rng)) for _ in range(3)]          # zeta = i w
        else:
            z = [C.rand_K(rng) for _ in range(3)]
        zz = z[0] * z[0] + z[1] * z[1] + z[2] * z[2]
        d = C.one - zz
        if not (d.r == C.mk(0) and d.i == C.mk(0)):
            return z


def shift(x, mu, L, s=1):
    y = list(x); y[mu] = (y[mu] + s) % L
    return tuple(y)


class NA:
    def __init__(self, L, C, su2, seed, commuting=False):
        self.L, self.C, self.su2 = L, C, su2
        self.rng = random.Random(seed)
        self.sites = list(itertools.product(range(L), repeat=D))
        self.U = {}
        axis = rand_anchor(self.rng, C, su2)
        for x in self.sites:
            for m in range(D):
                if commuting:                      # all links share one axis -> abelian subgroup
                    t = C.rs(self.rng)
                    sc = Kel(C.mk(0), t) if su2 else Kel(t, C.mk(0))
                    z = [sc * axis[0], sc * axis[1], sc * axis[2]]
                    zz = z[0] * z[0] + z[1] * z[1] + z[2] * z[2]
                    dd = C.one - zz
                    if dd.r == C.mk(0) and dd.i == C.mk(0):
                        z = [C.zero, C.zero, C.zero]
                else:
                    z = rand_anchor(self.rng, C, su2)
                self.U[(x, m)] = cap(z, C)

    def plaq(self, x, mu, nu, U=None):
        U, L = (U or self.U), self.L
        return mm(mm(U[(x, mu)], U[(shift(x, mu, L), nu)]),
                  mm(minv(U[(shift(x, nu, L), mu)]), minv(U[(x, nu)])))


# ------------------------------------------------------------ S1 tests -----
def test_basic(lat):
    C, L = lat.C, lat.L
    planes = [(m, n) for m in range(D) for n in range(m + 1, D)]
    out = {}
    out["det U = 1 for every link"] = (sum(det(u) == C.one for u in lat.U.values()), len(lat.U))
    if lat.su2:
        out["U^dagger U = 1 (SU(2)) for every link"] = (
            sum(meq(mm(mdag(u), u), C.Id) for u in lat.U.values()), len(lat.U))
    # gauge covariance: P -> g P g^-1, hence trace invariant
    g = {x: cap(rand_anchor(lat.rng, C, lat.su2), C) for x in lat.sites}
    Ug = {(x, m): mm(mm(g[x], lat.U[(x, m)]), minv(g[shift(x, m, L)])) for (x, m) in lat.U}
    cov = tr = tot = 0
    for x in lat.sites:
        for m, n in planes:
            P = lat.plaq(x, m, n)
            Pg = lat.plaq(x, m, n, Ug)
            cov += meq(Pg, mm(mm(g[x], P), minv(g[x])))
            t0, t1 = P[0][0] + P[1][1], Pg[0][0] + Pg[1][1]
            tr += (t0 == t1)
            tot += 1
    out["plaquette covariant: P' = g P g^-1"] = (cov, tot)
    out["tr P gauge invariant"] = (tr, tot)
    return out


def rect_direct(lat, x, mu, nu, m, n):
    L, B, y = lat.L, lat.C.Id, x
    for _ in range(m): B = mm(B, lat.U[(y, mu)]); y = shift(y, mu, L)
    for _ in range(n): B = mm(B, lat.U[(y, nu)]); y = shift(y, nu, L)
    for _ in range(m): y = shift(y, mu, L, -1); B = mm(B, minv(lat.U[(y, mu)]))
    for _ in range(n): y = shift(y, nu, L, -1); B = mm(B, minv(lat.U[(y, nu)]))
    return B


def row_ordered(lat, x, mu, nu, m):
    """width-m, height-1 rectangle = product over k = m-1 ... 0 of  T_k P(x+k mu) T_k^-1."""
    L, B = lat.L, None
    for k in reversed(range(m)):
        T, y = lat.C.Id, x
        for _ in range(k): T = mm(T, lat.U[(y, mu)]); y = shift(y, mu, L)
        term = mm(mm(T, lat.plaq(y, mu, nu)), minv(T))
        B = term if B is None else mm(B, term)
    return B


def rect_ordered(lat, x, mu, nu, m, n):
    """B(m,n) = B(m,1)(x) * [ U_nu(x) B(m,n-1)(x+nu) U_nu(x)^-1 ]."""
    row = row_ordered(lat, x, mu, nu, m)
    if n == 1:
        return row
    V = lat.U[(x, nu)]
    up = rect_ordered(lat, shift(x, nu, lat.L), mu, nu, m, n - 1)
    return mm(row, mm(mm(V, up), minv(V)))


def test_stokes(lat, nbase):
    rng = random.Random(5)
    planes = [(m, n) for m in range(D) for n in range(m + 1, D)]
    ok = tot = unordered_ok = 0
    for (mu, nu) in planes:
        for x in rng.sample(lat.sites, min(nbase, len(lat.sites))):
            for m in (1, 2, 3):
                for n in (1, 2):
                    B = rect_direct(lat, x, mu, nu, m, n)
                    ok += meq(B, rect_ordered(lat, x, mu, nu, m, n))
                    # naive "abelian" version: plain product of the plaquettes, no transport
                    I = lat.C.Id
                    for i in range(m):
                        for j in range(n):
                            z = shift(shift(x, mu, lat.L, i), nu, lat.L, j)
                            I = mm(I, lat.plaq(z, mu, nu))
                    unordered_ok += meq(B, I)
                    tot += 1
    return ok, unordered_ok, tot


def cube_lassos(lat, x, axes):
    a, L = axes, lat.L
    out = {}
    for k in range(3):
        b, c = a[(k + 1) % 3], a[(k + 2) % 3]
        for side in (0, 1):
            v = shift(x, a[k], L) if side else x
            T = lat.U[(x, a[k])] if side else lat.C.Id          # tail x -> corner of the face
            Lp = lat.plaq(v, b, c)
            if side == 0:
                Lp = minv(Lp)                                   # outward orientation
            out[(k, side)] = mm(mm(T, Lp), minv(T))
    return out


def label(f):
    return "xyz"[f[0]] + ("+" if f[1] else "-")


def cube_search(lat, ncubes=8):
    """Find all orderings of the 6 transported face loops whose product is 1 on random cubes."""
    rng = random.Random(11)
    cubes = [(x, ax) for x in lat.sites for ax in itertools.combinations(range(D), 3)]
    sample = rng.sample(cubes, ncubes)
    faces = [(k, s) for k in range(3) for s in (0, 1)]
    valid = None
    for (x, ax) in sample:
        las = cube_lassos(lat, x, ax)
        ok = set()
        for perm in itertools.permutations(faces):
            B = lat.C.Id
            for f in perm:
                B = mm(B, las[f])
            if meq(B, lat.C.Id):
                ok.add(perm)
        valid = ok if valid is None else valid & ok
    return valid, cubes


def cube_verify(lat, cubes, orders, nmax=None):
    rng = random.Random(3)
    cs = cubes if nmax is None else rng.sample(cubes, min(nmax, len(cubes)))
    res = []
    for perm in orders:
        good = 0
        for (x, ax) in cs:
            las = cube_lassos(lat, x, ax)
            B = lat.C.Id
            for f in perm:
                B = mm(B, las[f])
            good += meq(B, lat.C.Id)
        res.append(good)
    return res, len(cs)


def section1():
    print("\n" + "-" * 78 + "\nS1  exact non-abelian lattice identities (4D torus, L = 4)\n" + "-" * 78)
    L = 4
    runs = [("F_11[i], gauge group SL(2,K)", 11, False), ("F_11[i], gauge group SU(2)", 11, True),
            ("F_7[i],  gauge group SL(2,K)", 7, False), ("Q[i] exact, SL(2,K)", None, False),
            ("Q[i] exact, SU(2)", None, True)]
    summary_orders = None
    for name, p, su2 in runs:
        if p is None:
            C = Ctx(Fr, lambda r: Fr(r.randint(-3, 3), r.randint(1, 3)), name)
            nbase, ncube = 3, 40
        else:
            Fp.p = p
            C = Ctx(Fp, lambda r: Fp(r.randrange(Fp.p)), name)
            nbase, ncube = 12, None
        t0 = time.time()
        lat = NA(L, C, su2, seed=7)
        print(f"\n  == {name} ==")
        for k, (ok, tot) in test_basic(lat).items():
            print(f"     {k:<46} {ok}/{tot}")
        ok, naive, tot = test_stokes(lat, nbase)
        print(f"     ordered non-abelian Stokes (rect up to 3x2)    {ok}/{tot}"
              f"      [plain unordered product of plaquettes: {naive}/{tot}]")
        if name.startswith("F_11[i], gauge group SL"):
            valid, cubes = cube_search(lat, ncubes=8)
            summary_orders = sorted(valid)
            print(f"     cube search: {len(valid)} of 720 face orderings give 1 on all 8 test cubes")
            rep = summary_orders[0]
            print("     e.g. ordering:", " ".join(label(f) for f in rep))
            res, n = cube_verify(lat, cubes, summary_orders)
            print(f"     all {len(summary_orders)} orderings verified on EVERY cube ({n}): {min(res)}..{max(res)} / {n}")
            cyc = {tuple(rep[i:] + rep[:i]) for i in range(6)}
            print(f"     cyclic rotations of the example are all in the set: {cyc <= set(summary_orders)}")
            if len(summary_orders) > 6:
                print(f"     ({len(summary_orders)-6} further orderings: e.g. swapping adjacent commuting faces)")
        elif summary_orders is not None:
            _, cubes = None, [(x, ax) for x in lat.sites for ax in itertools.combinations(range(D), 3)]
            res, n = cube_verify(lat, cubes, summary_orders, nmax=ncube)
            print(f"     Bianchi, all {len(summary_orders)} found orderings on {n} cubes: "
                  f"{min(res)}..{max(res)} / {n}")
        print(f"     [{time.time()-t0:.1f}s]")
    # abelian control: commuting links -> every ordering works
    Fp.p = 11
    C = Ctx(Fp, lambda r: Fp(r.randrange(Fp.p)), "ab")
    lat = NA(L, C, True, seed=9, commuting=True)
    _, cubes = None, [(x, ax) for x in lat.sites for ax in itertools.combinations(range(D), 3)]
    faces = [(k, s) for k in range(3) for s in (0, 1)]
    allp = list(itertools.permutations(faces))
    res, n = cube_verify(lat, cubes, allp[:50], nmax=20)
    print(f"\n  abelian control (all links on one axis, F_11): 50 arbitrary orderings, 20 cubes: "
          f"{min(res)}..{max(res)} / {n}  -> ordering is irrelevant when links commute")


# ------------------------------------------------------------ S2 ----------
SIG = np.array([[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]], dtype=complex)


def lattice_sums(L, amp):
    a, b, c = amp
    eps = 1.0 / L
    idx = np.indices((L,) * D).astype(float)

    def w_of(mu):
        X = (eps * idx).copy(); X[mu] += 0.5 * eps
        w = np.zeros((L,) * D + (3,))
        if mu == 0: w[..., 0] = a * np.cos(2 * np.pi * X[1])
        if mu == 1: w[..., 1] = b * np.cos(2 * np.pi * X[0])
        if mu == 2: w[..., 2] = c * np.cos(2 * np.pi * X[3])
        return w * (eps / 2)                              # cap(i w.sigma) ~ exp(i eps A.sigma)

    U = []
    for mu in range(D):
        w = w_of(mu)
        n2 = (w ** 2).sum(-1)[..., None, None]
        M = np.einsum("...a,aij->...ij", w, SIG)
        U.append(((1 - n2) * np.eye(2) + 2j * M) / (1 + n2))
    dag = lambda A: np.conj(np.swapaxes(A, -1, -2))
    sh = lambda A, ax: np.roll(A, -1, axis=ax)
    W = Cp = 0.0
    for mu in range(D):
        for nu in range(mu + 1, D):
            P = U[mu] @ sh(U[nu], mu) @ dag(sh(U[mu], nu)) @ dag(U[nu])
            w = 0.5 * np.einsum("...ii", P).real
            W += (1 - w).sum()
            Cp += ((1 - w) ** 2).sum()
    return W, Cp


def continuum(amp, N=40):
    a, b, c = amp
    g = (np.arange(N) + 0.0) / N
    X0, X1, X3 = np.meshgrid(g, g, g, indexing="ij")
    s = lambda k: SIG[k]
    A = [a * np.cos(2 * np.pi * X1)[..., None, None] * s(0),
         b * np.cos(2 * np.pi * X0)[..., None, None] * s(1),
         c * np.cos(2 * np.pi * X3)[..., None, None] * s(2),
         np.zeros((N, N, N, 2, 2), complex)]
    z = np.zeros((N, N, N, 2, 2), complex)
    dA = [[z] * 4 for _ in range(4)]                       # dA[mu][nu] = d_mu A_nu
    dA[1][0] = -2 * np.pi * a * np.sin(2 * np.pi * X1)[..., None, None] * s(0)
    dA[0][1] = -2 * np.pi * b * np.sin(2 * np.pi * X0)[..., None, None] * s(1)
    dA[3][2] = -2 * np.pi * c * np.sin(2 * np.pi * X3)[..., None, None] * s(2)
    E = Q = 0.0
    for mu in range(D):
        for nu in range(mu + 1, D):
            F = dA[mu][nu] - dA[nu][mu] + 1j * (A[mu] @ A[nu] - A[nu] @ A[mu])
            f2 = sum((0.5 * np.einsum("...ij,ji->...", F, SIG[k])).real ** 2 for k in range(3))
            E += f2.mean(); Q += (f2 ** 2).mean()
    return E, Q


def section2():
    print("\n" + "-" * 78 + "\nS2  continuum regime, SU(2): Wilson vs the papers' potential (1 - w)^2\n" + "-" * 78)
    amp = (0.6, 0.5, 0.4)
    E, Q = continuum(amp)
    print(f"  smooth non-commuting field A_0 = a cos(2 pi x1) s1, A_1 = b cos(2 pi x0) s2, A_2 = c cos(2 pi x3) s3,"
          f" amplitudes {amp}")
    print(f"  continuum:  E = sum_(mu<nu) int F^a F^a = {E:.5f},   Q = sum int (F^a F^a)^2 = {Q:.5f}")
    print(f"  expected:   Wilson W = sum(1 - tr P/2) -> E/2 = {E/2:.5f};   paper S = sum (1-w)^2 ~ (Q/4) eps^4")
    print(f"  {'L':>3} {'W':>10} {'(W-E/2)/(E/2)':>14} {'L^2 * that':>11} {'S_paper':>12} {'S_paper * L^4':>14} {'Q/4':>9}")
    for L in (4, 6, 8, 12):
        W, Cp = lattice_sums(L, amp)
        print(f"  {L:>3} {W:>10.5f} {(W-E/2)/(E/2):>14.2e} {(W-E/2)/(E/2)*L*L:>11.3f} "
              f"{Cp:>12.3e} {Cp*L**4:>14.4f} {Q/4:>9.4f}")
    print("  => Wilson converges to the Yang-Mills F^2 energy (error O(eps^2)).")
    print("     The papers' potential N(1 - w) = (1 - w)^2 is QUARTIC in F: it scales like eps^4 -> 0,")
    print("     with S_paper * L^4 -> (1/4) int sum (F^a F^a)^2. It is not a Yang-Mills action.")
    print("\n  single plaquette (SU(2) rotation, a = tan(phi/2) the anchor norm):")
    print(f"  {'phi':>6} {'a':>9} {'1-w = 2a^2/(1+a^2)':>20} {'(1-w)^2':>12}")
    for phi in (0.05, 0.2, 0.5, 1.0, 2.0, 3.0):
        a = np.tan(phi / 2); w = (1 - a * a) / (1 + a * a)
        print(f"  {phi:>6.2f} {a:>9.4f} {1-w:>20.6f} {(1-w)**2:>12.3e}")
    print("\n  complexified anchors (SL(2,C)): 1 - w = -2 z.z/(1 - z.z) ~ -2 z.z = -2[(E^2-B^2) + 2i E.B]")
    rng = np.random.default_rng(1)
    for s in (1e-1, 1e-2, 1e-3):
        z = s * (rng.normal(size=3) + 1j * rng.normal(size=3))
        zz = z @ z; w = (1 + zz) / (1 - zz)
        print(f"     |z| ~ {s:g}: (1-w)/(-2 z.z) = {(1-w)/(-2*zz):.5f}")


def main():
    t0 = time.time()
    section1()
    section2()
    print(f"\ndone in {time.time()-t0:.1f}s")


main()
