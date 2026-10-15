"""
Reproduction script for Calculations 11 and 12 (Part 2).

  11. Lattice covariant difference  D_mu psi(x) = U_mu(x) psi(x+mu) - psi(x)  with Cayley link phases
      (a) [D_mu, D_nu] psi = (U_P - 1) T psi           exact identity, for any links and any psi
      (b) gauge covariance of D and of [D, D]; gauge invariance of U_P
      (c) Cayley form of the field strength:  a_P = i [D_mu, D_nu] psi / S psi   and   U_P - 1 = -2 i a_P/(1 + i a_P)
      (d) the same identity for non-abelian SL(2,Q) caps (the Yang-Mills hinge)
      (e) continuum limit in floating point: [D_mu, D_nu] psi -> i q eps^2 F_{mu nu} psi, with D = d + i q A
  12. Charge q:  U -> U^q is the Cayley map c_q(a) = tan(q arctan a) (the tangent multiple-angle map)
      (a) closed forms, Chebyshev relation Re U^q = T_q(Re U)
      (b) homomorphism, additivity in q, periodicity in q mod (p+1), exhaustive over F_p
      (c) the phase group over F_p is cyclic of order N = p+1 (a Z_N gauge theory)
      (d) charge-q covariant derivative and plaquette over Q, F_7, F_11
      (e) single-plaquette comparison Z_N versus U(1)

Needs: numpy, scipy (parts 11e and 12e only).  Runs in well under a minute.
"""
import itertools
import math
import random
from fractions import Fraction as Fr

# ------------------------------------------------------------------ fields ---
def make_Fp(p):
    """A distinct field class for each prime, so that several F_p can coexist."""
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
        def __rtruediv__(s, o): return FpP(_v(o) * pow(s.v, -1, FpP.P))
        def __eq__(s, o): return s.v == _v(o) % FpP.P
        def __hash__(s): return hash(s.v)
        def __repr__(s): return str(s.v)
    return FpP

def _v(o): return o.v if hasattr(o, "v") else int(o)

def kadd(u, v): return (u[0] + v[0], u[1] + v[1])
def ksub(u, v): return (u[0] - v[0], u[1] - v[1])
def kmul(u, v): return (u[0] * v[0] - u[1] * v[1], u[0] * v[1] + u[1] * v[0])
def kconj(u): return (u[0], -u[1])
def kinv(u):
    n = u[0] * u[0] + u[1] * u[1]
    return (u[0] / n, -u[1] / n)
def kpow(u, q, one, zero):
    if q < 0: return kpow(kinv(u), -q, one, zero)
    r = (one, zero)
    for _ in range(q): r = kmul(r, u)
    return r
def ki(u): return (-u[1], u[0])                       # multiplication by i

class Ring:
    """Bundles the base field F: constructor mk, constants, Cayley phase and its inverse."""
    def __init__(s, mk, p=None): s.mk, s.p, s.one, s.zero = mk, p, mk(1), mk(0)
    def phase(s, a):                                   # U = (1 - i a)/(1 + i a);  a = None means infinity
        o, z = s.one, s.zero
        if a is None: return (-o, z)
        d = o + a * a
        return ((o - a * a) / d, s.mk(-2) * a / d)
    def a_of(s, u):                                    # inverse Cayley map
        return None if u[0] == -s.one else -u[1] / (s.one + u[0])
    def oplus(s, a, b): return s.a_of(kmul(s.phase(a), s.phase(b)))      # tangent addition
    def cq(s, a, q): return s.a_of(kpow(s.phase(a), q, s.one, s.zero))   # Cayley charge map

RQ = Ring(Fr)
def RF(p): return Ring(make_Fp(p), p)

def sampler(R):
    """(num, par): random field element, random Cayley parameter (None = infinity is allowed in par)."""
    if R is RQ:
        num = lambda rng: Fr(rng.randint(-6, 6), rng.randint(1, 4))
        par = lambda rng: None if rng.random() < 0.06 else num(rng)
    else:
        p = R.p
        num = lambda rng: R.mk(rng.randrange(p))
        par = lambda rng: None if rng.randrange(p + 1) == p else R.mk(rng.randrange(p))
    return num, par

# --------------------------------------------------------------- lattice ----
D4, L4 = 4, 3
SITES = list(itertools.product(range(L4), repeat=D4))
def sh(x, m): return tuple((x[i] + (i == m)) % L4 for i in range(D4))
PAIRS = [(m, n) for m in range(D4) for n in range(m + 1, D4)]

def Dop(U, m, f):                                      # (D_m f)(x) = U_m(x) f(x+m) - f(x)
    return {x: ksub(kmul(U[(x, m)], f[sh(x, m)]), f[x]) for x in SITES}

def plaq(U, x, m, n):                                  # U_m(x) U_n(x+m) U_m(x+n)^-1 U_n(x)^-1
    return kmul(kmul(U[(x, m)], U[(sh(x, m), n)]), kmul(kconj(U[(sh(x, n), m)]), kconj(U[(x, n)])))

def transport(U, f, x, m, n):                          # T f(x) = U_n(x) U_m(x+n) f(x+m+n)
    return kmul(kmul(U[(x, n)], U[(sh(x, n), m)]), f[sh(sh(x, m), n)])

def twopath(U, f, x, m, n):                            # S f(x) = (U_m U_n' + U_n U_m') f(x+m+n)
    t = f[sh(sh(x, m), n)]
    return kmul(kadd(kmul(U[(x, m)], U[(sh(x, m), n)]), kmul(U[(x, n)], U[(sh(x, n), m)])), t)

def random_links(R, par, rng):
    return {(x, m): R.phase(par(rng)) for x in SITES for m in range(D4)}

def random_psi(R, num, rng):
    out = {}
    for x in SITES:
        while True:
            v = (num(rng), num(rng))
            if not (v[0] == R.zero and v[1] == R.zero): break
        out[x] = v
    return out

def checks(R, q=1, seed=11):
    rng = random.Random(seed); num, par = sampler(R)
    U1 = random_links(R, par, rng)
    U = {k: kpow(u, q, R.one, R.zero) for k, u in U1.items()} if q != 1 else U1       # charge-q links
    psi = random_psi(R, num, rng)
    n = dict(ident=0, cayley=0, ratio=0, covD=0, covC=0, inv=0, qmap=0)
    tot = dict(ident=0, cayley=0, ratio=0, covD=0, covC=0, inv=0, qmap=0)
    g = {x: R.phase(par(rng)) for x in SITES}                        # gauge transformation (charge-1 phase)
    gq = {x: kpow(g[x], q, R.one, R.zero) for x in SITES}
    Up = {(x, m): kmul(kmul(gq[x], U[(x, m)]), kconj(gq[sh(x, m)])) for x in SITES for m in range(D4)}
    psip = {x: kmul(gq[x], psi[x]) for x in SITES}
    Dn = {m: Dop(U, m, psi) for m in range(D4)}
    Dnp = {m: Dop(Up, m, psip) for m in range(D4)}
    for m in range(D4):                                               # covariance of D
        for x in SITES:
            tot["covD"] += 1; n["covD"] += Dnp[m][x] == kmul(gq[x], Dn[m][x])
    for (m, nn) in PAIRS:
        Cm = {x: ksub(Dop(U, m, Dn[nn])[x], Dop(U, nn, Dn[m])[x]) for x in SITES}
        Cp = {x: ksub(Dop(Up, m, Dnp[nn])[x], Dop(Up, nn, Dnp[m])[x]) for x in SITES}
        for x in SITES:
            P = plaq(U, x, m, nn)
            T = transport(U, psi, x, m, nn)
            tot["ident"] += 1; n["ident"] += Cm[x] == kmul(ksub(P, (R.one, R.zero)), T)          # [D,D] psi = (U_P - 1) T psi
            tot["covC"] += 1; n["covC"] += Cp[x] == kmul(gq[x], Cm[x])                          # covariance of the commutator
            tot["inv"] += 1; n["inv"] += plaq(Up, x, m, nn) == P                                # gauge invariance of U_P
            a = R.a_of(P)
            S = twopath(U, psi, x, m, nn)
            tot["ratio"] += 1
            if a is None: n["ratio"] += S == (R.zero, R.zero)                                   # U_P = -1: S psi = 0
            else: n["ratio"] += ki(kmul(Cm[x], kinv(S))) == (a, R.zero)                         # a_P = i [D,D]psi / S psi
            tot["cayley"] += 1
            if a is None: n["cayley"] += P == (-R.one, R.zero)
            else: n["cayley"] += kmul(ksub(P, (R.one, R.zero)), (R.one, a)) == (R.zero, R.mk(-2) * a)   # (U_P - 1)(1 + i a_P) = -2 i a_P
            tot["qmap"] += 1
            a1 = R.a_of(plaq(U1, x, m, nn))
            if q == 1: n["qmap"] += 1
            else: n["qmap"] += (R.cq(a1, q) == a)                                               # a_P^(q) = c_q(a_P)
    return n, tot

# ---------------------------------------------- 11d: non-abelian SL(2,Q) -----
def m2(a, b, c, d): return ((a, b), (c, d))
def mmul(A, B): return tuple(tuple(sum(A[i][k] * B[k][j] for k in range(2)) for j in range(2)) for i in range(2))
def msub(A, B): return tuple(tuple(A[i][j] - B[i][j] for j in range(2)) for i in range(2))
def madd(A, B): return tuple(tuple(A[i][j] + B[i][j] for j in range(2)) for i in range(2))
def mdet(A): return A[0][0] * A[1][1] - A[0][1] * A[1][0]
def minv(A):
    d = mdet(A)
    return ((A[1][1] / d, -A[0][1] / d), (-A[1][0] / d, A[0][0] / d))
def mvec(A, v): return tuple(sum(A[i][j] * v[j] for j in range(2)) for i in range(2))
def vsub(u, v): return tuple(a - b for a, b in zip(u, v))
I2 = ((Fr(1), Fr(0)), (Fr(0), Fr(1)))

def cap(A): return mmul(madd(I2, A), minv(msub(I2, A)))               # Cayley cap (I + A)(I - A)^-1

def nonabelian_check(seed=5):
    rng = random.Random(seed)
    def rA():
        while True:
            a, b, c = (Fr(rng.randint(-3, 3), rng.randint(1, 3)) for _ in range(3))
            A = ((a, b), (c, -a))
            if mdet(msub(I2, A)) != 0: return A
    U = {(x, m): cap(rA()) for x in SITES for m in range(D4)}
    psi = {x: (Fr(rng.randint(-5, 5)), Fr(rng.randint(-5, 5))) for x in SITES}
    Dm = lambda m, f: {x: vsub(mvec(U[(x, m)], f[sh(x, m)]), f[x]) for x in SITES}
    ok = tot = 0; nonabelian = 0
    for (m, n) in PAIRS:
        C = {x: vsub(Dm(m, Dm(n, psi))[x], Dm(n, Dm(m, psi))[x]) for x in SITES}
        for x in SITES:
            P = mmul(mmul(U[(x, m)], U[(sh(x, m), n)]), mmul(minv(U[(sh(x, n), m)]), minv(U[(x, n)])))
            T = mvec(mmul(U[(x, n)], U[(sh(x, n), m)]), psi[sh(sh(x, m), n)])
            ok += C[x] == mvec(msub(P, I2), T); tot += 1
            Q = mmul(mmul(U[(x, n)], U[(sh(x, n), m)]), mmul(U[(x, m)], U[(sh(x, m), n)]))      # reversed loop product
            nonabelian += mmul(U[(x, m)], U[(sh(x, m), n)]) != mmul(U[(x, n)], U[(sh(x, n), m)])
    return ok, tot, nonabelian

# --------------------------------- 11e: continuum limit in floating point ----
def continuum_limit(q=1.3):
    import cmath
    x0, y0 = 0.1, 0.2
    Ax = lambda x, y: 0.3 * math.sin(2 * math.pi * y) + 0.1 * x
    Ay = lambda x, y: 0.2 * math.cos(2 * math.pi * x) + 0.15 * y
    F = lambda x, y: -0.4 * math.pi * math.sin(2 * math.pi * x) - 0.6 * math.pi * math.cos(2 * math.pi * y)   # d_x A_y - d_y A_x
    psi = lambda x, y: cmath.exp(1j * (0.7 * x + 0.4 * y)) * (1 + 0.3 * x - 0.2 * y)
    out = []
    for k in (3, 4, 5, 6, 7):
        e = 2.0 ** -k
        Ux = lambda x, y: cmath.exp(1j * q * (0.3 * math.sin(2 * math.pi * y) * e + 0.05 * ((x + e) ** 2 - x ** 2)))
        Uy = lambda x, y: cmath.exp(1j * q * (0.2 * math.cos(2 * math.pi * x) * e + 0.075 * ((y + e) ** 2 - y ** 2)))
        Dx = lambda f: (lambda x, y: Ux(x, y) * f(x + e, y) - f(x, y))
        Dy = lambda f: (lambda x, y: Uy(x, y) * f(x, y + e) - f(x, y))
        C = Dx(Dy(psi))(x0, y0) - Dy(Dx(psi))(x0, y0)
        R = C / (1j * q * e * e * F(x0, y0) * psi(x0, y0))
        out.append((e, R))
    return out

# ----------------------------------------------------------- charge map -----
def cheb(q, x):                                        # Chebyshev T_q(x), exact on Fractions or Fp
    t0, t1 = x * 0 + 1, x
    if q == 0: return t0
    for _ in range(q - 1): t0, t1 = t1, 2 * x * t1 - t0
    return t1

def charge_checks():
    rng = random.Random(9)
    res = {}
    # closed forms over Q
    ok2 = ok3 = okim = okT = tot = 0
    for _ in range(300):
        a = Fr(rng.randint(-20, 20), rng.randint(1, 9))
        if 1 - a * a != 0 and 1 - 3 * a * a != 0:
            tot += 1
            ok2 += RQ.cq(a, 2) == 2 * a / (1 - a * a)
            ok3 += RQ.cq(a, 3) == (3 * a - a ** 3) / (1 - 3 * a * a)
        z = kpow((Fr(1), a), 5, Fr(1), Fr(0))                              # (1 + i a)^5
        okim += (RQ.cq(a, 5) == (z[1] / z[0] if z[0] != 0 else None))
        for q in range(1, 7):
            u = kpow(RQ.phase(a), q, Fr(1), Fr(0))
            okT += u[0] == cheb(q, (1 - a * a) / (1 + a * a))
    res["closed"] = (ok2, ok3, tot, okim, 300, okT, 300 * 6)
    # homomorphism over Q (random) and over F_p (exhaustive)
    h = hq = 0
    for _ in range(500):
        a, b = (None if rng.random() < 0.05 else Fr(rng.randint(-9, 9), rng.randint(1, 5)) for _ in range(2))
        for q in (2, 3, 4, 5, -1, -3):
            hq += 1; h += RQ.cq(RQ.oplus(a, b), q) == RQ.oplus(RQ.cq(a, q), RQ.cq(b, q))
    res["hom_Q"] = (h, hq)
    rows = []
    for p in (3, 7, 11, 19, 23):
        R = RF(p); N = p + 1
        pts = [None] + [R.mk(v) for v in range(p)]
        hom = add = per = comp = tt = 0
        for a in pts:
            for b in pts:
                for q in (2, 3, 5):
                    tt += 1; hom += R.cq(R.oplus(a, b), q) == R.oplus(R.cq(a, q), R.cq(b, q))
            for q in range(0, N + 2):
                for r in (1, 2, 3, 5):
                    add += R.cq(a, q + r) == R.oplus(R.cq(a, q), R.cq(a, r))
                    comp += R.cq(R.cq(a, r), q) == R.cq(a, q * r)
                per += R.cq(a, q + N) == R.cq(a, q)
        # group order and generator
        zero = R.mk(0)
        def order(a):
            k, b = 1, a
            while not (b is not None and b == zero): b = R.oplus(b, a); k += 1
            return k
        orders = [order(a) for a in pts]
        gens = [a for a, o in zip(pts, orders) if o == N]
        ncharge = len(pts)
        rows.append((p, N, ncharge, max(orders), len(gens), hom, tt, add, per, comp,
                     len(pts) * (N + 2) * 4, len(pts) * (N + 2)))
    res["Fp"] = rows
    return res

def tableau_generator(p):
    R = RF(p); N = p + 1
    pts = [None] + [R.mk(v) for v in range(p)]
    zero = R.mk(0)
    for a in pts:
        b, k = a, 1
        while not (b is not None and b == zero): b = R.oplus(b, a); k += 1
        if k == N:
            return [R.cq(a, q) for q in range(N)]

def zn_vs_u1(beta):
    import numpy as np
    from scipy.special import iv
    u1 = iv(1, beta) / iv(0, beta)
    out = {}
    for N in (4, 8, 12, 20, 24):
        th = 2 * np.pi * np.arange(N) / N
        w = np.exp(beta * np.cos(th))
        out[N] = float((np.cos(th) * w).sum() / w.sum())
    return u1, out

# ------------------------------------------------------------------ main ----
def main():
    names = dict(ident="[D,D]psi = (U_P - 1) T psi", cayley="(U_P - 1)(1 + i a_P) = -2 i a_P", ratio="a_P = i [D,D]psi / S psi",
                 covD="D'psi' = g D psi", covC="[D,D]'psi' = g [D,D] psi", inv="U_P' = U_P")
    rings = (("Q", RQ), ("F_7", RF(7)), ("F_11", RF(11)))
    print("11. covariant difference, 4D torus L = 3 (81 sites, 324 links, 486 plaquettes)")
    for label, R in rings:
        n, t = checks(R, 1)
        print(f"   {label:<5}", "  ".join(f"{k}: {n[k]}/{t[k]}" for k in names))
    ok, tot, na = nonabelian_check()
    print(f"   SL(2,Q) caps: [D,D]psi = (U_P - 1) T psi  {ok}/{tot};  U_mu U_nu' != U_nu U_mu' on {na}/{tot} plaquettes (non-abelian)")
    print("   continuum limit, q = 1.3, D = d + i q A:   eps, [D,D]psi / (i q eps^2 F psi)")
    for e, R in continuum_limit():
        print(f"      eps = {e:.6f}   ratio = {R.real:.6f} {R.imag:+.6f}i   |ratio - 1| = {abs(R - 1):.2e}")
    print("\n12. charge q")
    c = charge_checks()
    print("   Q: c_2, c_3 closed forms {}/{} {}/{};  c_5 = Im/Re of (1+ia)^5 {}/{};  Re U^q = T_q(Re U), q=1..6: {}/{}".format(
        c["closed"][0], c["closed"][2], c["closed"][1], c["closed"][2], c["closed"][3], c["closed"][4], c["closed"][5], c["closed"][6]))
    print("   Q: c_q(a (+) b) = c_q(a) (+) c_q(b): {}/{}".format(*c["hom_Q"]))
    for (p, N, npts, mo, ng, hom, tt, add, per, comp, tadd, tper) in c["Fp"]:
        print(f"   F_{p}: points {npts} (= p+1), max order {mo} (cyclic, N = {N}), generators {ng}, hom {hom}/{tt}, "
              f"c_(q+r) = c_q (+) c_r {add}/{tadd}, periodic mod N {per}/{tper}, c_q o c_r = c_qr {comp}/{tadd}")
    print("   F_7 generator g: c_q(g), q = 0..7 :", tableau_generator(7))
    for q in (2, 3, -1, 5):
        for label, R in rings:
            n, t = checks(R, q, seed=12)
            print(f"   q = {q:>2} {label:<5} identity {n['ident']}/{t['ident']}  covariance {n['covD']}/{t['covD']}  a_P^(q) = c_q(a_P) {n['qmap']}/{t['qmap']}  U_P invariant {n['inv']}/{t['inv']}")
    print("   single plaquette <cos theta>: U(1) versus Z_N")
    for beta in (0.25, 1.0, 5.0):
        u1, zn = zn_vs_u1(beta)
        print(f"      beta = {beta}: U(1) {u1:.6f} | " + " | ".join(f"Z_{N} {v:.6f}" for N, v in zn.items()))

main()
