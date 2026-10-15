"""
Reproduction script for Calculations 13 and 14 (Part 3).

  13. The photon mass term on the Cayley lattice
      (a) gauge transformation of a link in Cayley form: a' = gamma(x) (+) a (+) (-gamma(x+mu))
      (b) single-link no-go: the gauge orbit of a link is the whole phase group, so no non-constant
          function of one link is gauge invariant
      (c) continuum limit of the variation: matches the lecture's (A - d phi)^2 - A^2
      (d) orbit counting on a small torus over F_3[i]: gauge orbits are separated by holonomies
  14. A Wilson-Dirac matter field over K = F[i]
      (a) Euclidean gamma matrices with entries in K:  {g_mu, g_nu} = 2 delta,  g5 hermitian
      (b) exact gauge covariance D[U'](g psi) = g D[U] psi, invariance of psi-bar D psi; the free operator is not
          covariant under local transformations (only under global ones)
      (c) gamma5-hermiticity  <phi, g5 D g5 psi> = <D phi, psi>
      (d) exact lattice current identity   div j = (psi-bar D) psi - psi-bar (D psi),  gauge invariance of j
      (e) charge k:  U -> U^k, g -> g^k
      (f) fermion doubling in exact arithmetic on the L = 4 torus (omega = i is a 4th root of unity in K),
          and in floating point

Needs: numpy (parts 14f floating point only).  Runs in about a minute.
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
def kneg(u): return (-u[0], -u[1])
def kconj(u): return (u[0], -u[1])
def kinv(u):
    n = u[0] * u[0] + u[1] * u[1]
    return (u[0] / n, -u[1] / n)
def kpow(u, q, one, zero):
    if q < 0: return kpow(kinv(u), -q, one, zero)
    r = (one, zero)
    for _ in range(q): r = kmul(r, u)
    return r

class Ring:
    def __init__(s, mk, p=None):
        s.mk, s.p, s.one, s.zero = mk, p, mk(1), mk(0)
        s.K1, s.K0, s.Ki = (s.one, s.zero), (s.zero, s.zero), (s.zero, s.one)
    def phase(s, a):                                   # U = (1 - i a)/(1 + i a);  a = None means infinity
        if a is None: return (-s.one, s.zero)
        d = s.one + a * a
        return ((s.one - a * a) / d, s.mk(-2) * a / d)
    def a_of(s, u):
        return None if u[0] == -s.one else -u[1] / (s.one + u[0])
    def neg_a(s, a): return None if a is None else -a
    def oplus(s, a, b): return s.a_of(kmul(s.phase(a), s.phase(b)))
    def iszero(s, u): return u[0] == s.zero and u[1] == s.zero

RQ = Ring(Fr)
def RF(p): return Ring(make_Fp(p), p)

def sampler(R):
    if R is RQ:
        num = lambda rng: Fr(rng.randint(-6, 6), rng.randint(1, 4))
        par = lambda rng: None if rng.random() < 0.06 else num(rng)
    else:
        p = R.p
        num = lambda rng: R.mk(rng.randrange(p))
        par = lambda rng: None if rng.randrange(p + 1) == p else R.mk(rng.randrange(p))
    return num, par


# ======================================================================
#  13.  THE PHOTON MASS TERM
# ======================================================================
def mass_a_b_c():
    rng = random.Random(21)
    out = {}
    for label, R in (("Q", RQ), ("F_7", RF(7)), ("F_11", RF(11))):
        num, par = sampler(R)
        ok = n = 0
        for _ in range(1000):
            a, g1, g2 = par(rng), par(rng), par(rng)
            u = kmul(kmul(R.phase(g1), R.phase(a)), kconj(R.phase(g2)))        # g(x) U g(x+mu)^-1
            ok += R.a_of(u) == R.oplus(R.oplus(g1, a), R.neg_a(g2)); n += 1
        out[label] = (ok, n)
    # single-link orbit and non-invariance, exhaustive over F_p
    rows = []
    for p in (3, 7, 11):
        R = RF(p); pts = [None] + [R.mk(v) for v in range(p)]
        f = lambda a: R.one - R.phase(a)[0]                                    # 1 - Re U  (defined at a = infinity too)
        orbit_full = all(len({R.oplus(R.oplus(g1, a), R.neg_a(g2)) for g1 in pts for g2 in pts}) == p + 1 for a in pts)
        tot = inv = 0
        for a in pts:
            for g1 in pts:
                for g2 in pts:
                    b = R.oplus(R.oplus(g1, a), R.neg_a(g2)); tot += 1; inv += f(b) == f(a)
        always = sum(all(f(R.oplus(R.oplus(g1, a), R.neg_a(g2))) == f(a) for g1 in pts for g2 in pts) for a in pts)
        rows.append((p, orbit_full, inv, tot, always, len(pts)))
    out["Fp"] = rows
    # Q: random
    inv = tot = 0
    num, par = sampler(RQ)
    rng = random.Random(22)
    for _ in range(5000):
        a, g1, g2 = par(rng), par(rng), par(rng)
        b = RQ.oplus(RQ.oplus(g1, a), RQ.neg_a(g2)); tot += 1
        inv += (RQ.one - RQ.phase(b)[0]) == (RQ.one - RQ.phase(a)[0])
    out["Qinv"] = (inv, tot)
    return out

def mass_c_continuum():
    A, dphi0, x0 = 0.7, 0.4, 0.0                      # A_mu at the link, phi(x) = dphi0 x + x^2 / 2  (so dphi = dphi0 + x)
    res = []
    for k in (3, 4, 5, 6, 7, 8):
        e = 2.0 ** -k
        th = e * A                                    # link angle = e*eps*A, with e = 1
        th2 = th - (( dphi0 * (x0 + e) + (x0 + e) ** 2 / 2) - (dphi0 * x0 + x0 ** 2 / 2))   # U' = g(x) U g(x+mu)^-1
        d = (1 - math.cos(th2)) - (1 - math.cos(th))
        target = -2 * A * (dphi0 + x0) + (dphi0 + x0) ** 2
        res.append((e, d / (0.5 * e * e), target))
    return res

def mass_d_orbits(p=3):
    R = RF(p); pts = [None] + [R.mk(v) for v in range(p)]
    N = p + 1; L = 2
    sites = [(i, j) for i in range(L) for j in range(L)]
    links = [(x, m) for x in sites for m in (0, 1)]
    sh = lambda x, m: ((x[0] + (m == 0)) % L, (x[1] + (m == 1)) % L)
    phs = {a: R.phase(a) for a in pts}
    hol = set()
    key = lambda u: (u[0].v, u[1].v)
    for cfg in itertools.product(pts, repeat=len(links)):
        U = {l: phs[a] for l, a in zip(links, cfg)}
        t = []
        for x in sites:
            P = kmul(kmul(U[(x, 0)], U[(sh(x, 0), 1)]), kmul(kconj(U[(sh(x, 1), 0)]), kconj(U[(x, 1)])))
            t.append(key(P))
        t.append(key(kmul(U[((0, 0), 0)], U[((1, 0), 0)])))                # non-contractible loop in direction 0
        t.append(key(kmul(U[((0, 0), 1)], U[((0, 1), 1)])))                # non-contractible loop in direction 1
        hol.add(tuple(t))
    # free action of the gauge group modulo constants: orbit size N^(V-1)
    rng = random.Random(5)
    sizes = set()
    for _ in range(30):
        cfg = [rng.choice(pts) for _ in links]
        U = {l: phs[a] for l, a in zip(links, cfg)}
        imgs = set()
        for gs in itertools.product(pts, repeat=len(sites)):
            g = {x: phs[a] for x, a in zip(sites, gs)}
            imgs.add(tuple(key(kmul(kmul(g[x], U[(x, m)]), kconj(g[sh(x, m)]))) for (x, m) in links))
        sizes.add(len(imgs))
    n_cfg = N ** len(links); orbit = N ** (len(sites) - 1)
    return n_cfg, sizes, orbit, n_cfg // orbit, len(hol), N ** (len(links) - len(sites) + 1)


# ======================================================================
#  14.  A WILSON-DIRAC MATTER FIELD OVER K = F[i]
# ======================================================================
def mm(R, A, B):
    n = len(A); m = len(B[0]); k = len(B)
    out = []
    for i in range(n):
        row = []
        for j in range(m):
            s = R.K0
            for t in range(k): s = kadd(s, kmul(A[i][t], B[t][j]))
            row.append(s)
        out.append(row)
    return out
def mv(R, A, v):
    out = []
    for i in range(len(A)):
        s = R.K0
        for t in range(len(v)): s = kadd(s, kmul(A[i][t], v[t]))
        out.append(s)
    return out
def vm(R, v, A):
    out = []
    for j in range(len(A[0])):
        s = R.K0
        for t in range(len(v)): s = kadd(s, kmul(v[t], A[t][j]))
        out.append(s)
    return out
def vdot(R, a, b):
    s = R.K0
    for x, y in zip(a, b): s = kadd(s, kmul(x, y))
    return s
def vs(c, v): return [kmul(c, x) for x in v]
def vadd(a, b): return [kadd(x, y) for x, y in zip(a, b)]
def vsub(a, b): return [ksub(x, y) for x, y in zip(a, b)]
def msc(c, A): return [[kmul(c, x) for x in row] for row in A]
def madd(A, B): return [[kadd(x, y) for x, y in zip(r1, r2)] for r1, r2 in zip(A, B)]
def msub(A, B): return [[ksub(x, y) for x, y in zip(r1, r2)] for r1, r2 in zip(A, B)]
def mdag(A): return [[kconj(A[j][i]) for j in range(len(A))] for i in range(len(A[0]))]
def ident(R, n): return [[R.K1 if i == j else R.K0 for j in range(n)] for i in range(n)]
def zeros(R, n): return [[R.K0 for _ in range(n)] for _ in range(n)]
def meq(R, A, B): return all(x == y for r1, r2 in zip(A, B) for x, y in zip(r1, r2))

def rank_det(R, M):
    """Gaussian elimination over K: returns (rank, det)."""
    n = len(M); A = [row[:] for row in M]; det = R.K1; rank = 0
    for c in range(n):
        piv = next((r for r in range(rank, n) if not R.iszero(A[r][c])), None)
        if piv is None:
            det = R.K0; continue
        if piv != rank:
            A[piv], A[rank] = A[rank], A[piv]; det = kneg(det)
        det = kmul(det, A[rank][c]); inv = kinv(A[rank][c])
        for r in range(rank + 1, n):
            f = kmul(A[r][c], inv)
            A[r] = [ksub(A[r][t], kmul(f, A[rank][t])) for t in range(n)]
        rank += 1
    return rank, det

def gammas(R):
    o, z, i_ = R.K1, R.K0, R.Ki
    mi = kneg(i_)
    sig = [[[z, o], [o, z]], [[z, mi], [i_, z]], [[o, z], [z, kneg(o)]]]         # Pauli matrices over K
    def blocks(a, b, c, d):
        return [a[0] + b[0], a[1] + b[1], c[0] + d[0], c[1] + d[1]]
    Z2 = [[z, z], [z, z]]; I2 = [[o, z], [z, o]]
    g = []
    for k in range(3):
        mi_s = msc(mi, sig[k]); i_s = msc(i_, sig[k])                            # -i sigma_k and i sigma_k
        g.append(blocks(Z2, mi_s, i_s, Z2))
    g.append(blocks(Z2, I2, I2, Z2))
    return g                                                                      # gamma_1, gamma_2, gamma_3, gamma_4

def clifford_checks(R):
    g = gammas(R); I4 = ident(R, 4); ok = tot = 0
    two = msc((R.mk(2), R.zero), I4)
    for m in range(4):
        for n in range(4):
            lhs = madd(mm(R, g[m], g[n]), mm(R, g[n], g[m]))
            tot += 1; ok += meq(R, lhs, two if m == n else zeros(R, 4))
    g5 = mm(R, mm(R, g[0], g[1]), mm(R, g[2], g[3]))
    herm = all(meq(R, mdag(x), x) for x in g)
    return ok, tot, herm, meq(R, mm(R, g5, g5), I4), meq(R, mdag(g5), g5), all(meq(R, madd(mm(R, g5, x), mm(R, x, g5)), zeros(R, 4)) for x in g), g, g5


class Lat:
    def __init__(s, L, d=4):
        s.L, s.d = L, d
        s.sites = list(itertools.product(range(L), repeat=d))
    def f(s, x, m): return tuple((x[i] + (i == m)) % s.L for i in range(s.d))
    def b(s, x, m): return tuple((x[i] - (i == m)) % s.L for i in range(s.d))

class Dirac:
    """Wilson-Dirac operator  D psi(x) = (m+4r) psi(x) - sum_mu [ (r-g_mu)/2 U_mu(x) psi(x+mu) + (r+g_mu)/2 U_mu(x-mu)^-1 psi(x-mu) ]"""
    def __init__(s, R, lat, m, r):
        s.R, s.lat, s.m, s.r = R, lat, m, r
        s.g = gammas(R); I4 = ident(R, 4); half = (R.one / R.mk(2), R.zero)
        rI = msc((R.mk(r), R.zero), I4)
        s.Pm = [msc(half, msub(rI, s.g[mu])) for mu in range(4)]
        s.Pp = [msc(half, madd(rI, s.g[mu])) for mu in range(4)]
        s.diag = (R.mk(m + 4 * r), R.zero)
        s.g5 = mm(R, mm(R, s.g[0], s.g[1]), mm(R, s.g[2], s.g[3]))
    def D(s, U, psi):                                 # column action; U = None means the free operator
        R, lat = s.R, s.lat; out = {}
        for x in lat.sites:
            acc = vs(s.diag, psi[x])
            for mu in range(4):
                xf, xb = lat.f(x, mu), lat.b(x, mu)
                uf = U[(x, mu)] if U else R.K1
                ub = kinv(U[(xb, mu)]) if U else R.K1
                acc = vsub(acc, mv(R, s.Pm[mu], vs(uf, psi[xf])))
                acc = vsub(acc, mv(R, s.Pp[mu], vs(ub, psi[xb])))
            out[x] = acc
        return out
    def Dbar(s, U, pb):                               # row action  (psi-bar D)(y)
        R, lat = s.R, s.lat; out = {}
        for y in lat.sites:
            acc = vs(s.diag, pb[y])
            for mu in range(4):
                yb, yf = lat.b(y, mu), lat.f(y, mu)
                u1 = U[(yb, mu)] if U else R.K1
                u2 = kinv(U[(y, mu)]) if U else R.K1
                acc = vsub(acc, vs(u1, vm(R, pb[yb], s.Pm[mu])))
                acc = vsub(acc, vs(u2, vm(R, pb[yf], s.Pp[mu])))
            out[y] = acc
        return out
    def current(s, U, pb, psi, x, mu):
        R, lat = s.R, s.lat; xf = lat.f(x, mu)
        t1 = vdot(R, pb[x], mv(R, s.Pm[mu], vs(U[(x, mu)], psi[xf])))
        t2 = vdot(R, pb[xf], mv(R, s.Pp[mu], vs(kinv(U[(x, mu)]), psi[x])))
        return ksub(t1, t2)

def rand_vec(R, rng, num): return [(num(rng), num(rng)) for _ in range(4)]

def dirac_checks(R, label, seed=31):
    rng = random.Random(seed); num, par = sampler(R); lat = Lat(3)
    res = {}
    for r in (0, 1):
        dm = Dirac(R, lat, 2, r)
        U1 = {(x, m): R.phase(par(rng)) for x in lat.sites for m in range(4)}
        psi = {x: rand_vec(R, rng, num) for x in lat.sites}
        pb = {x: rand_vec(R, rng, num) for x in lat.sites}
        gph = {x: R.phase(par(rng)) for x in lat.sites}
        for k in (1, 2, 3):
            U = {key: kpow(u, k, R.one, R.zero) for key, u in U1.items()} if k > 1 else U1
            g = {x: kpow(gph[x], k, R.one, R.zero) for x in lat.sites}
            Up = {(x, m): kmul(kmul(g[x], U[(x, m)]), kconj(g[lat.f(x, m)])) for x in lat.sites for m in range(4)}
            psip = {x: vs(g[x], psi[x]) for x in lat.sites}
            pbp = {x: vs(kinv(g[x]), pb[x]) for x in lat.sites}
            Dpsi = dm.D(U, psi); Dpsip = dm.D(Up, psip)
            cov = sum(Dpsip[x] == vs(g[x], Dpsi[x]) for x in lat.sites)
            inv = sum(vdot(R, pbp[x], Dpsip[x]) == vdot(R, pb[x], Dpsi[x]) for x in lat.sites)
            mass = sum(vdot(R, pbp[x], psip[x]) == vdot(R, pb[x], psi[x]) for x in lat.sites)
            res[(r, k)] = dict(cov=cov, inv=inv, mass=mass)
            if k == 1:
                # current identity and gauge invariance of j
                Dbar = dm.Dbar(U, pb); ident_ok = jinv = 0
                for x in lat.sites:
                    lhs = R.K0
                    for mu in range(4):
                        lhs = kadd(lhs, ksub(dm.current(U, pb, psi, x, mu), dm.current(U, pb, psi, lat.b(x, mu), mu)))
                    rhs = ksub(vdot(R, Dbar[x], psi[x]), vdot(R, pb[x], Dpsi[x]))
                    ident_ok += lhs == rhs
                    for mu in range(4): jinv += dm.current(Up, pbp, psip, x, mu) == dm.current(U, pb, psi, x, mu)
                res[(r, 1)].update(cur=ident_ok, jinv=jinv)
                # free operator: global versus local gauge transformation
                glob = gph[lat.sites[0]]
                dfree = dm.D(None, {x: vs(glob, psi[x]) for x in lat.sites}); df = dm.D(None, psi)
                res[(r, 1)]["free_global"] = sum(dfree[x] == vs(glob, df[x]) for x in lat.sites)
                dfree_l = dm.D(None, psip)
                res[(r, 1)]["free_local"] = sum(dfree_l[x] == vs(g[x], df[x]) for x in lat.sites)
                # gamma5-hermiticity:  <phi, g5 D g5 psi> = <D phi, psi>
                ghe = 0
                for _ in range(3):
                    phi = {x: rand_vec(R, rng, num) for x in lat.sites}; ps = {x: rand_vec(R, rng, num) for x in lat.sites}
                    g5ps = {x: mv(R, dm.g5, ps[x]) for x in lat.sites}
                    t = dm.D(U, g5ps); g5t = {x: mv(R, dm.g5, t[x]) for x in lat.sites}
                    lhs = R.K0; rhs = R.K0; Dphi = dm.D(U, phi)
                    for x in lat.sites:
                        lhs = kadd(lhs, vdot(R, [kconj(c) for c in phi[x]], g5t[x]))
                        rhs = kadd(rhs, vdot(R, [kconj(c) for c in Dphi[x]], ps[x]))
                    ghe += lhs == rhs
                res[(r, 1)]["g5herm"] = ghe
    # constant field: j_mu = - psi-bar gamma_mu psi when U = 1
    dm = Dirac(R, lat, 2, 1); pb = rand_vec(R, rng, num); ps = rand_vec(R, rng, num)
    cst = {x: ps for x in lat.sites}; cpb = {x: pb for x in lat.sites}; U1 = {(x, m): R.K1 for x in lat.sites for m in range(4)}
    x0 = lat.sites[0]
    const_ok = sum(dm.current(U1, cpb, cst, x0, mu) == kneg(vdot(R, pb, mv(R, dm.g[mu], ps))) for mu in range(4))
    res["const"] = const_ok
    return res

# ---- doubling, exact on the L = 4 torus: omega = i is a primitive 4th root of unity in K ----
def doubling_exact(R, r, m=0):
    g = gammas(R); half = (R.one / R.mk(2), R.zero)
    omega = [R.K1, R.Ki, (-R.one, R.zero), kneg(R.Ki)]                        # i^0 ... i^3
    cnt = 0; nullity = 0; accidental = []
    for k in itertools.product(range(4), repeat=4):
        M = (R.mk(m), R.zero); Dk = zeros(R, 4)
        for mu in range(4):
            wp, wm = omega[k[mu]], omega[(-k[mu]) % 4]
            t = kmul(half, ksub(wp, wm)); c = kmul(half, kadd(wp, wm))
            M = kadd(M, kmul((R.mk(r), R.zero), ksub(R.K1, c)))
            Dk = madd(Dk, msc(t, g[mu]))
        Dk = madd(Dk, msc(M, ident(R, 4)))
        rk, _ = rank_det(R, Dk)
        if rk < 4:
            cnt += 1; nullity += 4 - rk
            accidental.append(k)
    return cnt, nullity, accidental

def doubling_check_position_space(R, r, ntest=6, seed=41):
    """Position-space free operator applied to a plane wave equals the momentum-space matrix times the wave."""
    rng = random.Random(seed); num, par = sampler(R); lat = Lat(4)
    g = gammas(R); half = (R.one / R.mk(2), R.zero); ok = 0
    omega = [R.K1, R.Ki, (-R.one, R.zero), kneg(R.Ki)]
    dm = Dirac(R, lat, 0, r)
    for _ in range(ntest):
        k = tuple(rng.randrange(4) for _ in range(4)); v = rand_vec(R, rng, num)
        psi = {x: vs(omega[sum(k[i] * x[i] for i in range(4)) % 4], v) for x in lat.sites}
        out = dm.D(None, psi)
        M = R.K0; Dk = zeros(R, 4)
        for mu in range(4):
            wp, wm = omega[k[mu]], omega[(-k[mu]) % 4]
            Dk = madd(Dk, msc(kmul(half, ksub(wp, wm)), g[mu]))
            M = kadd(M, kmul((R.mk(r), R.zero), ksub(R.K1, kmul(half, kadd(wp, wm)))))
        Dk = madd(Dk, msc(M, ident(R, 4)))
        pred = mv(R, Dk, v)
        ok += all(out[x] == vs(omega[sum(k[i] * x[i] for i in range(4)) % 4], pred) for x in lat.sites)
    return ok, ntest

def spin_generator_check(R):
    """The six spin generators [g_mu, g_nu]/4 are block diagonal with trace-free 2x2 blocks over K (sl(2,K) + sl(2,K))."""
    g = gammas(R); quarter = (R.one / R.mk(4), R.zero); ok = 0; tot = 0
    for m in range(4):
        for n in range(m + 1, 4):
            S = msc(quarter, msub(mm(R, g[m], g[n]), mm(R, g[n], g[m])))
            off = all(R.iszero(S[i][j]) for i in range(2) for j in range(2, 4)) and all(R.iszero(S[i][j]) for i in range(2, 4) for j in range(2))
            tr1 = kadd(S[0][0], S[1][1]); tr2 = kadd(S[2][2], S[3][3])
            tot += 1; ok += off and R.iszero(tr1) and R.iszero(tr2)
    return ok, tot

def chiral_check(R, r, seed=51):
    """g5 D g5 = -D at m = 0 for the naive operator (r = 0); broken by the Wilson term (r = 1)."""
    rng = random.Random(seed); num, par = sampler(R); lat = Lat(3)
    dm = Dirac(R, lat, 0, r)
    U = {(x, m): R.phase(par(rng)) for x in lat.sites for m in range(4)}
    psi = {x: rand_vec(R, rng, num) for x in lat.sites}
    g5psi = {x: mv(R, dm.g5, psi[x]) for x in lat.sites}
    t = dm.D(U, g5psi); lhs = {x: mv(R, dm.g5, t[x]) for x in lat.sites}
    Dp = dm.D(U, psi)
    return sum(lhs[x] == [kneg(c) for c in Dp[x]] for x in lat.sites)

def doubling_predicted(p):
    """Wilson r = 1, m = 0: det D(k) = (M^2 + n_odd)^2 with M = n_odd + 2 n_2.
    n_odd = j components of k odd (2 values each), n_2 = l components equal to 2.  Count momenta with M^2 = -j (mod p)."""
    from math import comb
    cnt = 0
    for j in range(0, 5):
        for l in range(0, 5 - j):
            if ((j + 2 * l) ** 2 + j) % p == 0:
                cnt += comb(4, j) * 2 ** j * comb(4 - j, l)
    return cnt

def doubling_float():
    import numpy as np
    s = [np.array([[0, 1], [1, 0]], dtype=complex), np.array([[0, -1j], [1j, 0]]), np.array([[1, 0], [0, -1]], dtype=complex)]
    Z = np.zeros((2, 2)); I2 = np.eye(2)
    g = [np.block([[Z, -1j * s[k]], [1j * s[k], Z]]) for k in range(3)] + [np.block([[Z, I2], [I2, Z]])]
    rows = []
    for npi in range(5):
        k = [math.pi] * npi + [0.0] * (4 - npi)
        out = []
        for r in (0, 1):
            Dk = sum(1j * math.sin(k[m]) * g[m] + r * (1 - math.cos(k[m])) * np.eye(4) for m in range(4))
            out.append(float(np.linalg.svd(Dk, compute_uv=False).min()))
        rows.append((npi, math.comb(4, npi), out[0], out[1]))
    return rows


def main():
    print("13. photon mass term")
    m = mass_a_b_c()
    print("   (a) a' = g(x) (+) a (+) (-g(x+mu)) matches U' = g U g^-1:", {k: f"{v[0]}/{v[1]}" for k, v in m.items() if k in ("Q", "F_7", "F_11")})
    print("   (b) single link, f(U) = 1 - Re U, over F_p exhaustively over (a, g1, g2):")
    for (p, orb, inv, tot, always, npts) in m["Fp"]:
        print(f"       F_{p}: orbit of every link is all {p+1} points: {orb};  f(a') = f(a) in {inv}/{tot} triples;  links with f invariant under ALL g: {always}/{npts}")
    print(f"       Q random: f(a') = f(a) in {m['Qinv'][0]}/{m['Qinv'][1]} triples")
    print("   (c) [f(theta') - f(theta)] / (eps^2/2)  versus  -2 A dphi + dphi^2 (lecture's (A - dphi)^2 - A^2), A = 0.7, dphi = 0.4:")
    for e, v, t in mass_c_continuum():
        print(f"       eps = {e:.6f}   lattice {v:.6f}   continuum {t:.6f}   diff {abs(v - t):.2e}")
    n_cfg, sizes, orbit, norb, nhol, pred = mass_d_orbits(3)
    print(f"   (d) 2x2 torus over F_3[i] (N = 4): {n_cfg} link configurations, gauge-orbit sizes {sizes} (predicted N^(V-1) = {orbit}),")
    print(f"       orbits = {norb} ;  distinct holonomy data (4 plaquettes + 2 non-contractible loops) = {nhol} ;  N^(E-V+1) = {pred}")

    print("\n14. Wilson-Dirac matter field over K = F[i]")
    for label, R in (("Q", RQ), ("F_7", RF(7)), ("F_11", RF(11))):
        ok, tot, herm, sq5, h5, ac5, g, g5 = clifford_checks(R)
        print(f"   {label:<5} Clifford {{g_mu, g_nu}} = 2 delta: {ok}/{tot}; gammas hermitian: {herm}; g5^2 = 1: {sq5}; g5 hermitian: {h5}; g5 anticommutes: {ac5}")
    for label, R in (("Q", RQ), ("F_7", RF(7)), ("F_11", RF(11))):
        res = dirac_checks(R, label)
        for r in (0, 1):
            a = res[(r, 1)]
            print(f"   {label:<5} r = {r}: covariance {a['cov']}/81  psibar D psi invariant {a['inv']}/81  psibar psi invariant {a['mass']}/81  "
                  f"g5-hermiticity {a['g5herm']}/3  current identity {a['cur']}/81  j invariant {a['jinv']}/324")
            print(f"            free operator (U = 1): covariant under global g {a['free_global']}/81, under local g(x) {a['free_local']}/81;   charge 2: {res[(r, 2)]['cov']}/81, {res[(r, 2)]['inv']}/81;  charge 3: {res[(r, 3)]['cov']}/81, {res[(r, 3)]['inv']}/81")
        print(f"   {label:<5} constant fields, U = 1: j_mu = -psibar g_mu psi in {res['const']}/4 directions")
    for label, R in (("Q", RQ), ("F_7", RF(7)), ("F_11", RF(11))):
        sg_ok, sg_tot = spin_generator_check(R)
        print(f"   {label:<5} spin generators [g_mu, g_nu]/4 block diagonal and trace free: {sg_ok}/{sg_tot};   chiral symmetry g5 D g5 = -D at m = 0:  r = 0: {chiral_check(R, 0)}/81,  r = 1: {chiral_check(R, 1)}/81")
    print("   exact doubling on the L = 4 torus (m = 0): momenta with a zero mode / total zero modes")
    for label, R in (("Q", RQ), ("F_3", RF(3)), ("F_7", RF(7)), ("F_11", RF(11)), ("F_19", RF(19)), ("F_23", RF(23))):
        c0, n0, _ = doubling_exact(R, 0); c1, n1, acc = doubling_exact(R, 1)
        pos = doubling_check_position_space(R, 1)
        pr = 1 if R is RQ else doubling_predicted(R.p)
        print(f"      {label:<5} r = 0: {c0} momenta, {n0} modes;   r = 1: {c1} momenta, {n1} modes (formula predicts {pr});  position-space check {pos[0]}/{pos[1]}")
    print("   floating point, corners of the Brillouin zone (smallest singular value of D(k)):")
    for npi, mult, s0, s1 in doubling_float():
        print(f"      n_pi = {npi} ({mult} corners): r = 0: {s0:.3f}   r = 1: {s1:.3f}")

main()
