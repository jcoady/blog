"""
Calculations 15 and 16 (Part 4).

  15. Reconstruction of Cayley link parameters from Cayley face parameters on an open 3D box.
      (a) the cube identity has SIX faces (three differences); the three-face statement a_xy (+) a_yz (+) a_zx = 0 is false
      (b) axial-gauge reconstruction recovers every face exactly when the cube identity holds (Q, F_7, F_11)
      (c) face data violating the cube identity is not recovered
      (d) on a torus the total flux of each plane must vanish (Stokes); random face data violates it
  16. A group-valued Yee update  B <- B (-) n curl E,  E <- E (+) n curl B,  B <- B (-) n curl E  (integer n):  a negative result
      (a) reversibility and the Gauss law hold equally for plain addition mod 7: the checks do not test the Cayley structure
      (b) the real-valued linear scheme with the same update: wave speed and stability versus dt

Needs numpy.  Runs in about a minute.
"""
import itertools
import random
from fractions import Fraction as Fr
import numpy as np


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
        def __rtruediv__(s, o): return FpP(_v(o) * pow(s.v, -1, FpP.P))
        def __eq__(s, o): return o is not None and s.v == _v(o) % FpP.P
        def __hash__(s): return hash(s.v)
        def __repr__(s): return str(s.v)
    return FpP

def _v(o): return o.v if hasattr(o, "v") else int(o)
def kmul(u, v): return (u[0] * v[0] - u[1] * v[1], u[0] * v[1] + u[1] * v[0])
def kpow(u, q, one, zero):
    r = (one, zero)
    for _ in range(q): r = kmul(r, u)
    return r

class Ring:
    """P^1(F) with the Cayley group law  a (+) b = (a + b)/(1 - ab); a = None means infinity."""
    def __init__(s, mk, p=None):
        s.mk, s.p, s.one, s.zero = mk, p, mk(1), mk(0)
    def phase(s, a):
        if a is None: return (-s.one, s.zero)
        d = s.one + a * a
        return ((s.one - a * a) / d, s.mk(-2) * a / d)
    def a_of(s, u): return None if u[0] == -s.one else -u[1] / (s.one + u[0])
    def oplus(s, a, b): return s.a_of(kmul(s.phase(a), s.phase(b)))
    def neg(s, a): return None if a is None else -a
    def ominus(s, a, b): return s.oplus(a, s.neg(b))
    def scale(s, n, a):                                  # n-fold (+), n an integer
        if n < 0: return s.neg(s.scale(-n, a))
        r = s.zero
        for _ in range(n): r = s.oplus(r, a)
        return r
    def eq(s, a, b): return (a is None and b is None) or (a is not None and b is not None and a == b)

class AddRing:
    """Z_p with plain addition: no Cayley structure at all (used to show that some checks are tautological)."""
    def __init__(s, p): s.p, s.zero = p, 0
    def oplus(s, a, b): return (a + b) % s.p
    def ominus(s, a, b): return (a - b) % s.p
    def neg(s, a): return (-a) % s.p
    def scale(s, n, a): return (n * a) % s.p
    def eq(s, a, b): return a == b

RQ = Ring(Fr)
def RF(p): return Ring(make_Fp(p), p)

def sampler(R, rng):
    if R is RQ: return lambda: None if rng.random() < 0.04 else Fr(rng.randint(-6, 6), rng.randint(1, 4))
    return lambda: None if rng.randrange(R.p + 1) == R.p else R.mk(rng.randrange(R.p))
def sampler_finite(R, rng):                              # no infinity
    if R is RQ: return lambda: Fr(rng.randint(-6, 6), rng.randint(1, 4))
    return lambda: R.mk(rng.randrange(R.p))


# ======================================================================
#  15.  RECONSTRUCTION OF LINKS FROM FACES (open 3D box)
# ======================================================================
def make_links(N, rnd): return {(i, j, k, m): rnd() for i in range(N + 1) for j in range(N + 1) for k in range(N + 1) for m in range(3)}
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
    """[a_xy(k+1) (-) a_xy(k)] (+) [a_yz(i+1) (-) a_yz(i)] (+) [a_zx(j+1) (-) a_zx(j)]  (all six faces of the cube)."""
    d = lambda f, p, q: R.ominus(F[f][p], F[f][q])
    return R.oplus(R.oplus(d('xy', (i, j, k + 1), (i, j, k)), d('yz', (i + 1, j, k), (i, j, k))), d('zx', (i, j + 1, k), (i, j, k)))

def reconstruct(R, F, N):
    """Axial gauge a_x = 0.   a_y(i+1) = a_y(i) (+) P_xy(i);   a_z(i+1) = a_z(i) (-) P_zx(i);
    on the slice i = 0:  a_z = 0 and a_y(0,j,k+1) = a_y(0,j,k) (-) P_yz(0,j,k)."""
    a = {(i, j, k, m): R.zero for i in range(N + 1) for j in range(N + 1) for k in range(N + 1) for m in range(3)}
    for j in range(N):
        for k in range(N): a[(0, j, k + 1, 1)] = R.ominus(a[(0, j, k, 1)], F['yz'][(0, j, k)])
    for i in range(N):
        for j in range(N):
            for k in range(N + 1): a[(i + 1, j, k, 1)] = R.oplus(a[(i, j, k, 1)], F['xy'][(i, j, k)])
        for j in range(N + 1):
            for k in range(N): a[(i + 1, j, k, 2)] = R.ominus(a[(i, j, k, 2)], F['zx'][(i, j, k)])
    return a

def match(R, F, a, N):
    G2 = faces_from_links(R, a, N)
    return {n: (sum(R.eq(G2[n][p], F[n][p]) for p in F[n]), len(F[n])) for n in F}

def torus_flux_checks(R, L=4, seed=7):
    """On an L x L torus the (+)-sum of all plaquettes is zero for faces coming from links, but not for random faces."""
    rng = random.Random(seed); rnd = sampler(R, rng); fin = sampler_finite(R, rng)
    a = {(i, j, m): rnd() for i in range(L) for j in range(L) for m in range(2)}
    def plaq(i, j):
        return R.oplus(R.oplus(R.oplus(a[(i, j, 0)], a[((i + 1) % L, j, 1)]), R.neg(a[(i, (j + 1) % L, 0)])), R.neg(a[(i, j, 1)]))
    tot_links = R.zero
    for i in range(L):
        for j in range(L): tot_links = R.oplus(tot_links, plaq(i, j))
    tot_rand = R.zero
    for i in range(L):
        for j in range(L): tot_rand = R.oplus(tot_rand, fin())
    return tot_links, tot_rand


# ======================================================================
#  16.  A GROUP-VALUED YEE UPDATE (negative result)
# ======================================================================
def curl_E(R, Ex, Ey, i, j, L):
    return R.oplus(R.oplus(R.oplus(Ex[(i, j)], Ey[((i + 1) % L, j)]), R.neg(Ex[(i, (j + 1) % L)])), R.neg(Ey[(i, j)]))
def curl_B_x(R, B, i, j, L): return R.ominus(B[(i, j)], B[(i, (j - 1) % L)])
def curl_B_y(R, B, i, j, L): return R.ominus(B[((i - 1) % L, j)], B[(i, j)])
def div_E(R, Ex, Ey, i, j, L):
    return R.oplus(R.oplus(R.ominus(Ex[(i, j)], Ex[((i - 1) % L, j)]), Ey[(i, j)]), R.neg(Ey[(i, (j - 1) % L)]))

def yee_step(R, Ex, Ey, B, n, L):
    """B advanced by n twice and E by n once (the 'half-steps' use the full n)."""
    P = list(itertools.product(range(L), repeat=2))
    B1 = {(i, j): R.ominus(B[(i, j)], R.scale(n, curl_E(R, Ex, Ey, i, j, L))) for i, j in P}
    Ex1 = {(i, j): R.oplus(Ex[(i, j)], R.scale(n, curl_B_x(R, B1, i, j, L))) for i, j in P}
    Ey1 = {(i, j): R.oplus(Ey[(i, j)], R.scale(n, curl_B_y(R, B1, i, j, L))) for i, j in P}
    B2 = {(i, j): R.ominus(B1[(i, j)], R.scale(n, curl_E(R, Ex1, Ey1, i, j, L))) for i, j in P}
    return Ex1, Ey1, B2

def yee_harness(R, rnd, L=4):
    P = list(itertools.product(range(L), repeat=2))
    Ex = {k: rnd() for k in P}; Ey = {k: rnd() for k in P}; B = {k: rnd() for k in P}
    s1 = yee_step(R, Ex, Ey, B, 1, L); s0 = yee_step(R, *s1, -1, L)
    rev = all(R.eq(s0[0][k], Ex[k]) and R.eq(s0[1][k], Ey[k]) and R.eq(s0[2][k], B[k]) for k in P)
    g0 = {k: div_E(R, Ex, Ey, *k, L) for k in P}
    n = (Ex, Ey, B)
    for _ in range(3): n = yee_step(R, *n, 1, L)
    gauss = all(R.eq(g0[k], div_E(R, n[0], n[1], *k, L)) for k in P)
    tot = R.zero
    for k in P: tot = R.oplus(tot, B[k])
    return rev, gauss, tot

def real_step_matrix(L, dt, half):
    """Real-valued linear Yee step.  half = False: B advanced by dt twice (as in the group-valued update);
    half = True: standard Strang step, B advanced by dt/2 twice."""
    n = 3 * L * L; M = np.zeros((n, n))
    def step(v):
        Ex = v[:L * L].reshape(L, L).copy(); Ey = v[L * L:2 * L * L].reshape(L, L).copy(); B = v[2 * L * L:].reshape(L, L).copy()
        curlE = lambda Ex, Ey: Ex + np.roll(Ey, -1, axis=0) - np.roll(Ex, -1, axis=1) - Ey
        h = dt / 2 if half else dt
        B = B - h * curlE(Ex, Ey)
        Ex = Ex + dt * (B - np.roll(B, 1, axis=1)); Ey = Ey + dt * (np.roll(B, 1, axis=0) - B)
        B = B - h * curlE(Ex, Ey)
        return np.concatenate([Ex.ravel(), Ey.ravel(), B.ravel()])
    for c in range(n):
        e = np.zeros(n); e[c] = 1; M[:, c] = step(e)
    return M


def main():
    print("15. reconstruction of links from faces, open 3D box N = 3 (36 faces per plane, 27 cubes)")
    N = 3
    for label, R in (("Q", RQ), ("F_7", RF(7)), ("F_11", RF(11))):
        rng = random.Random(3); rnd = sampler(R, rng)
        a = make_links(N, rnd); F = faces_from_links(R, a, N)
        cubes = list(itertools.product(range(N), repeat=3))
        ok6 = sum(R.eq(cube_residual(R, F, *c), R.zero) for c in cubes)
        ok3 = sum(R.eq(R.oplus(R.oplus(F['xy'][c], F['yz'][c]), F['zx'][c]), R.zero) for c in cubes)
        m = match(R, F, reconstruct(R, F, N), N)
        print(f"   {label:<5} six-face cube identity {ok6}/27;  three-face a_xy (+) a_yz (+) a_zx = 0: {ok3}/27;  "
              f"recovered faces from Bianchi-satisfying data: xy {m['xy'][0]}/{m['xy'][1]}, yz {m['yz'][0]}/{m['yz'][1]}, zx {m['zx'][0]}/{m['zx'][1]}")
        rng2 = random.Random(4); fin = sampler_finite(R, rng2); fr = face_ranges(N)
        Fbad = {n: {p: fin() for p in fr[n]} for n in fr}
        bad = sum(R.eq(cube_residual(R, Fbad, *c), R.zero) for c in cubes)
        m = match(R, Fbad, reconstruct(R, Fbad, N), N)
        print(f"         random faces violating the identity (holds on {bad}/27 cubes): recovered xy {m['xy'][0]}/36, yz {m['yz'][0]}/36, zx {m['zx'][0]}/36")
        tl, tr = torus_flux_checks(R)
        print(f"         4x4 torus, total flux: faces from links = {tl} (must be 0);  random faces = {tr}")
    print("\n16. group-valued Yee update, 2+1 D torus L = 4, integer step n = 1")
    for label, R in (("Q", RQ), ("F_7", RF(7))):
        rng = random.Random(17); rnd = sampler_finite(R, rng)
        rev, gauss, tot = yee_harness(R, rnd)
        print(f"   Cayley {label:<4}: reversibility {rev}, Gauss after 3 steps {gauss}, total flux of the random initial B = {tot}")
    Z = AddRing(7); rng = random.Random(17); rnd = lambda: rng.randrange(7)
    rev, gauss, tot = yee_harness(Z, rnd)
    print(f"   plain Z_7 addition (no Cayley structure): reversibility {rev}, Gauss after 3 steps {gauss}")
    L = 24; dt = 0.05; k = 2 * np.pi / L
    for half, name in ((False, "as above (B advanced by dt twice)"), (True, "standard Strang (B advanced by dt/2 twice)")):
        ev = np.linalg.eigvals(real_step_matrix(L, dt, half))
        w = np.sort(np.abs(np.angle(ev)) / dt); w = w[w > 1e-9]
        print(f"   real linear scheme, {name}: lowest wave frequency / k = {w[0] / k:.4f}")
    for half, name in ((False, "as above"), (True, "standard")):
        print("   max |eigenvalue| per step, " + name + ":  " + ",  ".join(
            f"dt = {dt_}: {np.max(abs(np.linalg.eigvals(real_step_matrix(8, dt_, half)))):.3f}" for dt_ in (0.5, 0.7, 0.9, 1.0)))

main()
