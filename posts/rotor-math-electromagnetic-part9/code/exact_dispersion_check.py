"""
Exact (rational-arithmetic) checks for the coupled scalar-gauge kick-Cayley-drift-kick scheme of Parts 7-8.

  (0) cross-check: this exact implementation equals the floating-point vectorised step of Part 8 (L = 3, random state)
  (1) plane-wave check over Q(i): the one-step map linearised at the vacuum, acting on a plane wave, equals the predicted
      matrix; determinant 1; trace 2 - h^2 Omega^2; minimal polynomial; decoupling of the gauge and scalar sectors
  (2) consistency in h: Taylor coefficients in a formal step size h of the full nonlinear step, against the exact flow,
      at a random rational state; Gauss function and energy as series in h

All numbers are exact rationals (Gaussian rationals Q(i) with truncated power series).  No angle, exp or arctan is used
inside (1) and (2).  Needs numpy (only for roll on object arrays) and, for (0), validate_coupled_scheme.py from Part 8.
Tested with Python 3.12.3, NumPy 2.4.4.
"""
import random
import numpy as np
from fractions import Fraction as Fr


# ---------- Gaussian rationals Q(i) ----------
class G:
    __slots__ = ("re", "im")
    def __init__(self, re=0, im=0): self.re = Fr(re); self.im = Fr(im)
    @staticmethod
    def co(o):
        if isinstance(o, G): return o
        if isinstance(o, (int, Fr)): return G(o)
        return None
    def __add__(s, o):
        o = G.co(o)
        return NotImplemented if o is None else G(s.re + o.re, s.im + o.im)
    __radd__ = __add__
    def __sub__(s, o):
        o = G.co(o)
        return NotImplemented if o is None else G(s.re - o.re, s.im - o.im)
    def __rsub__(s, o):
        o = G.co(o)
        return NotImplemented if o is None else o - s
    def __neg__(s): return G(-s.re, -s.im)
    def __mul__(s, o):
        o = G.co(o)
        return NotImplemented if o is None else G(s.re * o.re - s.im * o.im, s.re * o.im + s.im * o.re)
    __rmul__ = __mul__
    def inv(s): n = s.re ** 2 + s.im ** 2; return G(s.re / n, -s.im / n)
    def __truediv__(s, o):
        o = G.co(o)
        return NotImplemented if o is None else s * o.inv()
    def __rtruediv__(s, o):
        o = G.co(o)
        return NotImplemented if o is None else o * s.inv()
    def conj(s): return G(s.re, -s.im)
    def __eq__(s, o):
        o = G.co(o)
        return NotImplemented if o is None else (s.re == o.re and s.im == o.im)
    def __hash__(s): return hash((s.re, s.im))
    def iszero(s): return s.re == 0 and s.im == 0
    def mag(s): return max(abs(float(s.re)), abs(float(s.im)))
    def __repr__(s): return f"({s.re}+{s.im}i)"

I = G(0, 1)


# ---------- truncated power series over Q(i):  c[0] + c[1] x + ... + c[ORDER] x^ORDER ----------
ORDER = 1
class S:
    __slots__ = ("c",)
    def __init__(self, c=0):
        self.c = c if isinstance(c, list) else [G.co(c)] + [G() for _ in range(ORDER)]
    @staticmethod
    def co(o):
        if isinstance(o, S): return o
        if isinstance(o, (G, int, Fr)): return S(o)
        return None
    def __add__(s, o):
        o = S.co(o)
        return NotImplemented if o is None else S([a + b for a, b in zip(s.c, o.c)])
    __radd__ = __add__
    def __sub__(s, o):
        o = S.co(o)
        return NotImplemented if o is None else S([a - b for a, b in zip(s.c, o.c)])
    def __rsub__(s, o):
        o = S.co(o)
        return NotImplemented if o is None else o - s
    def __neg__(s): return S([-a for a in s.c])
    def __mul__(s, o):
        if not isinstance(o, S):
            o = G.co(o)
            return NotImplemented if o is None else S([a * o for a in s.c])
        c = [G() for _ in range(ORDER + 1)]
        for i, a in enumerate(s.c):
            if a.iszero(): continue
            for j, b in enumerate(o.c):
                if i + j > ORDER: break
                c[i + j] = c[i + j] + a * b
        return S(c)
    __rmul__ = __mul__
    def inv(s):
        b = [s.c[0].inv()]
        for n in range(1, ORDER + 1):
            acc = G()
            for k in range(1, n + 1): acc = acc + s.c[k] * b[n - k]
            b.append(-(acc * b[0]))
        return S(b)
    def __truediv__(s, o):
        if isinstance(o, S): return s * o.inv()
        o = G.co(o)
        return NotImplemented if o is None else s * o.inv()
    def __rtruediv__(s, o):
        o = S.co(o)
        return NotImplemented if o is None else o * s.inv()
    def integ(s): return S([G()] + [s.c[n] * Fr(1, n + 1) for n in range(ORDER)])


def amap(f, a):
    out = np.empty(a.shape, dtype=object)
    for idx in np.ndindex(a.shape): out[idx] = f(a[idx])
    return out

def roll(a, k, ax): return np.roll(a, k, axis=ax)


# ---------- the scheme (complexified: U, Ub = 1/U, phi, phib = conj(phi), pi, pib = conj(pi) independent) ----------
# state = [U0, U1, Ub0, Ub1, E0, E1, phi, phib, pi, pib]; each an array of S

def forces(st, m2):
    U, Ub, E = st[0:2], st[2:4], st[4:6]; phi, phib, pi, pib = st[6:10]
    plaq = U[0] * roll(U[1], -1, 0) * roll(Ub[0], -1, 1) * Ub[1]
    plaqb = Ub[0] * roll(Ub[1], -1, 0) * roll(U[0], -1, 1) * U[1]
    xi = (plaq - plaqb) * G(0, Fr(-1, 2))                                  # Im(plaq) = (plaq - conj plaq)/(2i)
    dE = [xi - roll(xi, 1, 1), -(xi - roll(xi, 1, 0))]
    dpi = m2 * phib; dpib = m2 * phi
    for mu in (0, 1):
        z = U[mu] * roll(phi, -1, mu) * phib; zb = Ub[mu] * roll(phib, -1, mu) * phi
        dE[mu] = dE[mu] + (z - zb) * G(0, -1)                              # 2 Im z = (z - zb)/i
        dpi = dpi + 2 * phib - roll(U[mu], 1, mu) * roll(phib, 1, mu) - Ub[mu] * roll(phib, -1, mu)
        dpib = dpib + 2 * phi - roll(Ub[mu], 1, mu) * roll(phi, 1, mu) - U[mu] * roll(phi, -1, mu)
    return dE, dpi, dpib

def kick(st, h, m2):
    dE, dpi, dpib = forces(st, m2)
    return st[0:4] + [st[4] - h * dE[0], st[5] - h * dE[1], st[6], st[7], st[8] - h * dpi, st[9] - h * dpib]

def drift(st, h):
    E = st[4:6]; out = list(st)
    for mu in (0, 1):
        c = h * E[mu] * Fr(1, 2)
        r = (1 + I * c) / (1 - I * c)                                      # exp(2i arctan(hE/2)), exactly
        out[mu] = st[mu] * r; out[2 + mu] = st[2 + mu] / r
    out[6] = st[6] + h * st[9]; out[7] = st[7] + h * st[8]                 # phi += h conj(pi), and its conjugate
    return out

def step(st, h, m2):
    st = kick(st, h * Fr(1, 2), m2); st = drift(st, h); return kick(st, h * Fr(1, 2), m2)

def vector_field(st, m2):
    """Continuous-time equations (the h -> 0 limit of the step)."""
    dE, dpi, dpib = forces(st, m2); E = st[4:6]
    return [I * st[0] * E[0], I * st[1] * E[1], -I * st[2] * E[0], -I * st[3] * E[1], -dE[0], -dE[1], st[9], st[8], -dpi, -dpib]

def gauss(st):
    E, phi, phib, pi, pib = st[4:6], st[6], st[7], st[8], st[9]
    return sum(E[mu] - roll(E[mu], 1, mu) for mu in (0, 1)) + (pi * phi - pib * phib) * I       # div E - 2 Im(pi phi)

def energy(st, m2):
    U, Ub, E = st[0:2], st[2:4], st[4:6]; phi, phib, pi, pib = st[6:10]
    plaq = U[0] * roll(U[1], -1, 0) * roll(Ub[0], -1, 1) * Ub[1]
    plaqb = Ub[0] * roll(Ub[1], -1, 0) * roll(U[0], -1, 1) * U[1]
    H = sum(np.sum(E[mu] * E[mu]) * Fr(1, 2) for mu in (0, 1)) + np.sum(1 - (plaq + plaqb) * Fr(1, 2))
    H = H + np.sum(pi * pib) + m2 * np.sum(phi * phib)
    for mu in (0, 1): H = H + np.sum((U[mu] * roll(phi, -1, mu) - phi) * (Ub[mu] * roll(phib, -1, mu) - phib))
    return H


# ---------- (0) cross-check against the floating-point step of Part 8 ----------
def cross_check():
    global ORDER
    from validate_coupled_scheme import step as fstep
    ORDER = 0; rng = random.Random(1); L = 3; m2 = 1; h = Fr(13, 100)
    def rnd(): return Fr(rng.randint(-9, 9), 11)
    a = [[[rnd() for _ in range(L)] for _ in range(L)] for _ in range(2)]
    Ev = [[[rnd() for _ in range(L)] for _ in range(L)] for _ in range(2)]
    ph = [[G(rnd(), rnd()) for _ in range(L)] for _ in range(L)]; pv = [[G(rnd(), rnd()) for _ in range(L)] for _ in range(L)]
    def arr(f): 
        o = np.empty((L, L), dtype=object)
        for i in range(L):
            for j in range(L): o[i, j] = S(f(i, j))
        return o
    ca = lambda mu, i, j: G(a[mu][i][j])
    U = [arr(lambda i, j, mu=mu: (1 - I * ca(mu, i, j)) / (1 + I * ca(mu, i, j))) for mu in (0, 1)]
    Ub = [arr(lambda i, j, mu=mu: (1 + I * ca(mu, i, j)) / (1 - I * ca(mu, i, j))) for mu in (0, 1)]
    E = [arr(lambda i, j, mu=mu: G(Ev[mu][i][j])) for mu in (0, 1)]
    st = U + Ub + E + [arr(lambda i, j: ph[i][j]), arr(lambda i, j: ph[i][j].conj()), arr(lambda i, j: pv[i][j]), arr(lambda i, j: pv[i][j].conj())]
    out = step(st, h, m2)
    cf = lambda x: complex(float(x.c[0].re), float(x.c[0].im))
    toc = lambda A: np.array([[cf(A[i, j]) for j in range(L)] for i in range(L)])
    th = np.array([[[-2 * np.arctan(float(a[mu][i][j])) for j in range(L)] for i in range(L)] for mu in range(2)])
    Ef = np.array([[[float(Ev[mu][i][j]) for j in range(L)] for i in range(L)] for mu in range(2)])
    phif = np.array([[complex(float(ph[i][j].re), float(ph[i][j].im)) for j in range(L)] for i in range(L)])
    pif = np.array([[complex(float(pv[i][j].re), float(pv[i][j].im)) for j in range(L)] for i in range(L)])
    th2, E2, phi2, pi2 = fstep((th, Ef, phif, pif), float(h), float(m2))
    err = 0.0
    for mu in (0, 1):
        err = max(err, abs(np.exp(1j * th2[mu]) - toc(out[mu])).max(), abs(E2[mu] - toc(out[4 + mu])).real.max())
    err = max(err, abs(phi2 - toc(out[6])).max(), abs(pi2 - toc(out[8])).max())
    return err


# ---------- (1) exact plane-wave check ----------
def mm(A, B): return np.dot(A, B)
def eye(n): return np.array([[G(1) if i == j else G() for j in range(n)] for i in range(n)], dtype=object)
def mat(rows): return np.array(rows, dtype=object)
def zero_matrix(M): return all(x.iszero() for x in M.flatten())
def scal(c, M): return amap(lambda x: x * c, M)

def spec(t): return Fr(4) * t * t / (1 + t * t)                            # 4 sin^2(k/2) = 4 t^2/(1+t^2), t = tan(k/2)

def plane_wave_matrix(t0, t1, h, m2, P=9, margin=3):
    """One-step linear map at the vacuum on the amplitude vector (theta0, theta1, E0, E1, phi, chi = conj(pi))."""
    global ORDER
    ORDER = 1
    z = [(1 + I * t) / (1 - I * t) for t in (t0, t1)]                      # exp(i k) as a Gaussian rational
    pw = np.empty((P, P), dtype=object)
    for i in range(P):
        for j in range(P):
            v = G(1)
            for _ in range(i): v = v * z[0]
            for _ in range(j): v = v * z[1]
            pw[i, j] = v
    sites = [(i, j) for i in range(margin, P - margin) for j in range(margin, P - margin)]
    def field(c0, amp):                                                     # c0 + eps * amp * plane wave
        o = np.empty((P, P), dtype=object)
        for i in range(P):
            for j in range(P): o[i, j] = S([G.co(c0), amp * pw[i, j]])
        return o
    cols = []
    for k in range(6):
        v = [G(1) if q == k else G() for q in range(6)]                    # one-hot perturbation
        st = [field(1, I * v[0]), field(1, I * v[1]), field(1, -I * v[0]), field(1, -I * v[1]),
              field(0, v[2]), field(0, v[3]), field(0, v[4]), field(0, G()), field(0, G()), field(0, v[5])]
        out = step(st, h, m2)
        col = None
        for (i, j) in sites:
            w = pw[i, j]
            vec = [out[0][i, j].c[1] / (I * w), out[1][i, j].c[1] / (I * w), out[4][i, j].c[1] / w, out[5][i, j].c[1] / w,
                   out[6][i, j].c[1] / w, out[9][i, j].c[1] / w]
            assert out[7][i, j].c[1].iszero() and out[8][i, j].c[1].iszero()           # conjugate sector stays at rest
            assert (out[2][i, j].c[1] + out[0][i, j].c[1]).iszero() and (out[3][i, j].c[1] + out[1][i, j].c[1]).iszero()  # Ub = 1/U
            assert all(out[q][i, j].c[0] == (1 if q < 4 else 0) for q in (0, 1, 2, 3, 4, 5, 6, 9))    # vacuum is fixed
            if col is None: col = vec
            else: assert all(a == b for a, b in zip(col, vec)), "output is not a plane wave"
        cols.append(col)
    return mat(cols).T, z

def predicted_matrix(z, h, m2):
    d = [zz - 1 for zz in z]
    K = mat([[d[1] * d[1].conj(), -(d[1].conj() * d[0])], [-(d[0].conj() * d[1]), d[0] * d[0].conj()]])
    h2 = h * h; Om2 = (d[0] * d[0].conj() + d[1] * d[1].conj())
    A = eye(2) - scal(h2 * Fr(1, 2), K)
    B = scal(-h, K) + scal(h2 * h * Fr(1, 4), mm(K, K))
    gauge = np.block([[A, scal(h, eye(2))], [B, A]])
    Lam = m2 + Om2
    scalar = mat([[1 - h2 * Lam * Fr(1, 2), G(h)], [-h * Lam + h2 * h * Lam * Lam * Fr(1, 4), 1 - h2 * Lam * Fr(1, 2)]])
    S6 = np.empty((6, 6), dtype=object)
    for a in range(6):
        for b in range(6): S6[a, b] = G()
    S6[:4, :4] = gauge; S6[4:, 4:] = scalar
    return S6, d, Om2, Lam

def check_case(t0, t1, h, m2):
    M, z = plane_wave_matrix(t0, t1, h, m2)
    Mp, d, Om2, Lam = predicted_matrix(z, h, m2)
    ok_pred = all(a == b for a, b in zip(M.flatten(), Mp.flatten()))
    Om2q = spec(t0) + spec(t1)
    ok_om = (Om2 == Om2q)
    Sg = M[:4, :4]; Ss = M[4:, 4:]
    tau_g = 2 - h * h * Om2q; tau_s = 2 - h * h * (m2 + Om2q)
    Id4 = eye(4); Id2 = eye(2)
    # physical gauge mode: theta, E both along w = (conj d1, -conj d0); two-dimensional invariant subspace
    w = [d[1].conj(), -d[0].conj()]
    uth = np.array([w[0], w[1], G(), G()], dtype=object)
    uE = np.array([G(), G(), w[0], w[1]], dtype=object)
    a1 = mm(Sg, uth); a2 = mm(Sg, uE)
    Pm = [[1 - h * h * Om2q * Fr(1, 2), G(h)], [-h * Om2q + h ** 3 * Om2q * Om2q * Fr(1, 4), 1 - h * h * Om2q * Fr(1, 2)]]
    inv_sub = all(a1[q] == Pm[0][0] * uth[q] + Pm[1][0] * uE[q] and a2[q] == Pm[0][1] * uth[q] + Pm[1][1] * uE[q] for q in range(4))
    det_g = (Pm[0][0] * Pm[1][1] - Pm[0][1] * Pm[1][0]) == 1
    tr_g = (Pm[0][0] + Pm[1][1]) == tau_g
    N = mm(mm(Sg - Id4, Sg - Id4), mm(Sg, Sg) - scal(tau_g, Sg) + Id4)
    minpoly_g = zero_matrix(N)
    det_s = (Ss[0, 0] * Ss[1, 1] - Ss[0, 1] * Ss[1, 0]) == 1
    tr_s = (Ss[0, 0] + Ss[1, 1]) == tau_s
    ch_s = zero_matrix(mm(Ss, Ss) - scal(tau_s, Ss) + Id2)
    ok = ok_pred and ok_om and inv_sub and det_g and tr_g and minpoly_g and det_s and tr_s and ch_s
    return ok, Om2q, m2 + Om2q, tau_g, tau_s, dict(pred=ok_pred, omega=ok_om, inv=inv_sub, det_g=det_g, tr_g=tr_g, minpoly=minpoly_g, det_s=det_s, tr_s=tr_s, ch=ch_s)


# ---------- (2) consistency in h ----------
def random_state(L, rng):
    def rnd(): return Fr(rng.randint(-9, 9), 11)
    def arr(f):
        o = np.empty((L, L), dtype=object)
        for i in range(L):
            for j in range(L): o[i, j] = S(f())
        return o
    a = [[[rnd() for _ in range(L)] for _ in range(L)] for _ in range(2)]
    U, Ub, E = [], [], []
    for mu in (0, 1):
        Um = np.empty((L, L), dtype=object); Ubm = np.empty((L, L), dtype=object)
        for i in range(L):
            for j in range(L):
                ai = G(a[mu][i][j]); Um[i, j] = S((1 - I * ai) / (1 + I * ai)); Ubm[i, j] = S((1 + I * ai) / (1 - I * ai))
        U.append(Um); Ub.append(Ubm); E.append(arr(lambda: G(rnd())))
    phi = np.empty((L, L), dtype=object); pi = np.empty((L, L), dtype=object)
    phib = np.empty((L, L), dtype=object); pib = np.empty((L, L), dtype=object)
    for i in range(L):
        for j in range(L):
            p = G(rnd(), rnd()); q = G(rnd(), rnd())
            phi[i, j] = S(p); phib[i, j] = S(p.conj()); pi[i, j] = S(q); pib[i, j] = S(q.conj())
    return U + Ub + E + [phi, phib, pi, pib]

def series_check(L=3, m2=1, seed=2026):
    global ORDER
    ORDER = 3
    st0 = random_state(L, random.Random(seed))
    hh = S([G(), G(1)] + [G() for _ in range(ORDER - 1)])                  # the formal step size h
    stp = step(st0, hh, m2)
    flow = st0
    for _ in range(ORDER):
        F = vector_field(flow, m2)
        flow = [a + amap(lambda v: v.integ(), f) for a, f in zip(st0, F)]
    res = {}
    sa = [x for arr in stp for x in arr.flatten()]; sb = [x for arr in flow for x in arr.flatten()]
    res["state"] = [max((a - b).c[n].mag() for a, b in zip(sa, sb)) for n in range(ORDER + 1)]
    G0 = gauss(st0); Gs = gauss(stp)
    res["gauss0"] = max(x.c[0].mag() for x in G0.flatten())
    res["gauss"] = [max(((a - b).c[n]).mag() for a, b in zip(Gs.flatten(), G0.flatten())) for n in range(ORDER + 1)]
    H0 = energy(st0, m2); Hs = energy(stp, m2); Hf = energy(flow, m2)
    res["H_step"] = [(Hs - H0).c[n] for n in range(ORDER + 1)]
    res["H_flow"] = [(Hf - H0).c[n] for n in range(ORDER + 1)]
    return res


def main():
    e0 = cross_check()
    print(f"(0) exact step versus floating-point step of Part 8 (L = 3, random state, h = 0.13): max difference {e0:.2e}")
    print("(1) exact plane-wave check over Q(i)  [t = tan(k/2); Omega^2 = sum 4t^2/(1+t^2); Lambda = m^2 + Omega^2]")
    cases = [(Fr(1, 2), Fr(1, 3), Fr(1, 5), Fr(1)), (Fr(2, 5), Fr(0), Fr(1, 2), Fr(1)), (Fr(3, 4), Fr(-1, 2), Fr(1, 4), Fr(1, 2)),
             (Fr(1), Fr(1), Fr(1, 3), Fr(0)), (Fr(2), Fr(1, 3), Fr(1, 4), Fr(2)), (Fr(1), Fr(1), Fr(2), Fr(0))]
    print("    t0     t1     h      m^2    Omega^2    Lambda     tr(gauge)  tr(scalar)  h^2 Omega^2/4  all exact checks")
    for t0, t1, h, m2 in cases:
        ok, Om2, Lam, tg, ts, parts = check_case(t0, t1, h, m2)
        print(f"    {str(t0):6} {str(t1):6} {str(h):6} {str(m2):6} {str(Om2):10} {str(Lam):10} {str(tg):10} {str(ts):11} {str(h*h*Om2/4):14} {'pass' if ok else 'FAIL ' + str(parts)}")
    M, z = plane_wave_matrix(Fr(1, 2), Fr(1, 3), Fr(1, 5), Fr(1))
    ctl = {lab: all(a == b for a, b in zip(M.flatten(), predicted_matrix(z, h_, m2_)[0].flatten()))
           for lab, (h_, m2_) in {"correct prediction": (Fr(1, 5), Fr(1)), "wrong mass (m^2 = 2)": (Fr(1, 5), Fr(2)), "wrong step (h = 1/4)": (Fr(1, 4), Fr(1))}.items()}
    print("    negative control (first row, entrywise equality with a prediction):", ctl)
    print("(2) consistency in h at a random rational state (L = 3, m^2 = 1, all fields nonzero; Gauss law not imposed)")
    r = series_check()
    print("    max |Taylor coefficient of (step - exact flow)|, orders h^0..h^3 :", ["%.3e" % x for x in r["state"]])
    print(f"    Gauss function: initial max|G| = {r['gauss0']:.3f}; max |coefficient of G(step) - G(x0)|, orders h^0..h^3 :", ["%.3e" % x for x in r["gauss"]])
    print("    energy H(step) - H(x0), coefficients h^0..h^3 :", [f"{x.re}" + ("" if x.im == 0 else f"+{x.im}i") if abs(x.re.denominator) < 10**6 else "%.6e" % float(x.re) for x in r["H_step"]])
    print("    energy H(exact flow) - H(x0), coefficients h^0..h^3 :", [str(x.re) for x in r["H_flow"]])


if __name__ == "__main__":
    main()
