"""
Can the rational Cayley U(1) lattice carry dynamics?   A concrete minimal test (2+1D torus).

Variables on every spatial link:  a = Cayley parameter  (U = (1 - i a)/(1 + i a) = e^{i theta}, theta = -2 atan a)
                                  E = electric field (conjugate to theta)
Hamiltonian (g = 1, lattice spacing 1, c = 1):   H = sum E^2/2 + sum_P (1 - cos theta_P)
Equations:  theta' = E ,   E' = -dV/dtheta  = -curl^T sin(theta_P)        (Ampere law, no current)
Gauss law:  div E = rho  is preserved EXACTLY because div(curl^T s) = 0 for any s.

Rational time step (kick - drift - kick):
    kick  : E <- E - (dt/2) curl^T sin(theta_P),   sin(theta_P) = Im( U_x U_y U_x^-1 U_y^-1 )  (rational in the a's)
    drift : a <- a (+) b,  b = -dt E/2,  (+) = tangent addition  (a + b)/(1 - a b)     (i.e. theta += 2 atan(dt E/2))
Everything is rational in (a, E): no exponentials, trig functions or square roots.

Tests
  A  exact arithmetic over Q and over F_7, F_11: Gauss law preserved exactly, map exactly reversible, size growth
  B  floats: (1) plane-wave dispersion  (2) bounded energy  (3) second-order convergence
  C  the map is exactly symplectic in (theta, E) coordinates

Needs numpy.   Colab / terminal: just run it.  main() is called at the bottom.
"""
import itertools
import random
import time
from fractions import Fraction as Fr

import numpy as np


# ======================================================================= A. exact
class Fp:
    p = 7
    __slots__ = ("v",)

    def __init__(s, v): s.v = int(v) % Fp.p
    def __add__(s, o): return Fp(s.v + _v(o))
    __radd__ = __add__
    def __sub__(s, o): return Fp(s.v - _v(o))
    def __neg__(s): return Fp(-s.v)
    def __mul__(s, o): return Fp(s.v * _v(o))
    __rmul__ = __mul__
    def __truediv__(s, o): return Fp(s.v * pow(_v(o), -1, Fp.p))
    def __eq__(s, o): return False if o is None else s.v == _v(o) % Fp.p
    def __hash__(s): return hash(s.v)


def _v(o): return o.v if isinstance(o, Fp) else int(o)


def phase(a, mk):                                  # (c, s) = ((1-a^2)/(1+a^2), -2a/(1+a^2));  a = None is infinity
    if a is None: return (mk(-1), mk(0))
    d = mk(1) + a * a
    return ((mk(1) - a * a) / d, mk(-2) * a / d)


def cmul(u, v): return (u[0] * v[0] - u[1] * v[1], u[0] * v[1] + u[1] * v[0])
def cconj(u): return (u[0], -u[1])


def oplus(a, b, mk):
    if a is None and b is None: return mk(0)
    if a is None: return None if b == 0 else -mk(1) / b
    if b is None: return None if a == 0 else -mk(1) / a
    den = mk(1) - a * b
    return None if den == 0 else (a + b) / den


def sin_plaq(a, L, mk):                            # S[i,j] = Im U_P(i,j), rational in the a's
    S = {}
    for i in range(L):
        for j in range(L):
            u = cmul(cmul(phase(a[(i, j, 0)], mk), phase(a[((i + 1) % L, j, 1)], mk)),
                     cmul(cconj(phase(a[(i, (j + 1) % L, 0)], mk)), cconj(phase(a[(i, j, 1)], mk))))
            S[(i, j)] = u[1]
    return S


def kick(a, E, L, h, mk):                          # E <- E - h curl^T S
    S = sin_plaq(a, L, mk)
    E2 = {}
    for i in range(L):
        for j in range(L):
            E2[(i, j, 0)] = E[(i, j, 0)] - h * (S[(i, j)] - S[(i, (j - 1) % L)])
            E2[(i, j, 1)] = E[(i, j, 1)] + h * (S[(i, j)] - S[((i - 1) % L, j)])
    return E2


def drift(a, E, h, mk):                            # a <- a (+) (-h E / 2)
    return {k: oplus(a[k], -(h * E[k]) / mk(2), mk) for k in a}


def step(a, E, L, dt, mk):
    E = kick(a, E, L, dt / mk(2), mk)
    a = drift(a, E, dt, mk)
    E = kick(a, E, L, dt / mk(2), mk)
    return a, E


def gauss(E, L):
    return {(i, j): E[(i, j, 0)] - E[((i - 1) % L, j, 0)] + E[(i, j, 1)] - E[(i, (j - 1) % L, 1)]
            for i in range(L) for j in range(L)}


def size_bits(x):
    if x is None or isinstance(x, Fp): return 0
    return max(abs(x.numerator).bit_length(), x.denominator.bit_length())


def section_A():
    print("\n" + "-" * 78 + "\nA. exact arithmetic (2D torus L = 3, 18 links, static charges: div E != 0)\n" + "-" * 78)
    L, rng = 3, random.Random(4)
    keys = [(i, j, m) for i in range(L) for j in range(L) for m in (0, 1)]
    # ---- over Q
    a = {k: Fr(rng.randint(-3, 3), rng.randint(1, 3)) for k in keys}
    E = {k: Fr(rng.randint(-3, 3), rng.randint(1, 3)) for k in keys}
    a0, E0, g0 = dict(a), dict(E), gauss(E, L)
    dt = Fr(1, 4)
    print("  Q (Fractions), dt = 1/4:   step   Gauss law unchanged   max bit-length of a, E")
    for n in range(1, 5):
        a, E = step(a, E, L, dt, Fr)
        bits = max(max(size_bits(v) for v in a.values()), max(size_bits(v) for v in E.values()))
        print(f"                             {n:>3}   {str(gauss(E, L) == g0):>19}   {bits:>12}")
    for n in range(4): a, E = step(a, E, L, -dt, Fr)                    # run time backwards
    print(f"  after 4 steps forward and 4 backward the state is recovered exactly: {a == a0 and E == E0}")
    print("  => exact over Q, but the numbers grow roughly fourteen-fold in size per step, so exact rational")
    print("     evolution is practical only for a few steps.")
    # ---- over F_p
    for p in (7, 11):
        Fp.p = p
        mk = lambda x: Fp(x)
        gok = rev = moved = 0
        trials = 200
        for _ in range(trials):
            a = {k: (None if rng.randrange(p + 1) == p else Fp(rng.randrange(p))) for k in keys}   # a in P^1(F_p)
            E = {k: Fp(rng.randrange(p)) for k in keys}
            a0, E0, g0 = dict(a), dict(E), gauss(E, L)
            dt = mk(1) / mk(4)
            ok = True
            for n in range(60):
                a, E = step(a, E, L, dt, mk)
                ok &= gauss(E, L) == g0
            gok += ok
            moved += (a != a0) or (E != E0)
            for n in range(60): a, E = step(a, E, L, -dt, mk)
            rev += (a == a0 and E == E0)
        print(f"  F_{p}: {trials} random states (a includes infinity), 60 steps: Gauss law exact {gok}/{trials};"
              f" state changed {moved}/{trials}; exactly reversible {rev}/{trials}")


# ====================================================================== B. floats
def sinP(ax, ay):
    ux, uy = (1 - ax ** 2 + -2j * ax) / (1 + ax ** 2), (1 - ay ** 2 + -2j * ay) / (1 + ay ** 2)
    UP = ux * np.roll(uy, -1, 0) * np.conj(np.roll(ux, -1, 1)) * np.conj(uy)
    return UP


def kick_f(ax, ay, Ex, Ey, h):
    UP = sinP(ax, ay); S = UP.imag
    return Ex - h * (S - np.roll(S, 1, 1)), Ey + h * (S - np.roll(S, 1, 0))


def cayley_step(ax, ay, Ex, Ey, dt):
    Ex, Ey = kick_f(ax, ay, Ex, Ey, dt / 2)
    bx, by = -dt * Ex / 2, -dt * Ey / 2
    ax, ay = (ax + bx) / (1 - ax * bx), (ay + by) / (1 - ay * by)
    Ex, Ey = kick_f(ax, ay, Ex, Ey, dt / 2)
    return ax, ay, Ex, Ey


def exact_step(thx, thy, Ex, Ey, dt):             # reference: same kick, drift theta += dt E exactly
    def S_of(thx, thy):
        return np.sin(thx + np.roll(thy, -1, 0) - np.roll(thx, -1, 1) - thy)
    S = S_of(thx, thy); Ex = Ex - dt / 2 * (S - np.roll(S, 1, 1)); Ey = Ey + dt / 2 * (S - np.roll(S, 1, 0))
    thx, thy = thx + dt * Ex, thy + dt * Ey
    S = S_of(thx, thy); Ex = Ex - dt / 2 * (S - np.roll(S, 1, 1)); Ey = Ey + dt / 2 * (S - np.roll(S, 1, 0))
    return thx, thy, Ex, Ey


a_of_theta = lambda th: -np.tan(th / 2)
theta_of_a = lambda a: -2 * np.arctan(a)


def energy(ax, ay, Ex, Ey):
    return 0.5 * ((Ex ** 2).sum() + (Ey ** 2).sum()) + (1 - sinP(ax, ay).real).sum()


def section_B():
    print("\n" + "-" * 78 + "\nB. floating point: does it behave like Maxwell theory?\n" + "-" * 78)
    # B1 dispersion of a standing wave  theta_y = eps cos(k x)
    L, n, dt, eps = 64, 4, 0.25, 1e-3
    k = 2 * np.pi * n / L
    x = np.arange(L)[:, None] * np.ones((1, L))
    ax = np.zeros((L, L)); ay = a_of_theta(eps * np.cos(k * x)); Ex = np.zeros((L, L)); Ey = np.zeros((L, L))
    amp = []
    for _ in range(400):
        amp.append((theta_of_a(ay) * np.cos(k * x)).sum() * 2 / L ** 2)
        ax, ay, Ex, Ey = cayley_step(ax, ay, Ex, Ey, dt)
    A_ = np.array(amp)
    c = (A_[1:-1] * (A_[2:] + A_[:-2])).sum() / (2 * (A_[1:-1] ** 2).sum())
    w_meas = np.arccos(c) / dt
    Om = 2 * np.sin(k / 2)
    w_pred = np.arccos(1 - dt ** 2 * Om ** 2 / 2) / dt
    print(f"  B1 standing wave, L = 64, mode k = {k:.4f}, dt = {dt}:  measured omega = {w_meas:.6f}")
    print(f"     predicted by the linear lattice formula = {w_pred:.6f};  continuum omega = c k = {k:.6f}"
          f"  (relative gap {abs(w_meas-k)/k:.1e}, expected O(k^2, dt^2))")
    # B2 bounded energy, nonlinear amplitude
    L, dt = 16, 0.2
    xx, yy = np.meshgrid(np.arange(L), np.arange(L), indexing="ij")
    thx0 = 0.6 * np.sin(2 * np.pi * yy / L); thy0 = 0.5 * np.cos(2 * np.pi * xx / L)
    ax, ay, Ex, Ey = a_of_theta(thx0), a_of_theta(thy0), np.zeros((L, L)), np.zeros((L, L))
    H0 = energy(ax, ay, Ex, Ey); Hs = []
    N = 20000
    for s in range(N):
        ax, ay, Ex, Ey = cayley_step(ax, ay, Ex, Ey, dt)
        if s % 20 == 0: Hs.append(energy(ax, ay, Ex, Ey))
    Hs = np.array(Hs); t = np.arange(len(Hs))
    slope = np.polyfit(t, Hs, 1)[0] * len(Hs) / H0
    print(f"  B2 nonlinear field, L = 16, {N} steps (dt = {dt}):  H0 = {H0:.4f},  max |H - H0|/H0 = {abs(Hs-H0).max()/H0:.2e},"
          f"  linear drift over the run = {slope:+.1e} (relative)")
    # B3 convergence against the exact-theta reference
    L, T = 8, 2.0
    xx, yy = np.meshgrid(np.arange(L), np.arange(L), indexing="ij")
    thx0 = 0.6 * np.sin(2 * np.pi * yy / L); thy0 = 0.5 * np.cos(2 * np.pi * xx / L)
    def run_c(dt):
        s = (a_of_theta(thx0), a_of_theta(thy0), np.zeros((L, L)), np.zeros((L, L)))
        for _ in range(int(round(T / dt))): s = cayley_step(*s, dt)
        return theta_of_a(s[0]), theta_of_a(s[1]), s[2], s[3]
    ref = (thx0, thy0, np.zeros((L, L)), np.zeros((L, L)))
    for _ in range(int(round(T / 0.0025))): ref = exact_step(*ref, 0.0025)
    errs = []
    for dt in (0.2, 0.1, 0.05):
        r = run_c(dt)
        errs.append(max(np.abs(r[q] - ref[q]).max() for q in range(4)))
    print(f"  B3 max error vs fine reference at T = 2:  dt = 0.2, 0.1, 0.05:  " + ", ".join(f"{e:.2e}" for e in errs)
          + f"   ratios {errs[0]/errs[1]:.2f}, {errs[1]/errs[2]:.2f}  (second order = 4)")


# ===================================================================== C. symplectic
def section_C():
    print("\n" + "-" * 78 + "\nC. the rational map is symplectic in (theta, E)\n" + "-" * 78)
    L, dt = 3, 0.3
    rng = np.random.default_rng(1)
    n = 2 * L * L
    z0 = rng.uniform(-0.7, 0.7, 2 * n)                      # (theta_x, theta_y, E_x, E_y) flattened

    def F(z):
        th, E = z[:n], z[n:]
        ax = a_of_theta(th[:n // 2].reshape(L, L)); ay = a_of_theta(th[n // 2:].reshape(L, L))
        Ex, Ey = E[:n // 2].reshape(L, L), E[n // 2:].reshape(L, L)
        ax, ay, Ex, Ey = cayley_step(ax, ay, Ex, Ey, dt)
        return np.concatenate([theta_of_a(ax).ravel(), theta_of_a(ay).ravel(), Ex.ravel(), Ey.ravel()])

    J = np.zeros((2 * n, 2 * n)); h = 1e-6
    for q in range(2 * n):
        d = np.zeros(2 * n); d[q] = h
        J[:, q] = (F(z0 + d) - F(z0 - d)) / (2 * h)
    Om = np.block([[np.zeros((n, n)), np.eye(n)], [-np.eye(n), np.zeros((n, n))]])
    print(f"  {2*n}-dimensional phase space: max |J^T Omega J - Omega| = {np.abs(J.T @ Om @ J - Om).max():.1e},  det J = {np.linalg.det(J):.10f}")
    print("  => exactly symplectic: the drift theta += 2 atan(dt E/2) is the exact flow of a modified kinetic term T(E),")
    print("     and the kick is the exact flow of the potential, so the composition is symplectic.")


def main():
    t0 = time.time()
    section_A(); section_B(); section_C()
    print(f"\ndone in {time.time()-t0:.1f}s")


main()
