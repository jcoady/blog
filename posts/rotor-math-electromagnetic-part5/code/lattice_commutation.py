"""
Calculation 21 (Part 5): one-step commutation between the exact rotor-side evolution and the Cayley time step.

  Lattice data in INTEGRATED variables (spacing 1):  E on edges = line integral of E,  Phi on faces = flux of B.
  Semi-discrete system  d/dt (E, Phi) = M (E, Phi),  M = [[0, C^T], [-C, 0]]  (the system of Calculation 18).
  The Cayley step Q(tau) = (I - tau M/2)^-1 (I + tau M/2) satisfies  x(t+tau) = Q(tau) x(t)  iff the trapezoid relation
        (I - tau M/2) x(t+tau) = (I + tau M/2) x(t)          (local: involves M only)
  (a) EXACT, over Q:  for polynomial Maxwell solutions of total degree <= 2 the trapezoid relation holds exactly;
      for degree 3 it fails.
  (b) FLOATING POINT, 1D chain (plane wave): convergence of the Cayley step to the exact continuum solution in space and in time,
      and agreement of the compact Wilson dynamics with the linear dynamics at weak field (error ~ amplitude^2).
Needs sympy, numpy, scipy.
Tested with Python 3.12.3, SymPy 1.14.0, NumPy 2.4.4, SciPy 1.17.1 (one sandbox environment, not a lockfile-built clean environment).
"""
import itertools, random
from fractions import Fraction as Fr
import numpy as np
import sympy as sp
from scipy.linalg import expm

t, x, y, z, tau_s = sp.symbols('t x y z tau'); I = sp.I
def curl(v): return [sp.diff(v[2], y) - sp.diff(v[1], z), sp.diff(v[0], z) - sp.diff(v[2], x), sp.diff(v[1], x) - sp.diff(v[0], y)]
def div(v): return sp.diff(v[0], x) + sp.diff(v[1], y) + sp.diff(v[2], z)
def tofr(expr):
    r = sp.Rational(sp.expand(expr)); return Fr(int(r.p), int(r.q))

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
    E, B = [], []
    for comp in zeta:
        P = sp.Poly(sp.expand(comp), t, x, y, z); Ep = 0; Bp = 0
        for mon, co in P.terms():
            m = t**mon[0] * x**mon[1] * y**mon[2] * z**mon[3]
            Ep += sp.re(co) * m; Bp += sp.im(co) * m
        E.append(sp.expand(Ep)); B.append(sp.expand(Bp))
    return E, B

def combo(sols, seed, n):
    rng = random.Random(seed); zeta = [0, 0, 0]
    for sol in rng.sample(sols, min(n, len(sols))):
        co = rng.randint(-2, 2) + I * rng.randint(-2, 2)
        zeta = [sp.expand(zeta[a] + co * sol[a]) for a in range(3)]
    return zeta

# ---------------- integrated lattice data (spacing 1) ----------------
def edge_data(E, t0, N):
    Et = [sp.expand(c_.subs(t, t0)) for c_ in E]; anti = [sp.integrate(Et[0], x), sp.integrate(Et[1], y), sp.integrate(Et[2], z)]; out = {}
    for i, j, k in itertools.product(range(N + 1), repeat=3):
        out[(i, j, k, 0)] = tofr(anti[0].subs({x: i + 1, y: j, z: k}) - anti[0].subs({x: i, y: j, z: k}))
        out[(i, j, k, 1)] = tofr(anti[1].subs({x: i, y: j + 1, z: k}) - anti[1].subs({x: i, y: j, z: k}))
        out[(i, j, k, 2)] = tofr(anti[2].subs({x: i, y: j, z: k + 1}) - anti[2].subs({x: i, y: j, z: k}))
    return out

def flux_data(B, t0, N):
    Bt = [sp.expand(c_.subs(t, t0)) for c_ in B]
    Iz = sp.integrate(sp.integrate(Bt[2], x), y); Ix = sp.integrate(sp.integrate(Bt[0], y), z); Iy = sp.integrate(sp.integrate(Bt[1], z), x)
    def box(F2, v1, v2, a1, a2, fixed):
        tot = 0
        for (p, q, sg) in ((a1 + 1, a2 + 1, 1), (a1, a2, 1), (a1 + 1, a2, -1), (a1, a2 + 1, -1)):
            tot += sg * F2.subs({v1: p, v2: q, **fixed})
        return tofr(tot)
    out = {'xy': {}, 'yz': {}, 'zx': {}}
    for (i, j, k) in itertools.product(range(N), range(N), range(N + 1)): out['xy'][(i, j, k)] = box(Iz, x, y, i, j, {z: k})
    for (i, j, k) in itertools.product(range(N + 1), range(N), range(N)): out['yz'][(i, j, k)] = box(Ix, y, z, j, k, {x: i})
    for (i, j, k) in itertools.product(range(N), range(N + 1), range(N)): out['zx'][(i, j, k)] = box(Iy, z, x, k, i, {y: j})
    return out

def CE(E, key, p):
    i, j, k = p
    if key == 'xy': return E[(i, j, k, 0)] + E[(i + 1, j, k, 1)] - E[(i, j + 1, k, 0)] - E[(i, j, k, 1)]
    if key == 'yz': return E[(i, j, k, 1)] + E[(i, j + 1, k, 2)] - E[(i, j, k + 1, 1)] - E[(i, j, k, 2)]
    return E[(i, j, k, 2)] + E[(i, j, k + 1, 0)] - E[(i + 1, j, k, 2)] - E[(i, j, k, 0)]

def CTPhi(P, e):
    i, j, k, m = e
    if m == 0: return P['xy'][(i, j, k)] - P['xy'][(i, j - 1, k)] + P['zx'][(i, j, k - 1)] - P['zx'][(i, j, k)]
    if m == 1: return P['yz'][(i, j, k)] - P['yz'][(i, j, k - 1)] + P['xy'][(i - 1, j, k)] - P['xy'][(i, j, k)]
    return P['zx'][(i, j, k)] - P['zx'][(i - 1, j, k)] + P['yz'][(i, j - 1, k)] - P['yz'][(i, j, k)]

def trapezoid_residuals(E, B, t0, tau, N=4):
    E0, E1 = edge_data(E, t0, N), edge_data(E, t0 + tau, N); P0, P1 = flux_data(B, t0, N), flux_data(B, t0 + tau, N)
    rE = Fr(0); rP = Fr(0); nE = nP = 0
    for key in P0:
        for p in P0[key]:
            try:
                r = (P1[key][p] - P0[key][p]) + (Fr(tau.p, tau.q) / 2) * (CE(E0, key, p) + CE(E1, key, p))
            except KeyError: continue
            rP = max(rP, abs(r)); nP += 1
    for e in E0:
        try:
            r = (E1[e] - E0[e]) - (Fr(tau.p, tau.q) / 2) * (CTPhi(P0, e) + CTPhi(P1, e))
        except KeyError: continue
        rE = max(rE, abs(r)); nE += 1
    return rE, nE, rP, nP


# ---------------- 1D chain, floating point ----------------
def chain_matrix(N):
    Dm = np.eye(N) - np.roll(np.eye(N), -1, axis=1)                              # (Dm B)_i = B_i - B_{i-1}  (ones at (i, i-1))
    M = np.zeros((2 * N, 2 * N)); M[:N, N:] = Dm; M[N:, :N] = -Dm.T
    return M
def check_chain_orientation(N=6):
    """dE_i/dt = B_i - B_{i-1}  and  dB_i/dt = E_{i+1} - E_i."""
    M = chain_matrix(N); B = np.zeros(N); B[2] = 1; E = np.zeros(N); E[2] = 1
    dE = (M @ np.concatenate([np.zeros(N), B]))[:N]; dB = (M @ np.concatenate([E, np.zeros(N)]))[N:]
    assert np.allclose(dE, [0, 0, 1, -1, 0, 0]) and np.allclose(dB, [0, 1, -1, 0, 0, 0]), (dE, dB)
def chain_initial(N, m, eps):
    k = 2 * np.pi * m / N; i = np.arange(N)
    th = lambda xx, tt: eps * np.sin(k * xx - k * tt)
    E = -eps * k * np.cos(k * i); B = th(i + 1, 0) - th(i, 0)
    Bex = lambda tt: th(i + 1, tt) - th(i, tt)
    return k, E, B, Bex
def wilson_rk4(N, theta, E, dt, steps):
    def rhs(th, Ev):
        P = np.roll(th, -1) - th
        return Ev, np.sin(P) - np.sin(np.roll(P, 1))
    for _ in range(steps):
        k1 = rhs(theta, E); k2 = rhs(theta + dt / 2 * k1[0], E + dt / 2 * k1[1]); k3 = rhs(theta + dt / 2 * k2[0], E + dt / 2 * k2[1]); k4 = rhs(theta + dt * k3[0], E + dt * k3[1])
        theta = theta + dt / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]); E = E + dt / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
    return theta, E
def linear_rk4(N, theta, E, dt, steps):
    def rhs(th, Ev): return Ev, np.roll(th, -1) - 2 * th + np.roll(th, 1)
    for _ in range(steps):
        k1 = rhs(theta, E); k2 = rhs(theta + dt / 2 * k1[0], E + dt / 2 * k1[1]); k3 = rhs(theta + dt / 2 * k2[0], E + dt / 2 * k2[1]); k4 = rhs(theta + dt * k3[0], E + dt * k3[1])
        theta = theta + dt / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]); E = E + dt / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
    return theta, E


def main():
    print("(a) EXACT one-step commutation (trapezoid relation) for integrated lattice data of polynomial Maxwell solutions")
    for deg, n in ((2, 5), (3, 6)):
        sols = solution_space(deg)
        for seed in (1, 2):
            zeta = combo(sols, seed, n); E, B = split_EB(zeta)
            degE = max(sp.Poly(c_, t, x, y, z).total_degree() for c_ in E + B if c_ != 0)
            rE, nE, rP, nP = trapezoid_residuals(E, B, sp.Rational(1, 3), sp.Rational(2, 3))
            print(f"      solution space deg <= {deg}, combination {seed} (actual total degree {degE}):  Ampere-half residual max {rE} on {nE} edges;  Faraday-half residual max {rP} on {nP} faces")

    check_chain_orientation()
    print("(b) floating point, 1D chain, plane wave theta = eps sin(kx - kt), one period T = 2 pi/k")
    N = 256; m = 8
    print("    spatial convergence (exact semi-discrete flow expm(M T) against the continuum solution), eps = 1:")
    prev = None
    for m_ in (4, 8, 16, 32):
        k, E0, B0, Bex = chain_initial(N, m_, 1.0); T = N / m_
        v = expm(chain_matrix(N) * T) @ np.concatenate([E0, B0]); err = np.max(abs(v[N:] - Bex(T))) / (k)
        print(f"      k = 2 pi*{m_}/{N}: error / (eps k) = {err:.3e}" + ("" if prev is None else f"   (ratio to previous {prev / err:.2f})")); prev = err
    print("    time convergence (Cayley step Q(tau)^n against the exact semi-discrete flow), N = 256, m = 8, eps = 1:")
    k, E0, B0, Bex = chain_initial(N, m, 1.0); T = N / m; M = chain_matrix(N); v0 = np.concatenate([E0, B0]); vex = expm(M * T) @ v0; prev = None
    for n in (16, 32, 64, 128):
        tau = T / n; Q = np.linalg.solve(np.eye(2 * N) - tau / 2 * M, np.eye(2 * N) + tau / 2 * M)
        v = np.linalg.matrix_power(Q, n) @ v0; err = np.max(abs(v - vex)) / k
        print(f"      tau = {tau:.4f} (n = {n:3d}): error / (eps k) = {err:.3e}" + ("" if prev is None else f"   (ratio to previous {prev / err:.2f})")); prev = err
    print("    weak-field agreement: compact Wilson dynamics versus linear dynamics (RK4, dt = 0.01), N = 256, m = 8, one period:")
    prev = None; steps = int(round(T / 0.01))
    for eps in (0.4, 0.2, 0.1, 0.05):
        i = np.arange(N); k = 2 * np.pi * m / N
        th0 = eps * np.sin(k * i); E0 = -eps * k * np.cos(k * i)
        thw, _ = wilson_rk4(N, th0.copy(), E0.copy(), 0.01, steps); thl, _ = linear_rk4(N, th0.copy(), E0.copy(), 0.01, steps)
        err = np.max(abs(thw - thl)) / eps
        print(f"      eps = {eps:5.3f}: max|theta_Wilson - theta_linear| / eps = {err:.3e}" + ("" if prev is None else f"   (ratio to previous {prev / err:.2f})")); prev = err

main()
