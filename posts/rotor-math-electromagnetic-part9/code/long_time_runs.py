"""
Floating-point long-time runs for the coupled scalar-gauge scheme of Parts 7-8.  These are the only floating-point
tests of Part 9; the dispersion relation itself and the order of the update are checked exactly in
exact_dispersion_check.py.

  (A) measured one-step trace for a small-amplitude gauge wave and scalar wave versus the exact prediction
      cos(omega h) = 1 - h^2 Omega^2 / 2, for several h  (L = 32, long runs)
  (B) refinement: L = 16, 32, 64, 128 at a fixed physical wavenumber, fixed Courant number h = 0.5 (lattice units)
  (C) long-time behaviour of the constraint, charge and energy on the nonlinear data of Part 8 (T up to 1000)
  (D) amplitude dependence of the measured trace (nonlinear shift of the gauge wave)
  (E) a single complex plane wave is not stationary: its net current drives the uniform electric field
  (F) the stability limit h^2 (m^2 + 8) <= 4 predicted by the exact trace, tested above and below the limit

The measured quantity in (A), (B), (D) is the trace of the one-step map on the mode: for any linear functional s_n of the
state at step n, a mode with eigenvalues exp(+-i omega h) satisfies s_{n+1} + s_{n-1} = 2 cos(omega h) s_n, so
cos(omega h) is estimated by least squares from the recorded series.  Needs numpy and validate_coupled_scheme.py (Part 8).
Tested with Python 3.12.3, NumPy 2.4.4.
"""
import math
import numpy as np
from validate_coupled_scheme import step, energy, gauss, initial_state


def trace_half(s):
    """Least-squares estimate of cos(omega h) from s_{n+1} + s_{n-1} = 2 cos(omega h) s_n."""
    s = np.asarray(s); return float(np.sum(s[1:-1] * (s[2:] + s[:-2])) / (2 * np.sum(s[1:-1] ** 2)))

def gauge_wave(L, n, amp):
    i, j = np.meshgrid(np.arange(L), np.arange(L), indexing='ij'); k = 2 * np.pi * n / L
    th = np.array([0 * i, amp * np.cos(k * i)], dtype=float)               # theta_y varying in x: transverse, E = 0, phi = pi = 0
    z = np.zeros((L, L), dtype=complex)
    probe = lambda s: 2 * np.sum(s[0][1] * np.cos(k * i)) / L ** 2
    return (th, np.zeros((2, L, L)), z.copy(), z.copy()), probe, k

def scalar_wave(L, n, amp):
    """Real standing wave phi = amp cos(k x), theta = 0, E = 0, pi = 0.  It carries no current (Im(phi(x+1) conj(phi(x))) = 0)
    and no charge, so the gauge field stays at rest and the scalar evolves exactly as the free lattice Klein-Gordon field."""
    i, j = np.meshgrid(np.arange(L), np.arange(L), indexing='ij'); k = 2 * np.pi * n / L
    phi = (amp * np.cos(k * i)).astype(complex)
    probe = lambda s: 2 * float(np.sum(s[2].real * np.cos(k * i))) / L ** 2
    return (np.zeros((2, L, L)), np.zeros((2, L, L)), phi, np.zeros((L, L), dtype=complex)), probe, k

def measure(make, L, n, amp, h, m2, nsteps):
    s, probe, k = make(L, n, amp); rec = [probe(s)]
    for _ in range(nsteps):
        s = step(s, h, m2); rec.append(probe(s))
    return trace_half(rec), k

def run_A():
    L, n, amp, T = 32, 2, 1e-4, 1000.0
    print(f"(A) L = {L}, mode n = {n} (k = 2 pi n / L = {2*np.pi*n/L:.5f}), amplitude {amp:g}, T = {T:g}; m^2 = 1 for the scalar wave")
    print("    wave     h       steps   Omega      cos(wh) measured    cos(wh) exact       |difference|   omega measured   omega exact (discrete)   omega continuum   rel. error vs continuum")
    for name, make, m2 in (("gauge ", gauge_wave, 1.0), ("scalar", scalar_wave, 1.0)):
        for h in (0.5, 0.25, 0.125, 0.0625):
            ns = int(round(T / h)); c, k = measure(make, L, n, amp, h, m2, ns)
            Om2 = 4 * math.sin(k / 2) ** 2 + (m2 if name == "scalar" else 0.0); Om = math.sqrt(Om2)
            cex = 1 - h * h * Om2 / 2; wm = math.acos(c) / h; wex = 2 * math.asin(h * Om / 2) / h
            print(f"    {name}  {h:<6} {ns:6d}  {Om:.6f}   {c:.12f}   {cex:.12f}   {abs(c-cex):.2e}      {wm:.9f}      {wex:.9f}            {Om:.9f}       {wm/Om-1:+.3e}")

def run_B():
    h = 0.3; ell = 2 * np.pi; m = 1.0
    print(f"(B) refinement at fixed physical box 2 pi, mode n = 1 (kappa = 1), Courant number h = {h} in lattice units, amplitude 1e-4")
    print("    wave     L     eps=2pi/L   steps   omega_phys measured   omega continuum   rel. error     ratio to previous")
    for name, make in (("gauge ", gauge_wave), ("scalar", scalar_wave)):
        prev = None
        for L in (16, 32, 64, 128):
            eps = ell / L; m2 = (m * eps) ** 2 if name == "scalar" else 0.0
            nsteps = int(round(10 * L / h)) if L <= 64 else int(round(6 * L / h))
            c, k = measure(make, L, 1, 1e-4, h, m2, nsteps)
            wlat = math.acos(c) / h; wphys = wlat / eps
            wc = math.sqrt(1.0 + m * m) if name == "scalar" else 1.0
            err = wphys / wc - 1
            print(f"    {name}  {L:4d}  {eps:.6f}   {nsteps:6d}   {wphys:.9f}         {wc:.9f}       {err:+.3e}    " + ("" if prev is None else f"{prev/err:.2f}"))
            prev = err

def run_C():
    m2 = 1.0; s0 = initial_state(32); H0 = energy(s0, m2); Q0 = np.sum(-2 * (s0[3] * s0[2]).imag)
    print("(C) nonlinear data of Part 8 (L = 32, m^2 = 1), unforced, T = 1000; running maxima of |H/H0 - 1| up to time T'")
    print("    h        steps    T'=20        T'=100       T'=500       T'=1000      max|G|     max|Q - Q0|")
    for h in (0.1, 0.05):
        n = int(round(1000 / h)); s = s0; Hm = 0.0; Gm = 0.0; Qm = 0.0; marks = {}; 
        for it in range(1, n + 1):
            s = step(s, h, m2)
            Hm = max(Hm, abs(energy(s, m2) / H0 - 1)); Gm = max(Gm, abs(gauss(s[1], s[2], s[3])).max()); Qm = max(Qm, abs(np.sum(-2 * (s[3] * s[2]).imag) - Q0))
            t = it * h
            for Tm in (20, 100, 500, 1000):
                if abs(t - Tm) < h / 2: marks[Tm] = Hm
        print(f"    {h:<7} {n:6d}   {marks[20]:.3e}    {marks[100]:.3e}    {marks[500]:.3e}    {marks[1000]:.3e}    {Gm:.2e}   {Qm:.2e}")

def run_D():
    L, n, h = 32, 2, 0.5; T = 1000.0; ns = int(round(T / h))
    print(f"(D) amplitude dependence, gauge wave, L = {L}, n = {n}, h = {h}, T = {T:g}: deviation of the measured cos(omega h) from the linear prediction")
    print("    amplitude    cos(wh) measured    cos(wh) linear      difference      difference / amplitude^2")
    for amp in (1e-3, 1e-2, 3e-2, 1e-1, 3e-1):
        c, k = measure(gauge_wave, L, n, amp, h, 0.0, ns)
        cex = 1 - h * h * 4 * math.sin(k / 2) ** 2 / 2
        print(f"    {amp:<10g}   {c:.12f}   {cex:.12f}   {c-cex:+.3e}      {(c-cex)/amp**2:+.4e}")

def run_E():
    L, n, A, m2, h = 32, 2, 0.01, 1.0, 0.05
    i, j = np.meshgrid(np.arange(L), np.arange(L), indexing='ij'); k = 2 * np.pi * n / L
    s = (np.zeros((2, L, L)), np.zeros((2, L, L)), A * np.exp(1j * k * i), np.zeros((L, L), dtype=complex))
    Om = math.sqrt(m2 + 4 * math.sin(k / 2) ** 2)
    print(f"(E) a single complex plane wave phi = {A} exp(ikx) carries a net current; it drives the uniform electric field (L = {L}, n = {n}, h = {h})")
    print("    t       mean E_x measured    prediction  -2 A^2 sin k [t/2 + sin(2 Omega t)/(4 Omega)]    relative difference")
    for it in range(1, 201):
        s = step(s, h, m2)
        if it in (50, 100, 200):
            t = it * h; pred = -2 * A ** 2 * math.sin(k) * (t / 2 + math.sin(2 * Om * t) / (4 * Om)); mE = s[1][0].mean()
            print(f"    {t:<6.2f}  {mE:+.6e}        {pred:+.6e}                                           {mE/pred-1:+.2e}")

def run_F():
    L, m2 = 8, 1.0
    i, j = np.meshgrid(np.arange(L), np.arange(L), indexing='ij')
    print(f"(F) stability limit: the zone-corner scalar mode (k = (pi, pi), Lambda = m^2 + 8 = {m2+8:g}) has |mu| = (|tau| + sqrt(tau^2 - 4))/2 per step when |tau| > 2, tau = 2 - h^2 Lambda; L = {L}")
    print("    h       tau        predicted |mu| per step     measured (steps 30 to 40)")
    for h in (0.6, 0.7, 1.0):
        tau = 2 - h * h * (m2 + 8); mu = 1.0 if abs(tau) <= 2 else (abs(tau) + math.sqrt(tau * tau - 4)) / 2
        s = (np.zeros((2, L, L)), np.zeros((2, L, L)), (1e-8 * (-1.0) ** (i + j)).astype(complex), np.zeros((L, L), dtype=complex)); env = []
        for _ in range(41):
            env.append(abs(s[2]).max()); s = step(s, h, m2)
        meas = (max(env[31:41]) / max(env[21:31])) ** 0.1
        print(f"    {h:<6} {tau:+8.4f}   {mu:12.4f}                {meas:12.4f}")

if __name__ == "__main__":
    run_A(); print(); run_B(); print(); run_C(); print(); run_D(); print(); run_E(); print(); run_F()
