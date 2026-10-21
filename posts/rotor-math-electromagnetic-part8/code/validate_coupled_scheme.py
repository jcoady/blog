"""
Numerical validation of the coupled scalar-gauge kick-drift-kick scheme of Part 7 (floating point, vectorised).

  (0) cross-check: the vectorised step equals the reference implementation of minimal_action_model.py (L = 3, floats)
  (1) unforced evolution with nonzero charge and magnetic curvature, Gauss-law initial data:
        max |G| over time, total charge, energy deviation (and its scaling with h)
  (2) forward-backward round trip on nontrivial coupled data
  (3) temporal convergence against a fine-step reference (spatial lattice fixed)
NOT tested here: spatial convergence of the coupled system, manufactured solutions, dispersion of the coupled system.
Needs numpy.  Tested with Python 3.12.3, NumPy 2.4.4.
"""
import math
import random
import numpy as np


def plaq_U(U): return U[0] * np.roll(U[1], -1, axis=0) * np.conj(np.roll(U[0], -1, axis=1)) * np.conj(U[1])

def kick(th, E, phi, pi, h, m2):
    U = np.exp(1j * th); xi = plaq_U(U).imag; dE = np.empty_like(E)
    dE[0] = xi - np.roll(xi, 1, axis=1); dE[1] = -(xi - np.roll(xi, 1, axis=0))
    dpi = m2 * np.conj(phi)
    for mu in (0, 1):
        z = U[mu] * np.roll(phi, -1, axis=mu) * np.conj(phi); dE[mu] += 2 * z.imag
        dpi = dpi + 2 * np.conj(phi) - np.roll(U[mu], 1, axis=mu) * np.conj(np.roll(phi, 1, axis=mu)) - np.conj(U[mu]) * np.conj(np.roll(phi, -1, axis=mu))
    return th, E - h * dE, phi, pi - h * dpi

def drift(th, E, phi, pi, h): return th + 2 * np.arctan(h * E / 2), E, phi + h * np.conj(pi), pi

def step(s, h, m2):
    s = kick(*s, h / 2, m2); s = drift(*s, h); return kick(*s, h / 2, m2)

def gauss(E, phi, pi):
    return sum(E[mu] - np.roll(E[mu], 1, axis=mu) for mu in (0, 1)) - 2 * (pi * phi).imag

def energy(s, m2):
    th, E, phi, pi = s; U = np.exp(1j * th); H = 0.5 * np.sum(E ** 2) + np.sum(1 - plaq_U(U).real) + np.sum(abs(pi) ** 2) + m2 * np.sum(abs(phi) ** 2)
    for mu in (0, 1): H += np.sum(abs(U[mu] * np.roll(phi, -1, axis=mu) - phi) ** 2)
    return H

def initial_state(L, A=0.5, B=0.3):
    i, j = np.meshgrid(np.arange(L), np.arange(L), indexing='ij'); k = 2 * np.pi / L
    phi = A * np.cos(k * i) * np.cos(k * j) * (1 + 0.3j) + 0.05j * np.sin(k * j)
    pi = 1j * B * np.sin(k * i) * np.cos(k * j) + 0.02 * np.cos(k * j)
    th = np.array([0.1 * np.sin(k * j), 0.15 * np.cos(k * i)])
    rho = -2 * (pi * phi).imag; rho = rho - rho.mean()                  # total charge must vanish on the torus
    # solve  lap(chi) = rho  with  E_mu = -(chi(x+mu) - chi(x))  so that  div E = -rho
    kk = np.fft.fftfreq(L) * 2 * np.pi; K0, K1 = np.meshgrid(kk, kk, indexing='ij')
    lam = 2 * (np.cos(K0) - 1) + 2 * (np.cos(K1) - 1); lam[0, 0] = 1.0
    chi = np.real(np.fft.ifft2(np.fft.fft2(rho) / lam))
    E = np.array([-(np.roll(chi, -1, axis=0) - chi), -(np.roll(chi, -1, axis=1) - chi)])
    E[0] += 0.05 * np.sin(k * j); E[1] += 0.04 * np.cos(k * i)         # divergence-free (transverse) parts
    # make the initial Gauss function vanish exactly (up to round-off): adjust rho in the data through pi
    return th, E, phi, pi

def cross_check():
    import minimal_action_model as mam
    rng = random.Random(1); Lc = mam.L; m2 = 1.0; h = 0.13
    th = np.array([[[rng.uniform(-1, 1) for _ in range(Lc)] for _ in range(Lc)] for _ in range(2)]); E = np.array([[[rng.uniform(-1, 1) for _ in range(Lc)] for _ in range(Lc)] for _ in range(2)])
    phi = np.array([[complex(rng.uniform(-1, 1), rng.uniform(-1, 1)) for _ in range(Lc)] for _ in range(Lc)]); pi = np.array([[complex(rng.uniform(-1, 1), rng.uniform(-1, 1)) for _ in range(Lc)] for _ in range(Lc)])
    Rf = mam.Ring(float)
    a = {((i, j), mu): -math.tan(th[mu, i, j] / 2) for i in range(Lc) for j in range(Lc) for mu in (0, 1)}
    st = mam.State(a, {((i, j), mu): E[mu, i, j] for i in range(Lc) for j in range(Lc) for mu in (0, 1)},
                   {(i, j): (phi[i, j].real, phi[i, j].imag) for i in range(Lc) for j in range(Lc)}, {(i, j): (pi[i, j].real, pi[i, j].imag) for i in range(Lc) for j in range(Lc)})
    ref = mam.step(Rf, st, h, m2); vec = step((th, E, phi, pi), h, m2); err = 0.0
    for i in range(Lc):
        for j in range(Lc):
            for mu in (0, 1):
                err = max(err, abs(-2 * math.atan(ref.a[((i, j), mu)]) - vec[0][mu, i, j]), abs(ref.E[((i, j), mu)] - vec[1][mu, i, j]))
            err = max(err, abs(complex(*ref.phi[(i, j)]) - vec[2][i, j]), abs(complex(*ref.pi[(i, j)]) - vec[3][i, j]))
    return err

def main():
    m2 = 1.0; L = 32
    print(f"(0) vectorised step versus the reference implementation (L = 3, random state): max difference {cross_check():.2e}")
    s0 = initial_state(L); G0 = gauss(s0[1], s0[2], s0[3]); rho0 = -2 * (s0[3] * s0[2]).imag
    print(f"(1) L = {L}, m^2 = 1: initial max|G| = {abs(G0).max():.2e}, max|rho| = {abs(rho0).max():.3f}, max magnetic plaquette angle = {abs(np.angle(plaq_U(np.exp(1j * s0[0])))).max():.3f}, H0 = {energy(s0, m2):.5f}")
    T = 20.0
    print("    unforced evolution to T = 20 (nonzero charge density and magnetic curvature):")
    print("      h        steps   max|G|      max|total charge - initial|   max|H/H0 - 1|    round trip error (forward T then back)")
    for h in (0.1, 0.05, 0.025):
        n = int(round(T / h)); s = s0; Gm = 0.0; Qm = 0.0; Hm = 0.0; H0 = energy(s0, m2); Q0 = np.sum(-2 * (s0[3] * s0[2]).imag)
        for _ in range(n):
            s = step(s, h, m2)
            Gm = max(Gm, abs(gauss(s[1], s[2], s[3])).max()); Qm = max(Qm, abs(np.sum(-2 * (s[3] * s[2]).imag) - Q0)); Hm = max(Hm, abs(energy(s, m2) / H0 - 1))
        back = s
        for _ in range(n): back = step(back, -h, m2)
        rt = max(abs(back[0] - s0[0]).max(), abs(back[1] - s0[1]).max(), abs(back[2] - s0[2]).max(), abs(back[3] - s0[3]).max())
        print(f"      {h:<7} {n:5d}   {Gm:.2e}   {Qm:.2e}                     {Hm:.3e}        {rt:.2e}")
    print("(3) temporal convergence at T = 2 (lattice fixed), reference step h = 0.003125; max-norm error over all fields:")
    def run(h, T=2.0):
        s = s0
        for _ in range(int(round(T / h))): s = step(s, h, m2)
        return s
    ref = run(0.003125); prev = None
    for h in (0.2, 0.1, 0.05, 0.025):
        s = run(h); err = max(abs(s[0] - ref[0]).max(), abs(s[1] - ref[1]).max(), abs(s[2] - ref[2]).max(), abs(s[3] - ref[3]).max())
        print(f"      h = {h:<6}: error {err:.3e}" + ("" if prev is None else f"   (ratio to previous {prev / err:.2f})")); prev = err

if __name__ == "__main__":
    main()
