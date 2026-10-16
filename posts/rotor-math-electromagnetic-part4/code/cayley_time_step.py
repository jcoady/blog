"""
Cayley (Crank-Nicolson) time step for the lattice Maxwell system  d/dt (E, B) = M (E, B),  M = [[0, C^T], [-C, 0]]  (antisymmetric).
Q(dt) = (I - dt M/2)^-1 (I + dt M/2) is a RATIONAL ORTHOGONAL matrix (Wildberger's cap for SO(n)).
Exact checks over Q (sympy), accuracy against the matrix exponential (scipy).
2+1 D periodic torus, E on edges, B on plaquettes.
"""
import itertools, numpy as np, sympy as sp
from scipy.linalg import expm
def build(L):
    n=L*L; ix=lambda a,i,j: a*n+(i%L)*L+(j%L)
    C=sp.zeros(n,2*n)                      # (C E)(i,j) = Ex(i,j)+Ey(i+1,j)-Ex(i,j+1)-Ey(i,j)
    for i in range(L):
        for j in range(L):
            C[i*L+j,ix(0,i,j)]+=1; C[i*L+j,ix(1,i+1,j)]+=1; C[i*L+j,ix(0,i,j+1)]-=1; C[i*L+j,ix(1,i,j)]-=1
    D=sp.zeros(n,2*n)                      # div E (i,j) = Ex(i,j)-Ex(i-1,j)+Ey(i,j)-Ey(i,j-1)
    for i in range(L):
        for j in range(L):
            D[i*L+j,ix(0,i,j)]+=1; D[i*L+j,ix(0,i-1,j)]-=1; D[i*L+j,ix(1,i,j)]+=1; D[i*L+j,ix(1,i,j-1)]-=1
    M=sp.zeros(3*n,3*n); M[:2*n,2*n:]=C.T; M[2*n:,:2*n]=-C
    return M,D,C
L=4; M,D,C=build(L); N=M.shape[0]; Id=sp.eye(N)
print("M antisymmetric:", M.T==-M, ";  div(curl^T) = 0 (so Gauss is a conserved linear functional):", D*C.T==sp.zeros(D.shape[0],C.shape[0]))
for dt in (sp.Rational(1),sp.Rational(3,2),sp.Rational(7,3)):
    Q=(Id-dt*M/2).inv()*(Id+dt*M/2)
    orth=(Q.T*Q==Id)
    gauss=(D*Q[:, :]).extract(list(range(D.shape[0])),list(range(N)))  if False else None
    Dfull=sp.zeros(D.shape[0],N); Dfull[:,:2*L*L]=D
    gauss_ok=(Dfull*Q==Dfull)
    Qinv=(Id+dt*M/2).inv()*(Id-dt*M/2)            # Q(-dt)
    rev=(Q*Qinv==Id)
    print(f"dt = {dt}: Q^T Q = I exactly: {orth};  Gauss functional preserved exactly (D Q = D): {gauss_ok};  Q(-dt) = Q(dt)^-1: {rev};  max entry denominator bits: {max(sp.Rational(e).q.bit_length() for e in Q)}")
# accuracy against exp(M t) (floating point), unconditional stability
Mf=np.array(M.tolist(),dtype=float); T=1.0
for steps in (4,8,16,32):
    dt=T/steps; Qf=np.linalg.solve(np.eye(N)-dt*Mf/2,np.eye(N)+dt*Mf/2)
    err=np.max(abs(np.linalg.matrix_power(Qf,steps)-expm(Mf*T))); print(f"accuracy at T = 1: {steps:3d} steps, max error {err:.3e}")
rng=np.random.default_rng(1); v=rng.normal(size=N)
for dt in (0.5,5.0,50.0):
    Qf=np.linalg.solve(np.eye(N)-dt*Mf/2,np.eye(N)+dt*Mf/2); w=v.copy()
    for _ in range(200): w=Qf@w
    print(f"dt = {dt:5.1f}: norm after 200 steps / initial norm = {np.linalg.norm(w)/np.linalg.norm(v):.12f}")
