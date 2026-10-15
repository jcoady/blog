"""
Test of the paper's Proposition 6.3(3): how the vector Cayley anchor (Lambda - I)(Lambda + I)^-1 relates to the rotor anchor.

For a spin Cayley parameter zeta_s with sigma = zeta_s . zeta_s, let G = U_spin(zeta_s) = (1+sigma) I + 2 zeta_s . sigma / (1 - sigma),
Lambda its Lorentz matrix.  Tested, in exact arithmetic over Q(i):
  (i)   rotor anchor (Theorem 6.2)            = 2 zeta_s / (1 + sigma)                      [Theorem 7.1]
  (ii)  vector Cayley anchor                   = 2 zeta_s (1 - conj(sigma)) / (1 - sigma conj(sigma))
  (iii) vector Cayley anchor = rotor anchor    iff  sigma is real
  (iv)  2 zeta/(1 + zeta.zeta) with zeta = rotor anchor is NOT the vector Cayley anchor (Prop 6.3(3) as worded)
Last, the paper's own non-simple rotor (a, p, kappa, rho) = (2, 3, (1, 1/2, 0), (6, 0, 0)) is tested numerically.
Needs sympy and numpy.
"""
import sympy as sp, cmath
I=sp.I
sg=[sp.Matrix([[0,1],[1,0]]),sp.Matrix([[0,-I],[I,0]]),sp.Matrix([[1,0],[0,-1]])]
def Xm(x): return x[0]*sp.eye(2)+x[1]*sg[0]+x[2]*sg[1]+x[3]*sg[2]
def comps(H): return [(H[0,0]+H[1,1])/2,(H[0,1]+H[1,0])/2,(H[1,0]-H[0,1])/(2*I),(H[0,0]-H[1,1])/2]
def Amat(zeta):
    k=[sp.re(c) for c in zeta]; r=[sp.im(c) for c in zeta]
    return sp.Matrix([[0,k[0],k[1],k[2]],[k[0],0,r[2],-r[1]],[k[1],-r[2],0,r[0]],[k[2],r[1],-r[0],0]])
def anchor_of(M):                       # M in so(1,3): read zeta = kappa + i rho
    return [M[0,1]+I*M[2,3], M[0,2]+I*M[3,1], M[0,3]+I*M[1,2]]
def lorentz(G):
    N=cl(sp.expand(G.det()))
    cols=[[cl(c) for c in comps(G*Xm([int(i==j) for i in range(4)])*G.H/N)] for j in range(4)]
    return sp.Matrix(cols).T, N
def cl(x): return sp.expand(sp.radsimp(sp.expand(x)))
def simp(v): return [cl(c) for c in v]
def dot(a,b): return cl(sum(x*y for x,y in zip(a,b)))
eta=sp.diag(1,-1,-1,-1)
def analyse(name, zs):
    R=sp.Rational
    sig=dot(zs,zs)
    G=((1+sig)*sp.eye(2)+2*sum((zs[i]*sg[i] for i in range(3)),sp.zeros(2)))*sp.radsimp(1/(1-sig))
    Lm,N=lorentz(G)
    assert (Lm.T*eta*Lm-eta).applyfunc(cl)==sp.zeros(4) and N==1
    Las=(Lm-eta*Lm.T*eta)/2
    zrot=simp([c for c in anchor_of(2*Las*sp.radsimp(1/Lm.trace()))])
    zrot_pred=simp([2*c*sp.radsimp(1/(1+sig)) for c in zs])
    AC=((Lm-sp.eye(4))*(Lm+sp.eye(4)).inv()).applyfunc(cl)
    assert (eta*AC.T*eta+AC).applyfunc(cl)==sp.zeros(4)
    zC=simp(anchor_of(AC))
    tau=dot(zrot,zrot)
    dbl=simp([2*c*sp.radsimp(1/(1+tau)) for c in zrot])           # paper's Prop 6.3(3): doubling applied to the rotor anchor
    sb=sp.conjugate(sig)
    myf=simp([2*c*(1-sb)*sp.radsimp(1/(1-sig*sb)) for c in zs])
    eq=lambda a,b: all(cl(x-y)==0 for x,y in zip(a,b))
    print(f"{name}\n   sigma=zeta_s.zeta_s = {sig};  kappa.rho of zeta_s: {sp.im(sig)/2}")
    print("   rotor anchor from Thm 6.2 == 2 zeta_s/(1+sigma) (Thm 7.1 doubling):", eq(zrot,zrot_pred))
    print("   vector Cayley anchor == rotor anchor:                                 ", eq(zC,zrot))
    print("   vector Cayley anchor == 2 zeta_rot/(1+zeta_rot.zeta_rot) (Prop 6.3(3)): ", eq(zC,dbl))
    print("   vector Cayley anchor == 2 zeta_s (1-conj(sigma))/(1-sigma conj(sigma)): ", eq(zC,myf))
R=sp.Rational
analyse("boost, zeta_s=(1/2,0,0)", [R(1,2),0,0])
analyse("rotation, zeta_s=(0,0,i/2)", [0,0,I*R(1,2)])
analyse("simple mixed (kappa.rho=0), zeta_s=(1/2, i/3, 0)", [R(1,2),I*R(1,3),0])
analyse("non-simple, zeta_s=(1/2+i/3,0,0)", [R(1,2)+I*R(1,3),0,0])
analyse("generic, zeta_s=(1/2+i/3, 1/5, 2i/7)", [R(1,2)+I*R(1,3),R(1,5),2*I*R(1,7)])

# ---- the Part 1 / paper non-simple rotor (a,p,kappa,rho)=(2,3,(1,1/2,0),(6,0,0)), numerically
import numpy as np
x=2+3j; z=np.array([1+6j,0.5,0]); zr=z/x; tau=zr@zr
G=x*np.eye(2)+sum(z[i]*np.array(np.complex128(sg[i].tolist())) for i in range(3))
N=np.linalg.det(G)
sgn=[np.array(np.complex128(m.tolist())) for m in sg]
def Xn(v): return v[0]*np.eye(2)+v[1]*sgn[0]+v[2]*sgn[1]+v[3]*sgn[2]
def cn(Hm): return np.array([(Hm[0,0]+Hm[1,1])/2,(Hm[0,1]+Hm[1,0])/2,(Hm[1,0]-Hm[0,1])/(2j),(Hm[0,0]-Hm[1,1])/2])
Lm=np.array([cn(G@Xn(np.eye(4)[j])@G.conj().T/N) for j in range(4)]).T.real
AC=(Lm-np.eye(4))@np.linalg.inv(Lm+np.eye(4))
zC=np.array([AC[0,1]+1j*AC[2,3],AC[0,2]+1j*AC[3,1],AC[0,3]+1j*AC[1,2]])
print("paper rotor: zeta_rot =",np.round(zr,6)," vector-Cayley anchor =",np.round(zC,6))
rts=[(2-tau+s_*2*np.sqrt(1-tau+0j))/tau for s_ in (1,-1)]
for sig in rts:
    zs=zr*(1+sig)/2; f=2*zs*(1-np.conj(sig))/(1-sig*np.conj(sig))
    print("  sigma=%s |sigma|=%.4f  formula residual %.2e ; equals rotor anchor? %s ; doubling-of-rotor-anchor residual %.2e"%(np.round(sig,4),abs(sig),np.max(abs(f-zC)),np.allclose(zr,zC),np.max(abs(2*zr/(1+tau)-zC))))
