"""
Does boost covariance force the Riemann-Silberstein equation?   (exact arithmetic over Q(i), polynomial solutions)

Equation:   d_t zeta = c curl zeta,  div zeta = 0,     zeta in K^3 (K = Q(i)),  c in K.
Rotations act by  J_x zeta = (y d_z - z d_y) zeta + s e_x x zeta
Boosts    act by  K_x zeta = (x d_t + t d_x) zeta + mu e_x x zeta          (mu = spin coupling; imaginary-angle rotation <-> mu = +-i)
We find the polynomial solution space (total degree <= 3) for each c and test closure under K_x and J_x.
"""
import itertools, sympy as sp
t,x,y,z=sp.symbols('t x y z'); X=(x,y,z); I=sp.I
mons=[t**a*x**b*y**c_*z**d for a in range(4) for b in range(4) for c_ in range(4) for d in range(4) if a+b+c_+d<=3]
def curl(v): return [sp.diff(v[2],y)-sp.diff(v[1],z), sp.diff(v[0],z)-sp.diff(v[2],x), sp.diff(v[1],x)-sp.diff(v[0],y)]
def div(v): return sum(sp.diff(v[i],X[i]) for i in range(3))
def cross_ex(v): return [0, -v[2], v[1]]            # e_x x v
def solution_space(c):
    syms=[sp.Symbol(f'q{a}_{k}') for a in range(3) for k in range(len(mons))]
    zeta=[sum(syms[a*len(mons)+k]*mons[k] for k in range(len(mons))) for a in range(3)]
    eqs=[sp.diff(zeta[a],t)-c*curl(zeta)[a] for a in range(3)]+[div(zeta)]
    rows=[]
    for e in eqs:
        e=sp.expand(e)
        if e==0: continue
        for co in sp.Poly(e,t,x,y,z).coeffs(): rows.append(co)
    M,_=sp.linear_eq_to_matrix(rows,syms)
    ns=M.nullspace()
    return [[sp.expand(sum(v[a*len(mons)+k]*mons[k] for k in range(len(mons)))) for a in range(3)] for v in ns]
def residual_ok(zeta,c):
    r=[sp.expand(sp.diff(zeta[a],t)-c*curl(zeta)[a]) for a in range(3)]+[sp.expand(div(zeta))]
    return all(e==0 for e in r)
def boost(zeta,mu): 
    cr=cross_ex(zeta); return [sp.expand(x*sp.diff(zeta[a],t)+t*sp.diff(zeta[a],x)+mu*cr[a]) for a in range(3)]
def rot(zeta,s):
    cr=cross_ex(zeta); return [sp.expand(y*sp.diff(zeta[a],z)-z*sp.diff(zeta[a],y)+s*cr[a]) for a in range(3)]
vals={'-i':-I,'+i':I,'+1':sp.Integer(1),'-1':sp.Integer(-1),'+2':sp.Integer(2)}
for cn,c in vals.items():
    sols=solution_space(c)
    boosts=[mn for mn,mu in vals.items() if all(residual_ok(boost(s,mu),c) for s in sols)]
    rots=[s_ for s_ in (1,-1) if all(residual_ok(rot(s,s_),c) for s in sols)]
    print(f"c = {cn:>3}: polynomial solutions (deg<=3): {len(sols):3d};  boost-covariant spin couplings mu: {boosts if boosts else 'none'};  rotation-covariant s: {rots}")
