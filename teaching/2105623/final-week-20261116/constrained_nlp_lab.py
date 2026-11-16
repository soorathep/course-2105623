"""Final-class example (2105623, November 16, 2026). Requires NumPy and SciPy.

Run:  python constrained_nlp_lab.py          prints every part
      python constrained_nlp_lab.py --step   pauses after each part (classroom)

Model: minimize f=(x-3)^2+(y-2)^2 subject to g1=x+y-b<=0, g2=-x<=0, g3=-y<=0.
Notation: mu = inequality multipliers (mu1 capacity, mu2 x-bound, mu3 y-bound).
"""
import sys
import numpy as np
import scipy
from scipy.optimize import minimize

TOL = 1e-6      # our verification tolerance; separate from SLSQP's ftol

def objective(r):
    x,y=r
    return (x-3)**2+(y-2)**2

def gradient(r):
    return 2*(np.asarray(r)-[3,2])

def solve_capacity(b=3.0,start=(0.5,0.5),path=None):
    # SciPy's "ineq" form is c(r) >= 0, so c = b - x - y = -g1.
    constraint={"type":"ineq","fun":lambda r:b-r[0]-r[1],
                "jac":lambda r:np.array([-1.,-1.])}
    record=None if path is None else (lambda xk: path.append([float(v) for v in xk]))
    return minimize(objective,start,jac=gradient,method="SLSQP",
                    bounds=[(0,None),(0,None)],constraints=[constraint],
                    options={"ftol":1e-12,"maxiter":200},callback=record)

def audit(r,b,multipliers):
    x,y=r
    lam=np.asarray(multipliers)
    g=np.array([x+y-b,-x,-y])
    normals=np.array([[1,1],[-1,0],[0,-1]])
    return {"objective":objective(r),
            "primal_violation":max(0.,float(np.max(g))),
            "dual_violation":max(0.,float(-np.min(lam))),
            "stationarity":float(np.max(np.abs(gradient(r)+lam@normals))),
            "complementarity":float(np.max(np.abs(lam*g)))}

def analytic_capacity(b):
    """Independent piecewise KKT certificate, including active lower bounds."""
    if b < 0:
        raise ValueError('Capacity b must be nonnegative')
    if b <= 1:
        return np.array([b,0.]), np.array([6-2*b,0.,2-2*b])
    if b <= 5:
        return np.array([(b+1)/2,(b-1)/2]), np.array([5-b,0.,0.])
    return np.array([3.,2.]), np.zeros(3)

def classroom_results():
    """Every number shown in class, as plain Python types (also used by demo.html)."""
    fl=float
    out={'env':{'numpy':np.__version__,'scipy':scipy.__version__,'tol':TOL}}
    path=[[0.5,0.5]]
    r=solve_capacity(3.0,(0.5,0.5),path)
    assert r.success and np.allclose(r.x,[2,1],atol=TOL)
    x,y=map(fl,r.x)
    out['status']={'b':3.0,'start':[0.5,0.5],'success':bool(r.success),'message':str(r.message),
                   'x':x,'y':y,'fun':fl(r.fun),'nit':int(r.nit),'path':path}
    out['recompute']={'f':fl(objective(r.x)),'g':[x+y-3,-x,-y]}
    out['multipliers']={'from_x':2*(3-x),'from_y':2*(2-y),'certificate':[2.0,0.0,0.0]}
    a=audit(r.x,3,[2,0,0])
    out['residuals']={k:fl(v) for k,v in a.items() if k!='objective'}
    assert max(out['residuals'].values())<=TOL
    rows=[]
    for b in [0.5,1.,3.,3.2,5.,6.]:
        expected,mu=analytic_capacity(b)
        rb=solve_capacity(b)
        res=audit(rb.x,b,mu)
        worst=max(v for k,v in res.items() if k!='objective')
        xb,yb=map(fl,rb.x)
        act=[n for n,gv in [('capacity',xb+yb-b),('x>=0',-xb),('y>=0',-yb)] if abs(gv)<=TOL]
        assert rb.success and np.allclose(rb.x,expected,atol=TOL) and worst<TOL
        rows.append({'b':b,'x':xb,'y':yb,'f':fl(rb.fun),'mu':[fl(m) for m in mu],'active':act,'worst':fl(worst)})
    out['capacity']=rows
    old,_=analytic_capacity(3.);new,_=analytic_capacity(3.2)
    out['sensitivity']={'b0':3.0,'b1':3.2,'exact':fl(objective(new)-objective(old)),'linear':-2*(3.2-3.)}
    starts=[]
    for start in [(0.5,0.5),(3,2),(0,0),(4,0)]:
        sp=[[float(start[0]),float(start[1])]]
        rs=solve_capacity(3.0,start,sp)
        assert rs.success and np.allclose(rs.x,[2,1],atol=TOL)
        starts.append({'start':list(start),'start_sum':fl(sum(start)),'success':bool(rs.success),
                       'x':fl(rs.x[0]),'y':fl(rs.x[1]),'f':fl(rs.fun),'nit':int(rs.nit),'path':sp})
    out['starts']=starts
    out['manual']=[{'alpha':al,'x':6*al,'y':4*al,'sum':10*al,'f':fl(objective((6*al,4*al))),'feasible':10*al<=3+1e-12} for al in [.1,.3,.5]]
    s=.3;g=4*s*(s*s-1);h=12*s*s-4;p=-g/h
    assert g*p>0
    out['newton']={'s':s,'g':g,'h':h,'p':p,'slope':g*p,'safe_slope':-g*g}
    return out

STEP = '--step' in sys.argv and sys.stdin.isatty()

def part(n,slide,title):
    print('\n'+'='*72)
    print(f'PART {n} · Slide {slide} · {title}')
    print('='*72)

def pause(nxt):
    if STEP:
        input(f'\n[Return] next: {nxt} ')

def yes(ok):
    return 'yes' if ok else 'NO'

def main():
    R=classroom_results()
    print(f"NumPy {R['env']['numpy']} · SciPy {R['env']['scipy']} · check tolerance {TOL:g}")

    part(1,34,'Solve with SLSQP and read the status')
    st=R['status']
    print('  b = 3, start (0.5, 0.5), method SLSQP, ftol 1e-12')
    print(f"  result.success   {st['success']}")
    print(f"  result.message   {st['message']}")
    print(f"  result.x         x = {st['x']:.6f}   y = {st['y']:.6f}")
    print(f"  result.fun       f = {st['fun']:.6f}")
    pause('Part 2 · Slide 35')

    part(2,35,'Recompute everything from the returned point')
    g=R['recompute']['g']
    print(f"  f(x,y) recomputed      {R['recompute']['f']:10.6f}")
    print(f'  g1 = x + y - 3         {g[0]:10.1e}   must be <= 0  (active)')
    print(f'  g2 = -x                {g[1]:10.6f}   must be <= 0')
    print(f'  g3 = -y                {g[2]:10.6f}   must be <= 0')
    print('  Numbers like 1e-14 are round-off. Do not test with "== 0".')
    pause('Part 3 · Slide 36')

    part(3,36,'Obtain the multipliers')
    m=R['multipliers']
    print('  x > 0 and y > 0, so complementarity gives mu2 = mu3 = 0')
    print(f"  x-stationarity: mu1 = 2(3 - x) = {m['from_x']:.6f}")
    print(f"  y-stationarity: mu1 = 2(2 - y) = {m['from_y']:.6f}")
    print('  Analytic certificate: mu = (2, 0, 0)')
    pause('Part 4 · Slide 37')

    part(4,37,'Four KKT residuals at the returned point')
    print(f'  {"check":<18}{"residual":>10}   pass (<= {TOL:g})')
    for name,key in [('primal','primal_violation'),('dual','dual_violation'),
                     ('stationarity','stationarity'),('complementarity','complementarity')]:
        v=R['residuals'][key]
        print(f'  {name:<18}{v:>10.1e}   {yes(v<=TOL)}')
    print('  1e-6 is OUR check. ftol = 1e-12 is the solver stopping rule.')
    pause('Part 5 · Slides 38-44')

    part(5,'38-44','Change the capacity b')
    print(f'  {"b":>4} {"x":>7} {"y":>7} {"f":>8} {"mu1":>6} {"mu2":>5} {"mu3":>5}  {"active":<17}{"max res":>8}')
    for row in R['capacity']:
        label=', '.join('cap' if a=='capacity' else a for a in row['active']) or 'none'
        if row['b']==5.: label+=' (mu1=0)'
        mu=row['mu']
        print(f"  {row['b']:>4} {row['x']:>7.4f} {row['y']:>7.4f} {row['f']:>8.4f} {mu[0]:>6.2f} {mu[1]:>5.2f} {mu[2]:>5.2f}  {label:<17}{row['worst']:>8.0e}")
    se=R['sensitivity']
    print(f"\n  b: 3 -> 3.2   exact {se['exact']:+.4f}   linear {se['linear']:+.4f}   gap {se['exact']-se['linear']:.4f}")
    pause('Part 6 · Slide 46')

    part(6,46,'Several starting points (b = 3)')
    print(f'  {"start":<12}{"feasible start?":<18}{"success":<9}{"x":>8}{"y":>8}{"f":>9}')
    for row in R['starts']:
        feas='yes' if row['start_sum']<=3 else f"no (x+y={row['start_sum']:g})"
        print(f"  {str(tuple(row['start'])):<12}{feas:<18}{str(row['success']):<9}{row['x']:>8.4f}{row['y']:>8.4f}{row['f']:>9.4f}")
    print('  All starts agree. Convexity proves the global minimum.')
    pause('Part 7 · Slide 47')

    part(7,47,'A manual feasible path from (0,0) along p = (6,4)')
    print('  (This is a hand calculation, not the SLSQP iteration trace.)')
    print(f'  {"alpha":<7}{"point":<14}{"x+y":>5}{"f":>9}   feasible (x+y <= 3)')
    for row in R['manual']:
        print(f"  {row['alpha']:<7}{'(%.1f, %.1f)'%(row['x'],row['y']):<14}{row['sum']:>5.1f}{row['f']:>9.4f}   {yes(row['feasible'])}")
    pause('Part 8 · Slide 48')

    part(8,48,'A failed Newton direction: q(s) = (s^2 - 1)^2 at s = 0.3')
    nw=R['newton']
    print(f"  q'(s)            {nw['g']:+.4f}")
    print(f"  q''(s)           {nw['h']:+.4f}   negative curvature")
    print(f"  Newton p=-q'/q'' {nw['p']:+.4f}")
    print(f"  slope q'*p       {nw['slope']:+.4f}   > 0, so a small step goes UP")
    print(f"  Safeguard p=-q'  {-nw['g']:+.4f}   slope {nw['safe_slope']:+.4f} < 0, descent")
    print('\nAll parts checked.')

if __name__=="__main__":
    main()
