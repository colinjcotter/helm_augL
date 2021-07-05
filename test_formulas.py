#Here we check out the accuracy of the formal Schur complement approximation
import firedrake as fd
import numpy as np
from scipy.fft import fft, ifft

n = 20
mesh = fd.PeriodicUnitSquareMesh(n, n)
degree = 1
V = fd.VectorFunctionSpace(mesh, "BDM", degree+1)
Q = fd.VectorFunctionSpace(mesh, "DG", degree)
Q0 = fd.FunctionSpace(mesh, "DG", degree)

u = fd.TrialFunction(V)
ur = u[0, :]
ui = u[1, :]

p = fd.TrialFunction(Q)
pr = p[0]
pi = p[1]

v = fd.TestFunction(V)
vr = v[0, :]
vi = v[1, :]

q = fd.TestFunction(Q)
qr = q[0]
qi = q[1]

#number of steps
T = 1.
M = 20
Dt = T/M
#timestep
#circulant coefficient
alphav = 0.01
#timestep offset parameter
thetav = 0.5
gamma = fd.Constant(1.0e6)

# Gamma coefficients
Nt = M
exponents = np.arange(Nt)/Nt
Gam = alphav**exponents

# Di coefficients
C1col = np.zeros(Nt)
C2col = np.zeros(Nt)
C1col[:2] = np.array([1, -1])/Dt
C2col[:2] = np.array([thetav, 1-thetav])
D1 = np.sqrt(Nt)*fft(Gam*C1col)
D2 = np.sqrt(Nt)*fft(Gam*C2col)
i = 0
D1r = fd.Constant(np.real(D1[i]))
D1i = fd.Constant(np.imag(D1[i]))
D2r = fd.Constant(np.real(D2[i]))
D2i = fd.Constant(np.imag(D2[i]))

sgr = fd.Constant(0)
sgi = fd.Constant(0)

#some functions to use in RHS
x, y = fd.SpatialCoordinate(mesh)
p1 = fd.sin(fd.pi*x)*fd.sin(fd.pi*y)
p2 = fd.cos(2*fd.pi*x)*fd.sin(4*fd.pi*y)
fur = fd.as_vector([fd.exp(p1+2*p2), fd.cos(2*p2 - p1)])
fui = fd.as_vector([fd.sin(3*p1+2*p2), fd.exp(3*p2 + 2*p1)])
fpr = fd.Function(Q0, name='fpr').interpolate(fd.cos(fd.sin(p1+p2)))
fpi = fd.Function(Q0, name='fpi').interpolate(fd.sin(fd.cos(2*p1-p2)))
one = fd.Function(Q0).assign(1.)
fpr -= fd.assemble(fpr*fd.dx)/fd.assemble(one*fd.dx)
fpi -= fd.assemble(fpi*fd.dx)/fd.assemble(one*fd.dx)

#some useful abbrevations
D1xu_r = D1r*ur - D1i*ui
D2xu_r = D2r*ur - D2i*ui
D1xp_r = D1r*fpr - D1i*fpi
D2xp_r = D2r*fpr - D2i*fpi
D1xu_i = D1r*ui + D1i*ur
D2xu_i = D2r*ui + D2i*ur
D1xp_i = D1r*fpi + D1i*fpr
D2xp_i = D2r*fpi + D2i*fpr

#solving for u(D) where D = fpr+i*fpi

eqn = (
    fd.inner(vr, D1xu_r) + fd.inner(vi, D1xu_i)
    - fd.div(vr)*(D2xp_r) - fd.div(vr)*D2xp_r
    + gamma*(fd.div(vr)*(D1xp_r + fd.div(D2xu_r))
             +fd.div(vi)*(D1xp_i + fd.div(D2xu_i)))
    )*fd.dx

uD = fd.Function(V)
uD_prob = fd.LinearVariationalProblem(fd.lhs(eqn),
                                      fd.rhs(eqn),
                                      uD, constant_jacobian=False)
ud_parameters = {
    'ksp_type':'preonly',
    'pc_type':'lu',
    "pc_factor_mat_solver_type": "mumps"
}
uD_solver = fd.LinearVariationalSolver(uD_prob,
                                       solver_parameters=
                                       ud_parameters)

#some useful abbrevations (making ur and ui now from uD)
ur = uD[0, :]
ui = uD[1, :]
D1xu_r = D1r*ur - D1i*ui
D2xu_r = D2r*ur - D2i*ui
D1xp_r = D1r*fpr - D1i*fpi
D2xp_r = D2r*fpr - D2i*fpi
D1xu_i = D1r*ui + D1i*ur
D2xu_i = D2r*ui + D2i*ur
D1xp_i = D1r*fpi + D1i*fpr
D2xp_i = D2r*fpi + D2i*fpr

eqn = (pr*qr + pi*qi
       - (
           qr*(D1xp_r + fd.div(D2xu_r)) +
           qi*(D1xp_i + fd.div(D2xu_i))
       )
       )*fd.dx

p_exact = fd.Function(Q, name='exact')
p_exact_prob = fd.LinearVariationalProblem(fd.lhs(eqn), fd.rhs(eqn),
                                           p_exact)
v_basis = fd.VectorSpaceBasis(constant=True)
p_exact_solver = fd.LinearVariationalSolver(p_exact_prob,
                                            solver_parameters=
                                            ud_parameters,
                                            nullspace=v_basis)

V0 = fd.FunctionSpace(mesh, "BDM", degree+1)
W = V0 * Q0
u, p = fd.TrialFunctions(W)
v, q = fd.TestFunctions(W)

psirhs = fd.Function(Q0)
eqn = (
    fd.inner(v, u) - fd.div(v)*p
    + q*fd.div(u)
    - q*psirhs
    )*fd.dx

w = fd.Function(W)

laplace_parameters = {
    'mat_type':'aij',
    'ksp_type':'preonly',
    'pc_type':'lu',
    "pc_factor_mat_solver_type": "mumps"
}

laplace_prob = fd.LinearVariationalProblem(fd.lhs(eqn), fd.rhs(eqn), w)
nullspace = fd.MixedVectorSpaceBasis(W, [W.sub(0), v_basis])
laplace_solver = fd.LinearVariationalSolver(laplace_prob,
                                            solver_parameters=
                                            laplace_parameters,
                                            nullspace=nullspace)

psir = fd.Function(Q0, name='psir')
psii = fd.Function(Q0, name='psii')

p = fd.TrialFunction(Q)
pr = p[0]
pi = p[1]
q = fd.TestFunction(Q)
qr = q[0]
qi = q[1]

Dwxp_r = (D2r*fpr - D2i*fpi)
D2xp_i = (D2r*fpi + D2i*fpr)

sr = fd.Constant(0.)
si = fd.Constant(0.)

sxpsi_r = (sr*psir - si*psii)
sxpsi_i = (sr*psii + si*psir)

eqn = ((pr*qr + pi*qi)*gamma
       - (
           qr*(D2xp_r + sxpsi_r) +
           qi*(D2xp_i + sxpsi_i)
       ))*fd.dx

p_approx = fd.Function(Q, name='approx')
p_approx_prob = fd.LinearVariationalProblem(fd.lhs(eqn),
                                            fd.rhs(eqn),
                                            p_approx)
p_approx_solver = fd.LinearVariationalSolver(p_approx_prob,
                                             solver_parameters=
                                             ud_parameters,
                                             nullspace=v_basis)

file0 = fd.File('test_formulas.pvd')
p_error = fd.Function(Q, name='error')

for i in range(M):
    D1i.assign(np.imag(D1[i]))
    D1r.assign(np.real(D1[i]))
    D2i.assign(np.imag(D2[i]))
    D2r.assign(np.real(D2[i]))
    uD_solver.solve()
    p_exact_solver.solve()
    sigma = D1[i]**2/D2[i]
    sr.assign(np.real(sigma))
    si.assign(np.imag(sigma))
    psirhs.assign(fpr)
    laplace_solver.solve()
    v, psi0 = w.split()
    psir.assign(psi0)
    psirhs.assign(fpi)
    laplace_solver.solve()
    psii.assign(psi0)
    p_approx_solver.solve()
    p_error.assign(p_approx - p_exact)
    file0.write(p_exact, p_approx, p_error, psir, psii, fpi, fpr)
