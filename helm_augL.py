#standalone solver for the block problem for paradiag applied to the wave
#equation
import firedrake as fd
import numpy as np
from scipy.fft import fft, ifft

n = 20
mesh = fd.PeriodicUnitSquareMesh(n, n)
degree = 0
V = fd.VectorFunctionSpace(mesh, "BDM", degree+1)
Q = fd.VectorFunctionSpace(mesh, "DG", degree)
W = V * Q

#
# from IPython import embed; embed()


u, p = fd.TrialFunctions(W)
ur = u[0, :]
ui = u[1, :]
pr = p[0]
pi = p[1]

v, q = fd.TestFunctions(W)
vr = v[0, :]
vi = v[1, :]
qr = q[0]
qi = q[1]

#number of steps
T = 1
M = 20
Dt = T/M
#timestep
print(Dt)
# Dt = 0.01
# Dt = 10
#circulant coefficient
alphav = 0.01
#timestep offset parameter
thetav = 0.5
gamma = fd.Constant(1.0e5)

old_method = False
old_method = True

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
# sigma = (D1r * D2r - D1i * D2i)/(D2r**2 + D2i**2)
sigma = (D1/D2)**2

sgr = fd.Constant(0)
sgi = fd.Constant(0)

if old_method:
    D1t = D1
    D2t = D2
else:
    D1t = D1*(D2)
    D2t = D2*(D2)
D1ti = fd.Constant(np.imag(D1t[i]))
D1tr = fd.Constant(np.real(D1t[i]))
D2ti = fd.Constant(np.imag(D2t[i]))
D2tr = fd.Constant(np.real(D2t[i]))


#d1*u + d2*grad h = ...
#d1*h + d2*div u = ...
#u = ur + i*ui
#p = pr + i*pi

# d_2*M
# (d2r * I + d2i * J) otimes M
# I otimes M

#(d1r + i*d1i)*(ur + i*ui) + (d2r + i*d2i)*grad(pr + i*pi)
# = d1r*ur - d1i*ui + grad(d2r*pr - d2i*pi)
#  + i*(d1r*ui + d1i*ur) + grad(d2i*pr + d2r*pi)
x, y = fd.SpatialCoordinate(mesh)
fu = fd.as_vector([fd.exp(x*(1-x) + y*(1-y)), fd.sin(x+y)])
# fp = fd.cos(1 + x*(1-x) + y*(1-y))
fp = fd.exp(fd.cos(2*fd.pi*x)*fd.cos(2*fd.pi*y))
# F = fd.inner(vr, fu)*fd.dx + fp*qi*fd.dx
F = fp*qi*fd.dx
a = (
    fd.inner(vr, D1r*ur - D1i*ui)
    - fd.div(vr)*(D2r*pr - D2i*pi)
    + fd.inner(vi, D1i*ur + D1r*ui)
    - fd.div(vi)*(D2i*pr + D2r*pi)
    + qr*(D1r*pr - D1i*pi)
    + qr*fd.div(D2r*ur - D2i*ui)
    + qi*(D1i*pr + D1r*pi)
    + qi*fd.div(D2i*ur + D2r*ui)
    + gamma*(fd.div(vr)*(D1tr*pr - D1ti*pi)
             + fd.div(vr)*fd.div(D2tr*ur - D2ti*ui)
             + fd.div(vi)*(D1ti*pr + D1tr*pi)
             + fd.div(vi)*fd.div(D2ti*ur + D2tr*ui))
    )*fd.dx

w = fd.Function(W)
fd.assemble(F)
fd.assemble(a)
fd.action(a, w)
prob = fd.LinearVariationalProblem(a, F, w, constant_jacobian=False)

diag_parameters = {
    "mat_type": "matfree",
    "ksp_type": "fgmres",
    "ksp_converged_reason": None,
    # "ksp_monitor": None,
    "ksp_atol": 1.0e-6,
    "pc_type": "fieldsplit",
    "pc_fieldsplit_type": "schur",
    "pc_fieldsplit_schur_fact_type": "full",
    "pc_fieldsplit_off_diag_use_amat": True,
}

# bottomright = {
#     "ksp_type": "gmres",
#     "ksp_max_it": 3,
#     "pc_type": "python",
#     # "pc_python_type": "firedrake.MassInvPC",
#     "pc_python_type": "asQ.MassInvPC",
#     "Mp_pc_type": "ilu"
# }

# change the bottom right part:
bottomright = {
    "ksp_type": "gmres",
    "ksp_max_it": 10,
    # "ksp_converged_reason": None,
    "pc_type": "composite",
    "pc_composite_type": "multiplicative",
    "pc_composite_pcs": "python,python",
    "sub_0_pc_python_type": "firedrake.MassInvPC",
    "sub_0_Mp_pc_type": "ilu",
    "sub_1_pc_python_type": "asQ.HelmholtzPC",
    "sub_1_Hp_pc_type": "lu"
}

# fd.MassInvPC

diag_parameters["fieldsplit_1"] = bottomright

topleft_LU = {
    "ksp_type": "preonly",
    "pc_type": "python",
    "pc_python_type": "firedrake.AssembledPC",
    "assembled_pc_type": "lu",
    "assembled_pc_factor_mat_solver_type": "mumps"
}

diag_parameters["fieldsplit_0"] = topleft_LU

# direct solver parameters
diag_parameters_dir = {
    'ksp_type': 'preonly',
    'pc_type': 'lu',
    'ksp_monitor': None,
    'pc_factor_mat_solver_type': 'mumps',
    'mat_type': 'aij'}


v_basis = fd.VectorSpaceBasis(constant=True)
nullspace = fd.MixedVectorSpaceBasis(W, [W.sub(0), v_basis])
solver = fd.LinearVariationalSolver(prob, solver_parameters=diag_parameters,
                                    appctx={"sgr": sgr, "sgi": sgi})

solver_dir = fd.LinearVariationalSolver(prob, solver_parameters=diag_parameters_dir,
                                    appctx={"sgr": sgr, "sgi": sgi})


u, D = w.split()
err = fd.Function(W)


file0 = fd.File("output/output.pvd")

for i in range(M):
    # print(i, D1[i], D2[i], D1[i]/D2[i], np.abs(D1[i]/D2[i]))
    w.assign(0.)
    sigma = (D1[i]/D2[i])**2
    sgr.assign(np.real(sigma))
    sgi.assign(np.imag(sigma))
    D1i.assign(np.imag(D1[i]))
    D1r.assign(np.real(D1[i]))
    D2i.assign(np.imag(D2[i]))
    D2r.assign(np.real(D2[i]))
    D1ti.assign(np.imag(D1t[i]))
    D1tr.assign(np.real(D1t[i]))
    D2ti.assign(np.imag(D2t[i]))
    D2tr.assign(np.real(D2t[i]))
    solver.solve()
    # err.assign(w)
    # solver_dir.solve()
    # err -= w
    # err_u, err_D = err.split()
    # file0.write(u, D,err_u,err_D)
    print(fd.norm(w))
