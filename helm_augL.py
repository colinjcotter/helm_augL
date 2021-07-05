#standalone solver for the block problem for paradiag applied to the wave
#equation
import firedrake as fd
import numpy as np
from scipy.fft import fft, ifft

import petsc4py.PETSc as PETSc
PETSc.Sys.popErrorHandler()

class IPHelmholtzPC(fd.PCBase):

    needs_python_pmat = True

    def initialize(self, pc):
        from firedrake import TrialFunction, TestFunction, dx, assemble, inner, parameters
        prefix = pc.getOptionsPrefix()
        options_prefix = prefix + "Hp_"
        # we assume P has things stuffed inside of it
        _, P = pc.getOperators()
        context = P.getPythonContext()

        test, trial = context.a.arguments()

        if test.function_space() != trial.function_space():
            raise ValueError("MassInvPC only makes sense if test and trial space are the same")

        V = test.function_space()
        mesh = V.mesh()

        # Input/Output wrapper Functions
        self.xf = fd.Function(V)  # input
        self.yf = fd.Function(V)  # output

        mu = context.appctx.get("mu", 1.0)
        D1r = context.appctx.get("D1r", None)
        D1i = context.appctx.get("D1i", None)
        sr = context.appctx.get("sr", None)
        si = context.appctx.get("si", None)
        gamma = context.appctx.get("gamma", None)

        self.D1r = D1r
        self.D1i = D1i
        self.sr = sr
        self.si = si

        u = fd.TrialFunction(V)
        v = fd.TestFunction(V)

        #mass solve
        m = fd.inner(u,v)*fd.dx
        sp = {
            "ksp_type":"preonly",
            "pc_type": "bjacobi",
            "sub_pc_type":"ilu"
        }
        self.mSolver = fd.LinearSolver(assemble(m),
                                       solver_parameters=sp)
        vr = v[0]
        vi = v[1]
        ur = u[0]
        ui = u[1]
        xr = self.xf[0]
        xi = self.xf[1]

        def get_laplace(q,phi):
            h = fd.avg(fd.CellVolume(mesh))/fd.FacetArea(mesh)
            eta = fd.Constant(10.)
            mu = eta/h
            n = fd.FacetNormal(mesh)
            ad = (- inner(2 * fd.avg(phi*n),
                          fd.avg(fd.grad(q)))
                  - inner(fd.avg(fd.grad(phi)),
                          2 * fd.avg(q*n))
                  + mu * inner(2 * fd.avg(phi*n),
                               2 * fd.avg(q*n))) * fd.dS
            ad += inner(fd.grad(q), fd.grad(phi)) * fd.dx
            return ad

        D2u_r = D2r*ur - D2i*ui
        D2u_i = D2i*ur + D2r*ui
        su_r = sr*ur - si*ui
        su_i = si*ur + sr*ui

        self.D2r = D2r
        self.D2i = D2i
        self.sr = sr
        self.si = si

        a = vr * D2u_r * dx + get_laplace(vr, su_r)
        a += vi * D2u_i * dx + get_laplace(vi, su_i)
        L = get_laplace(xr, vr/gamma) + get_laplace(xi, vi/gamma)

        Hprob = fd.LinearVariationalProblem(a, L, self.yf,
                                            constant_jacobian=False)
        Hparameters = {
            "ksp_type":"preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps"
        }
        nullspace = fd.VectorSpaceBasis(constant=True)
        self.solver = fd.LinearVariationalSolver(Hprob,
                                                 #nullspace = nullspace,
                                                 solver_parameters
                                                 = Hparameters)

    def update(self, pc):
        pass

    def apply(self, pc, x, y):
        # copy petsc vec into Function
        with self.xf.dat.vec_wo as v:
            x.copy(v)

        #solve mass matrix on xf, put solution in yf
        self.mSolver.solve(self.yf, self.xf)
        #copy into yf for RHS of Helmholtz operator
        self.xf.assign(self.yf)
        #Do Helmholtz solve
        self.solver.solve()

        # copy Function into petsc vec
        with self.yf.dat.vec_ro as v:
            v.copy(y)

    def applyTranspose(self, pc, x, y):
        raise NotImplementedError

class MixedHelmholtzPC(fd.PCBase):
    #rather tautological
    needs_python_pmat = True

    def initialize(self, pc):
        from firedrake import TrialFunction, TestFunction, dx, assemble, inner, parameters
        prefix = pc.getOptionsPrefix()
        options_prefix = prefix + "Hp_"
        # we assume P has things stuffed inside of it
        _, P = pc.getOperators()
        context = P.getPythonContext()

        test, trial = context.a.arguments()

        if test.function_space() != trial.function_space():
            raise ValueError("MassInvPC only makes sense if test and trial space are the same")

        Q = test.function_space()
        mesh = Q.mesh()

        # Input/Output wrapper Functions
        self.xf = fd.Function(V)  # input
        self.yf = fd.Function(V)  # output

        D1r = context.appctx.get("D1r", None)
        D1i = context.appctx.get("D1i", None)
        sr = context.appctx.get("sr", None)
        si = context.appctx.get("si", None)
        gamma = context.appctx.get("gamma", None)
        W = context.appctx.get("W", None)
        V = context.appctx.get("V", None)

        self.D1r = D1r
        self.D1i = D1i
        self.sr = sr
        self.si = si

        p = fd.TrialFunction(Q)
        q = fd.TestFunction(Q)

        #mass solve
        m = fd.inner(p,q)*fd.dx
        sp = {
            "ksp_type":"preonly",
            "pc_type": "bjacobi",
            "sub_pc_type":"ilu"
        }
        self.mSolver = fd.LinearSolver(assemble(m),
                                       solver_parameters=sp)

        #Applying the gradient
        v = fd.TrialFunction(V)
        u = fd.TestFunction(V)
        ur = u[0, :]
        ui = u[1, :]
        vr = v[0, :]
        vi = v[1, :]
        
        a = fd.inner(v,u)*fd.dx 
        L = fd.div(vr)*
        
        Hprob = fd.LinearVariationalProblem(a, L, self.yf,
                                            constant_jacobian=False)
        Hparameters = {
            "ksp_type":"preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps"
        }
        nullspace = fd.VectorSpaceBasis(constant=True)
        self.solver = fd.LinearVariationalSolver(Hprob,
                                                 #nullspace = nullspace,
                                                 solver_parameters
                                                 = Hparameters)

    def update(self, pc):
        pass

    def apply(self, pc, x, y):
        # copy petsc vec into Function
        with self.xf.dat.vec_wo as v:
            x.copy(v)

        #solve mass matrix on xf, put solution in yf
        self.mSolver.solve(self.yf, self.xf)
        #copy into yf for RHS of Helmholtz operator
        self.xf.assign(self.yf)
        #Do Helmholtz solve
        self.solver.solve()

        # copy Function into petsc vec
        with self.yf.dat.vec_ro as v:
            v.copy(y)

    def applyTranspose(self, pc, x, y):
        raise NotImplementedError


n = 20
mesh = fd.PeriodicUnitSquareMesh(n, n)
degree = 1
V = fd.VectorFunctionSpace(mesh, "BDM", degree+1)
Q = fd.VectorFunctionSpace(mesh, "DG", degree)
Q0 = fd.FunctionSpace(mesh, "DG", degree)
W = V * Q

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
T = 1.
M = 20
Dt = T/M
#timestep
#circulant coefficient
alphav = 0.01
#timestep offset parameter
thetav = 0.5
gamma = fd.Constant(1.0e4)

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
p1 = fd.sin(fd.pi*x)*fd.sin(fd.pi*y)
p2 = fd.cos(2*fd.pi*x)*fd.sin(4*fd.pi*y)
fur = fd.as_vector([fd.exp(p1+2*p2), fd.cos(2*p2 - p1)])
fui = fd.as_vector([fd.sin(3*p1+2*p2), fd.exp(3*p2 + 2*p1)])
fpr = fd.Function(Q0).interpolate(fd.cos(fd.sin(p1+p2)))
fpi = fd.Function(Q0).interpolate(fd.sin(fd.exp(2*p1-p2)))
one = fd.Function(Q0).assign(1.)
fpr -= fd.assemble(fpr*fd.dx)/fd.assemble(one*fd.dx)
fpi -= fd.assemble(fpi*fd.dx)/fd.assemble(one*fd.dx)
F = fd.inner(vr, fur)*fd.dx + fd.inner(vi, fui)*fd.dx
F += (fpr*qr + fpi*qi)*fd.dx
D1xu_r = D1r*ur - D1i*ui
D2xu_r = D2r*ur - D2i*ui
D1xp_r = D1r*pr - D1i*pi
D2xp_r = D2r*pr - D2i*pi
D1xu_i = D1r*ui + D1i*ur
D2xu_i = D2r*ui + D2i*ur
D1xp_i = D1r*pi + D1i*pr
D2xp_i = D2r*pi + D2i*pr

a = (
    fd.inner(vr, D1xu_r) - fd.div(vr)*D2xp_r
    + fd.inner(vi, D1xu_i) - fd.div(vi)*D2xp_i
    + qr*(D1xp_r + fd.div(D2xu_r))
    + qi*(D1xp_i + fd.div(D2xu_i))
    + gamma*(
        fd.div(vr)*(D1xp_r + fd.div(D2xu_r))
        + fd.div(vi)*(D1xp_i + fd.div(D2xu_i))
    )
    )*fd.dx

w = fd.Function(W)
prob = fd.LinearVariationalProblem(a, F, w, constant_jacobian=False)

diag_parameters = {
    "mat_type": "matfree",
    "ksp_type": "preonly",
    "ksp_monitor": None,
    "ksp_atol": 1.0e-6,
    "pc_type": "fieldsplit",
    "pc_fieldsplit_type": "schur",
    "pc_fieldsplit_schur_fact_type": "full",
    "pc_fieldsplit_off_diag_use_amat": True,
}

# change the bottom right part:
bottomright = {
    "ksp_type": "gmres",
    "ksp_gmres_modifiedgramschmidt": None,
    "ksp_max_it": 50,
    "ksp_converged_reason": None,
    "pc_type": "python",
    "pc_python_type": "__main__.IPHelmholtzPC"
}

diag_parameters["fieldsplit_1"] = bottomright

topleft_LU = {
    "ksp_type": "preonly",
    "pc_type": "python",
    "pc_python_type": "firedrake.AssembledPC",
    "assembled_pc_type": "lu",
    "assembled_pc_factor_mat_solver_type": "mumps"
}

diag_parameters["fieldsplit_0"] = topleft_LU

sr = fd.Constant(0.)
si = fd.Constant(0.)

v_basis = fd.VectorSpaceBasis(constant=True)
nullspace = fd.MixedVectorSpaceBasis(W, [W.sub(0), v_basis])
solver = fd.LinearVariationalSolver(prob, solver_parameters=diag_parameters,
                                    #nullspace=nullspace,
                                    appctx={"D1r": D1r, "D1i": D1i,
                                            "sr": sr, "si": si,
                                            "gamma": gamma,
                                            "W": W, "V": V})

u, D = w.split()
err = fd.Function(W)


file0 = fd.File("output/output.pvd")

for i in range(M):
    w.assign(0.)
    sigma = D1[i]**2/D2[i]
    sr.assign(np.real(sigma))
    si.assign(np.imag(sigma))
    D1i.assign(np.imag(D1[i]))
    D1r.assign(np.real(D1[i]))
    D2i.assign(np.imag(D2[i]))
    D2r.assign(np.real(D2[i]))
    solver.solve()
