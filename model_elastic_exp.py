# Copyright (c) 2016-2018, The University of Texas at Austin 
# & University of California--Merced.
# Copyright (c) 2019-2020, The University of Texas at Austin 
# University of California--Merced, Washington University in St. Louis.
#
# All Rights reserved.
# See file COPYRIGHT for details.
#
# This file is part of the hIPPYlib library. For more information and source code
# availability see https://hippylib.github.io.
#
# hIPPYlib is free software; you can redistribute it and/or modify it under the
# terms of the GNU General Public License (as published by the Free
# Software Foundation) version 2.0 dated June 1991.

# based on continuous code

import dolfin as dl
import ufl
import math
import numpy as np
import matplotlib.pyplot as plt
import argparse

import sys
import os
sys.path.append( os.environ.get('HIPPYLIB_BASE_DIR', "../../") )
from hippylib import *

def u_boundary(x, on_boundary):
    return on_boundary and ( x[0] < dl.DOLFIN_EPS )

# Poisson's ratio
Nu = 0.25
    
# strain = 1/2 (grad u + grad u^T)
def strain(v):
    return ufl.sym(ufl.nabla_grad(v))

# stress = 2 mu strain + lambda tr(strain) I
def sigma(v):
    return (1/(1+Nu))*strain(v) + ((1*Nu)/((1+Nu)*(1-2*Nu)))*ufl.tr(strain(v))*ufl.Identity(v.geometric_dimension()) #v.cell().d

class Elasticity:
    def __init__(self, mesh, Vh, prior):
        """
        Construct a model by proving
        - the mesh
        - the finite element spaces for the STATE/ADJOINT variable and the PARAMETER variable
        - the prior information
        """
        self.mesh = mesh
        self.Vh = Vh
        
        # Initialize Expressions
        mtrue_exp = dl.Expression('0.2*(15 - 5*sin(3.1416*(x[0]/8.0 - 0.5)))',
                                element=Vh[PARAMETER].ufl_element())
        self.mtrue = dl.interpolate(mtrue_exp, self.Vh[PARAMETER]).vector()
        self.f = dl.Expression(("0.0","0.0065"),
                                element=Vh[STATE].ufl_element()) 
        self.u_o = dl.Vector()
        
        self.u_bdr = dl.Expression(("0.0","0.0"), element=Vh[STATE].ufl_element()) 
        self.u_bdr0 = dl.Expression(("0.0","0.0"), element=Vh[STATE].ufl_element()) 
        self.bc = dl.DirichletBC(self.Vh[STATE], self.u_bdr, u_boundary)
        self.bc0 = dl.DirichletBC(self.Vh[STATE], self.u_bdr0, u_boundary)
        
        # Assemble constant matrices      
        self.prior = prior
        self.Wuu = self.assembleWuu()
        

        self.computeObservation(self.u_o)
                
        self.A = None
        self.At = None
        self.C = None
        self.Wmm = None
        self.Wmu = None
        
        self.gauss_newton_approx=True
        
        self.solver = PETScKrylovSolver(mesh.mpi_comm(), "cg", amg_method())
        self.solver_fwd_inc = PETScKrylovSolver(mesh.mpi_comm(), "cg", amg_method())
        self.solver_adj_inc = PETScKrylovSolver(mesh.mpi_comm(), "cg", amg_method())
       
        self.solver.parameters["relative_tolerance"] = 1e-9
        self.solver.parameters["absolute_tolerance"] = 1e-12
        self.solver_fwd_inc.parameters = self.solver.parameters
        self.solver_adj_inc.parameters = self.solver.parameters
        
    def generate_vector(self, component="ALL"):
        """
        Return the list x=[u,m,p] where:
        - u is any object that describes the state variable
        - m is a Vector object that describes the parameter variable.
          (Need to support linear algebra operations)
        - p is any object that describes the adjoint variable
        
        If component is STATE, PARAMETER, or ADJOINT return x[component]
        """
        if component == "ALL":
            x = [dl.Vector(), dl.Vector(), dl.Vector()]
            self.Wuu.init_vector(x[STATE],0)
            self.prior.init_vector(x[PARAMETER],0)
            self.Wuu.init_vector(x[ADJOINT], 0)
        elif component == STATE:
            x = dl.Vector()
            self.Wuu.init_vector(x,0)
        elif component == PARAMETER:
            x = dl.Vector()
            self.prior.init_vector(x,0)
        elif component == ADJOINT:
            x = dl.Vector()
            self.Wuu.init_vector(x,0)
            
        return x
    
    def init_parameter(self, m):
        """
        Reshape m so that it is compatible with the parameter variable
        """
        self.prior.init_vector(m,0)
        
    def assembleA(self,x, assemble_adjoint = False, assemble_rhs = False):
        """
        Assemble the matrices and rhs for the forward/adjoint problems
        """
        trial = dl.TrialFunction(self.Vh[STATE])
        test = dl.TestFunction(self.Vh[STATE])
        m = vector2Function(x[PARAMETER], self.Vh[PARAMETER])
        Avarf = ufl.inner(ufl.exp(m)*sigma(trial), strain(test))*ufl.dx
        if not assemble_adjoint:
            bform = ufl.inner(self.f, test)*ufl.dx
            Matrix, rhs = dl.assemble_system(Avarf, bform, self.bc)
        else:
            # Assemble the adjoint of A (i.e. the transpose of A)
            u = vector2Function(x[STATE], self.Vh[STATE])
            obs = vector2Function(self.u_o, self.Vh[STATE])
            bform = ufl.inner(obs - u, test)*ufl.dx
            Matrix, rhs = dl.assemble_system(dl.adjoint(Avarf), bform, self.bc0)
            
        if assemble_rhs:
            return Matrix, rhs
        else:
            return Matrix
    
    def assembleC(self, x):
        """
        Assemble the derivative of the forward problem with respect to the parameter
        """
        trial = dl.TrialFunction(self.Vh[PARAMETER])
        test = dl.TestFunction(self.Vh[STATE])
        u = vector2Function(x[STATE], Vh[STATE])
        m = vector2Function(x[PARAMETER], Vh[PARAMETER])
        Cvarf = ufl.inner(ufl.exp(m)*trial*sigma(u), strain(test))*ufl.dx
        C = dl.assemble(Cvarf)
#        print ( "||m||", x[PARAMETER].norm("l2"), "||u||", x[STATE].norm("l2"), "||C||", C.norm("linf") )
        self.bc0.zero(C)
        return C
       
    def assembleWuu(self):
        """
        Assemble the misfit operator
        """
        trial = dl.TrialFunction(self.Vh[STATE])
        test = dl.TestFunction(self.Vh[STATE])
        varf = ufl.inner(trial, test)*ufl.dx
        Wuu = dl.assemble(varf)
        Wuu_t = Transpose(Wuu)
        self.bc0.zero(Wuu_t)
        Wuu = Transpose(Wuu_t)
        self.bc0.zero(Wuu)
        return Wuu
    
    def assembleWmu(self, x):
        """
        Assemble the derivative of the parameter equation with respect to the state
        """
        trial = dl.TrialFunction(self.Vh[STATE])
        test  = dl.TestFunction(self.Vh[PARAMETER])
        p = vector2Function(x[ADJOINT], self.Vh[ADJOINT])
        m = vector2Function(x[PARAMETER], self.Vh[PARAMETER])
        varf = ufl.inner(ufl.exp(m)*sigma(trial), strain(p))*test*ufl.dx
        Wmu = dl.assemble(varf)
        Wmu_t = Transpose(Wmu)
        self.bc0.zero(Wmu_t)
        Wmu = Transpose(Wmu_t)
        return Wmu
    
    def assembleWmm(self, x):
        """
        Assemble the derivative of the parameter equation with respect to the parameter (Newton method)
        """
        trial = dl.TrialFunction(self.Vh[PARAMETER])
        test  = dl.TestFunction(self.Vh[PARAMETER])
        u = vector2Function(x[STATE], self.Vh[STATE])
        m = vector2Function(x[PARAMETER], self.Vh[PARAMETER])
        p = vector2Function(x[ADJOINT], self.Vh[ADJOINT])
        varf = ufl.inner(ufl.exp(m)*sigma(u),strain(p))*test*trial*ufl.dx
        return dl.assemble(varf)

        
    def computeObservation(self, u_o):
        """
        Compute the synthetic observation
        """
        x = [self.generate_vector(STATE), self.mtrue, None]
        A, b = self.assembleA(x, assemble_rhs = True)
        
        A.init_vector(u_o, 1)
        dl.solve(A, u_o, b, "cg", amg_method())
        
        # Create noisy data, ud
        MAX = u_o.norm("linf")
        parRandom.normal_perturb(.01 * MAX, u_o)
    
    def cost(self, x):
        """
        Given the list x = [u,m,p] which describes the state, parameter, and
        adjoint variable compute the cost functional as the sum of 
        the misfit functional and the regularization functional.
        
        Return the list [cost functional, regularization functional, misfit functional]
        
        Note: p is not needed to compute the cost functional
        """        
        assert x[STATE] is not None
                
        diff = x[STATE] - self.u_o
        Wuudiff = self.Wuu*diff
        misfit = .5 * diff.inner(Wuudiff)
        
        Rm = dl.Vector()
        self.prior.init_vector(Rm,0)
        self.prior.R.mult(x[PARAMETER], Rm)
        reg = .5 * x[PARAMETER].inner(Rm)
        
        cost = misfit + reg
        
        return cost, reg, misfit
    
    def solveFwd(self, out, x):
        """
        Solve the forward problem.
        """
        A, b = self.assembleA(x, assemble_rhs = True)
        A.init_vector(out, 1)

        self.solver.set_operator(A)
        self.solver.solve(out,b)

    
    def solveAdj(self, out, x):
        """
        Solve the adjoint problem.
        """
        At, badj = self.assembleA(x, assemble_adjoint = True,assemble_rhs = True)
        At.init_vector(out, 1)
                    
        self.solver.set_operator(At)
        self.solver.solve(out,badj)
        
#        print ("ADJ", (self.At*out - badj).norm("l2")/badj.norm("l2"), nit)
    
    def evalGradientParameter(self,x, mg, misfit_only=False):
        """
        Evaluate the gradient for the variation parameter equation at the point x=[u,m,p].
        Parameters:
        - x = [u,m,p] the point at which to evaluate the gradient.
        - mg the variational gradient (g, mtest) being mtest a test function in the parameter space
          (Output parameter)
        
        Returns the norm of the gradient in the correct inner product g_norm = sqrt(g,g)
        """ 
        C = self.assembleC(x)

        self.prior.init_vector(mg,0)
        C.transpmult(x[ADJOINT], mg)
        if misfit_only == False:
            Rm = dl.Vector()
            self.prior.init_vector(Rm,0)
            self.prior.R.mult(x[PARAMETER], Rm)   
            mg.axpy(1., Rm)
        
        g = dl.Vector()
        self.prior.init_vector(g,1)
        
        self.prior.Msolver.solve(g, mg)
        g_norm = np.sqrt( g.inner(mg) )
        
        return g_norm
        
    
    def setPointForHessianEvaluations(self, x, gauss_newton_approx=False):  
        """
        Specify the point x = [u,m,p] at which the Hessian operator (or the Gauss-Newton approximation)
        need to be evaluated.
        """  
        self.gauss_newton_approx = gauss_newton_approx    
        self.A  = self.assembleA(x)
        self.At = self.assembleA(x, assemble_adjoint=True )
        self.C  = self.assembleC(x)
        if gauss_newton_approx:
            self.Wmu = None
            self.Wmm = None
        else:
            self.Wmu = self.assembleWmu(x)
            self.Wmm = self.assembleWmm(x)
            
        self.solver_fwd_inc.set_operator(self.A)
        self.solver_adj_inc.set_operator(self.At)

        
    def solveFwdIncremental(self, sol, rhs):
        """
        Solve the incremental forward problem for a given rhs
        """
        self.A.init_vector(sol,1)
        self.solver_fwd_inc.solve(sol,rhs)
        
    def solveAdjIncremental(self, sol, rhs):
        """
        Solve the incremental adjoint problem for a given rhs
        """            
        self.At.init_vector(sol,1)
        self.solver_adj_inc.solve(sol, rhs)
#        print ("AdjInc", (self.At*sol-rhs).norm("l2")/rhs.norm("l2"), nit)
    
    def applyC(self, dm, out):
        self.C.mult(dm,out)
    
    def applyCt(self, dp, out):
        self.C.transpmult(dp,out)
    
    def applyWuu(self, du, out, gn_approx=False):
        self.Wuu.mult(du, out)
    
    def applyWum(self, dm, out):
        if self.gauss_newton_approx:
            out.zero()
        else:
            self.Wmu.transpmult(dm,out)

    
    def applyWmu(self, du, out):
        if self.gauss_newton_approx:
            out.zero()
        else:
            self.Wmu.mult(du, out)
    
    def applyR(self, dm, out):
        self.prior.R.mult(dm, out)
        
    def Rsolver(self):        
        return self.prior.Rsolver
    
    def applyWmm(self, dm, out):
        if self.gauss_newton_approx:
            out.zero()
        else:
            self.Wmm.mult(dm, out)
            



        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Model Elastic Beam')
    parser.add_argument('--nx',
                        default=100,
                        type=int,
                        help="Number of elements in x-direction")
    parser.add_argument('--ny',
                        default=20,
                        type=int,
                        help="Number of elements in y-direction")
    args = parser.parse_args()
    sep = "\n"+"#"*80+"\n"
    try:
        dl.set_log_active(False)
    except:
        pass
    #nx = args.nx
    #ny = args.ny
    #mesh = dl.UnitSquareMesh(nx, ny)
    nx = 100
    ny = 20
    mesh = dl.RectangleMesh(dl.Point(0, 0),dl.Point(8,0.5),nx,ny, "right")
    
    rank = dl.MPI.rank(mesh.mpi_comm())
    nproc = dl.MPI.size(mesh.mpi_comm())
        
    Vh2 = dl.VectorFunctionSpace(mesh, 'Lagrange', 1)
    Vh1 = dl.FunctionSpace(mesh, 'Lagrange', 1)
    Vh = [Vh2, Vh1, Vh2]
    
    prior = BiLaplacianPrior(Vh[PARAMETER], gamma=4e-1, delta=1e-1)
    model = Elasticity(mesh, Vh, prior)
        
    m0 = dl.interpolate(dl.Expression("x[0] + 2.2", element=Vh[PARAMETER].ufl_element()), Vh[PARAMETER])
    modelVerify(model, m0.vector(), is_quadratic = False, verbose = (rank==0))

    m0 = dl.interpolate(dl.Constant(1.0),Vh[PARAMETER])
    parameters = ReducedSpaceNewtonCG_ParameterList()
    parameters["rel_tolerance"] = 1e-12
    parameters["abs_tolerance"] = 1e-12
    parameters["max_iter"]      = 50
    parameters["globalization"] = "LS"
    parameters["GN_iter"] = 6
    if rank != 0:
        parameters["print_level"] = -1
        
    solver = ReducedSpaceNewtonCG(model, parameters)

    
    x = solver.solve([None, m0.vector(), None])
    
    if rank == 0:
        if solver.converged:
            print ("\nConverged in ", solver.it, " iterations.")
        else:
            print ("\nNot Converged")

        print ("Termination reason: ", solver.termination_reasons[solver.reason])
        print ("Final gradient norm: ", solver.final_grad_norm)
        print ("Final cost: ", solver.final_cost)
    
    model.setPointForHessianEvaluations(x, gauss_newton_approx=False)
    Hmisfit = ReducedHessian(model, misfit_only=True)
    p = 25
    k = 50
    Omega = MultiVector(x[PARAMETER], k+p)
    parRandom.normal(1., Omega)

    d, U = doublePassG(Hmisfit, prior.R, prior.Rsolver, Omega, k, s=1, check=False)

    posterior = GaussianLRPosterior(prior, d, U)
    posterior.mean = x[PARAMETER]
    
    xxname = ["state", "parameter", "adjoint"]
    xx = [vector2Function(x[i], Vh[i], name=xxname[i]) for i in range(len(Vh))]
    
    with dl.XDMFFile(mesh.mpi_comm(), "results/results.xdmf") as fid:
        fid.parameters["functions_share_mesh"] = True
        fid.parameters["rewrite_function_mesh"] = False 
           
        fid.write(xx[STATE],0)
        fid.write(xx[PARAMETER],0)
        fid.write(vector2Function(model.mtrue, Vh[PARAMETER], name = "true parameter"), 0)
        fid.write(vector2Function(prior.mean, Vh[PARAMETER], name = "prior mean"), 0)
        fid.write(xx[ADJOINT],0)
        fid.write(vector2Function(model.u_o, Vh[STATE], name = "observation"), 0)
    
    U.export(Vh[PARAMETER], "results/evect.xdmf", varname = "gen_evects", normalize = True)
    if rank == 0:
        np.savetxt("results/eigevalues.dat", d)

    if rank == 0:
        plt.figure()
        plt.plot(range(0,k), d, 'b*',range(0,k), np.ones(k), '-r')
        plt.yscale('log')
        plt.show()
    

    compute_trace = True
    if compute_trace:
        post_tr, prior_tr, corr_tr = posterior.trace(method="Estimator", tol=1e-2, min_iter=20, max_iter=20)
        print( "Posterior trace {0:5e}; Prior trace {1:5e}; Correction trace {2:5e}".format(post_tr, prior_tr, corr_tr) )
    post_pw_variance, pr_pw_variance, corr_pw_variance = posterior.pointwise_variance(method="Exact")

    objs = [dl.Function(Vh[PARAMETER], pr_pw_variance), dl.Function(Vh[PARAMETER], post_pw_variance)]
    mytitles = ["Prior variance", "Posterior variance"]
    nb.multi1_plot(objs, mytitles, logscale=False)
    plt.show()

    fid = dl.File("results/pointwise_variance.pvd")
    fid << dl.Function(Vh[PARAMETER], post_pw_variance, name="Posterior")
    fid << dl.Function(Vh[PARAMETER], pr_pw_variance, name="Prior")
    fid << dl.Function(Vh[PARAMETER], corr_pw_variance, name="Correction")
    
    #plt.figure()
    #plt.plot(range(0,k), d, 'ob')
    #plt.yscale('log')
    #plt.title("Spectrum of data misfit Hessian")

    print(sep, "Generate samples from Prior and Posterior\n","Export generalized Eigenpairs", sep)
    fid_prior = dl.File("samples/sample_prior.pvd")
    fid_post  = dl.File("samples/sample_post.pvd")
    nsamples = 8
    noise = dl.Vector()
    posterior.init_vector(noise,"noise")
    s_prior = dl.Function(Vh[PARAMETER], name="sample_prior")
    s_post = dl.Function(Vh[PARAMETER], name="sample_post")

    pr_max   =  2.5*math.sqrt( pr_pw_variance.max() ) + prior.mean.max()
    pr_min   = -2.5*math.sqrt( pr_pw_variance.max() ) + prior.mean.min()
    ps_max   =  2.5*math.sqrt( post_pw_variance.max() ) + posterior.mean.max()
    ps_min   = -2.5*math.sqrt( post_pw_variance.max() ) + posterior.mean.min()

    for i in range(nsamples):
        parRandom.normal(1., noise)
        posterior.sample(noise, s_prior.vector(), s_post.vector())
        plt.figure(figsize=(15,5))
        fid_prior << s_prior
        fid_post << s_post
        nb.plot(s_prior, subplot_loc=121,mytitle="Prior sample", vmin=pr_min, vmax=pr_max)
        nb.plot(s_post, subplot_loc=122,mytitle="Posterior sample", vmin=ps_min, vmax=ps_max)
        plt.show()
        
        
    # save and plot
    xx = [dl.Function(Vh[i], x[i]) for i in range(len(Vh))]
    dl.File("results/u_state.pvd") << xx[STATE]
    dl.File("results/E_parameter_inv.pvd") << xx[PARAMETER]
    atrue = dl.interpolate(dl.Expression('0.2*(15 - 5*sin(3.1416*(x[0]/8.0 - 0.5)))',element=Vh[PARAMETER].ufl_element()), Vh[PARAMETER])
    dl.File("results/E_parameter_true.pvd") << atrue
    nb.plot(atrue, mytitle = "True Parameter")
    #nb.plot(xx[STATE], mytitle = "State", mode="displacement")
    #nb.plot(xx[PARAMETER], mytitle = "Parameter")
    #nb.plot(xx[ADJOINT], mytitle = "Adjoint")
    #interactive()
    plt.show()
