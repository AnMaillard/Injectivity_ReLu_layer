# pylint: disable=invalid-name, missing-class-docstring, missing-function-docstring, fixme, line-too-long, missing-module-docstring, pointless-string-statement, too-many-arguments

import numpy as np
from scipy.integrate import nquad
from scipy.optimize import root_scalar
from numpy.polynomial import Polynomial
from .jitted_functions.OneRSB import integrand_A0, integrand_A1, integrand_c_n, integrand_estar, integrand_logn

"""
Solver of the 1-RSB zero-temperature equations.
In this file we use chim to denote the parameter cm of the paper, and we often call estar = estar(beta = infty), which is called fstar in the paper.
"""

class OneRSB_Solver():

    def __init__(self, parameters_):
        self.verbosity = parameters_['verbosity']
        self.alpha = parameters_["alpha"]
        self.damping_q0 = parameters_["damping_q0"]
        self.damping_chi = parameters_["damping_chi"]
        self.TOL = parameters_["TOL"]
        self.TOL_chim = parameters_["TOL_chim"]
        self.q0 = 0.5
        self.chi = 0.5
        self.chim = 1e0
        self.BOUND = np.inf

        #Options for integration
        self.quad_options = {'limit':200}
        self.integration_interval = [[-self.BOUND, self.BOUND]]

    def dphi_dm(self, chim):
        #Computes the current value of dPhi/dm for the current values of q0 and chi (which we want to zero out to get chim_eq)
        q0, chi = self.q0, self.chi
        entropic_part = - (np.log(1.+chim*(1.-q0)/chi) - chim*(1-q0)*(chi+chim*(1-2*q0)) / (chi+chim*(1-q0))**2 )/(2*chim**2)
        integration_arguments = (q0, chi, chim)
        I_logn, _ = nquad(integrand_logn, self.integration_interval, args = integration_arguments, opts = self.quad_options)
        I_c_n, _ = nquad(integrand_c_n, self.integration_interval, args = integration_arguments, opts = self.quad_options)
        interaction_part = - (self.alpha / chim**2)*I_logn - (self.alpha / chim) * I_c_n
        return entropic_part + interaction_part

    def estar(self):
        #The current estimate of the ground-state energy estar
        integration_arguments = (self.q0, self.chi, self.chim)
        I_estar, _ =  nquad(integrand_estar, self.integration_interval, args = integration_arguments, opts = self.quad_options)
        return self.alpha*np.exp(-self.chim)*I_estar
    
    def fstar(self):
        #An equivalent estimate of the ground-state energy
        integration_arguments = (self.q0, self.chi, self.chim)
        I_logn, _ =  nquad(integrand_logn, self.integration_interval, args = integration_arguments, opts = self.quad_options)
        first_term = - self.q0/(2*(self.chi+self.chim*(1-self.q0))) - np.log((self.chi+self.chim*(1-self.q0))/self.chi) / (2.*self.chim)
        return first_term - self.alpha*I_logn/self.chim

    def iterate(self, chim):
        #Does an iteration of the 1RSB equations, at fixed chim
        integration_arguments = (self.q0, self.chi, chim)
        I_A0, _ = nquad(integrand_A0, self.integration_interval, args = integration_arguments, opts = self.quad_options)
        I_A1, _ = nquad(integrand_A1, self.integration_interval, args = integration_arguments, opts = self.quad_options)

        #We first iterate the equations on A0, A1
        A0 = self.alpha * I_A0/(chim * (1-self.q0) * self.chi)
        A1 = self.alpha * I_A1/(self.chi**2)

        #Now we update q0 with a polynomial solver
        assert A0/A1 > 0 and A0/A1 < 1, "ERROR: We need q* = A0/A1 to be in [0,1]"
        qstar = A0/A1

        P = Polynomial([A0**2, -2*A0*A1-A0*(A0-A1)**2*chim**2, A1**2+2*A0*(A1-A0)**2*chim**2, -A0*(A0-A1)**2*chim**2])
        solutions = P.roots()
        #Select the only solution in (qstar, 1)
        list_indices = []
        for (i, q) in enumerate(solutions):
            if q < qstar or q > 1:
                list_indices.append(i)
        solutions = np.delete(solutions, list_indices)
        assert len(solutions) == 1, f"ERROR: there is not a unique solution, we have {str(len(solutions))} solutions!"
        new_q0 = solutions[0]
        #Just a check that the original equation is satisfied
        assert np.abs(A0 - (A0-A1*new_q0)**2 / ((A0-A1)**2*new_q0*chim**2*(1-new_q0)**2)) < 1e-6, "ERROR: The original equation on q0 is not satisfied by the solution"
        #chi is then explicit
        new_chi = A0*chim*(1-new_q0)**2/(new_q0*A1-A0)

        err_q0, err_chi = np.abs(new_q0 - self.q0), np.abs(new_chi - self.chi)
        self.q0 = self.damping_q0*self.q0 + (1.-self.damping_q0)*new_q0
        self.chi = self.damping_chi*self.chi + (1.-self.damping_chi)*new_chi

        return max(err_q0, err_chi)


    def solve_fixed_chim(self, chim):
        #Solves the 1RSB equations for a fixed value of chim, returns the value of dphi_dm
        converged = False
        MAX_ITERATIONS = int(1e10)
        MIN_ITERATIONS = int(1e1)
        counter, error = 0, np.inf
        while (not(converged) and counter < MAX_ITERATIONS) or (counter < MIN_ITERATIONS):
            error = self.iterate(chim)
            counter += 1
            converged = (error < self.TOL)

        assert converged, "ERROR: the iterations did not converge"
        return self.dphi_dm(chim)

    def solve(self):
        #Solve the complete 1RSB equations by finding the value of chim using Brent's algorithm.
        #The function to zero is solve_fixed_chim

        #Finding good initialization points for the bracketing algorithm. We start with our current estimate
        chim0, dphi0 = self.chim, self.solve_fixed_chim(self.chim)
        chim1, dphi1 = chim0, dphi0
        dphi1 = dphi0
        if dphi0 < 0:
            while dphi1 < 0:
                chim0 = chim1
                chim1 *= 2.
                dphi1 = self.solve_fixed_chim(chim1)
        else:
            while dphi1 > 0:
                chim0 = chim1
                chim1 /= 2.
                dphi1 = self.solve_fixed_chim(chim1)

        assert dphi1*dphi0 < 0, f"ERROR: dphi1*dphi0 = {dphi1*dphi0} > 0 !"

        sol = root_scalar(self.solve_fixed_chim, bracket=[chim0, chim1], method='brentq')
        self.chim = sol.root
        dphi = self.solve_fixed_chim(self.chim) #This also updates the values of self.q, self.chi
        return {'chim':self.chim, 'q0':self.q0, 'chi':self.chi, 'dphi_dm': dphi, "estar": self.estar(), "fstar":self.fstar()}
