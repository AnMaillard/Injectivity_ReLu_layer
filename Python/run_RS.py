# pylint: disable=invalid-name, missing-class-docstring, missing-function-docstring, fixme, line-too-long, missing-module-docstring, multiple-imports, pointless-string-statement

import time, pickle
from math import erf
from tqdm import tqdm
import numpy as np
from scipy.integrate import quad
from scipy import optimize, LowLevelCallable
import numba as nb
from numba import cfunc,carray
from numba.types import intc, CPointer, float64
from Class.jitted_functions.OneRSB import H, H1, H2, Gauss

"""
Replica-symmetric computations for the injectivity threshold. This file computes:
- The RS prediction for the ground state energy over a range of alphas.
- The injectivity threshold predicted by RS computations
- The d-AT lower bound
"""

#Some numba decorators
def jit_2(integrand_function):
    jitted_function = nb.njit(integrand_function)
    @cfunc(float64(intc, CPointer(float64)))
    def wrapped(n_, xx):
        values = carray(xx,n_)
        return jitted_function(values[0], values[1])
    return LowLevelCallable(wrapped.ctypes)

def jit_1(integrand_function):
    jitted_function = nb.njit(integrand_function)
    @cfunc(float64(intc, CPointer(float64)))
    def wrapped(n_, xx):
        values = carray(xx,n_)
        return jitted_function(values[0])
    return LowLevelCallable(wrapped.ctypes)

def q_RS(alpha, beta): #RS prediction for the overlap at finite beta < infty
    @jit_2
    def integrand(xi, x):
        #We use the auxiliary variable x = q / (1-q) >= 0
        return alpha*Gauss(xi, 0, 1)*(1-np.exp(-beta))*xi*H1(xi*np.sqrt(x)) / (1 - (1-np.exp(-beta))*H(xi*np.sqrt(x)) )
    def q_eq(x):
        q = x / (1.+x)
        integral, _ = quad(integrand, -np.inf, np.inf, args = (x,))
        return q*np.sqrt(x) - integral
    X_THRESHOLD = 1e5
    start, finish = q_eq(1./X_THRESHOLD), q_eq(X_THRESHOLD)
    if start*finish < 0:
        #We use brentq to find the value of q that satisfies it outside zero
        root = optimize.brentq(q_eq, 1./X_THRESHOLD, X_THRESHOLD)
    else: #Then the solution is q = 0
        root = 0
    return root / (1.+root)

def estar_RS(alpha, beta): #RS prediction for the energy
    q = q_RS(alpha, beta)
    x = q / (1.-q)
    @jit_1
    def integrand(xi):
        return alpha*np.exp(-beta)*Gauss(xi, 0, 1)*H(xi*np.sqrt(x)) / (1 - (1-np.exp(-beta))*H(xi*np.sqrt(x)) )
    integral, _ = quad(integrand, -np.inf, np.inf)
    return integral

def beta_star(alpha, beta_MAX_): #Computes betastar(alpha) according to the RS solution
    e_beta_MAX = estar_RS(alpha, beta_MAX_)
    if e_beta_MAX >= 1:
        return beta_MAX_
    else:
        root = optimize.brentq(lambda beta: estar_RS(alpha, beta) - 1, 0., beta_MAX_)
        return root

def AT_stable(alpha, beta): #RS stability condition
    q = q_RS(alpha, beta)
    @jit_2
    def integrand(xi, x):
        #This is (1-q)^2*f''(Sqrt[q]*xi)^2
        return Gauss(xi, 0, 1)*( (1-np.exp(-beta))*H2(-xi*np.sqrt(x)) / (1 - (1-np.exp(-beta))*H(-xi*np.sqrt(x))) + (1-np.exp(-beta))**2*H1(-xi*np.sqrt(x))**2 / (1 - (1-np.exp(-beta))*H(-xi*np.sqrt(x)))**2 )**2
    x = q / (1.-q)
    integral, _ = quad(integrand, -np.inf, np.inf, args = (x,))
    return (1./alpha) - integral

if __name__== "__main__":

    global_seed = int(time.time())
    np.random.seed(global_seed)

    #The computation of RS predictions for all alpha
    def to_zero(chi, alpha):
        return alpha*(erf(np.sqrt(chi))/2. - np.exp(-chi)*np.sqrt(chi/np.pi)) - 1. #chi is given by the zero to this equation

    def fstar(chi, alpha):
        return alpha*H(np.sqrt(2*chi))

    def injectivity_threshold(alpha):
        solution = optimize.root_scalar(to_zero, bracket=[0, 10], method='brentq', args = (alpha,))
        return fstar(solution.root, alpha) - 1.

    alphas = np.linspace(3, 10, 1000)
    estars, chis = np.zeros_like(alphas), np.zeros_like(alphas)
    for i in tqdm(range(np.size(alphas)), desc = 'Iteration over alphas', leave = True):
        alpha_ = alphas[i]
        sol = optimize.root_scalar(to_zero, bracket=[0, 10], method='brentq', args = (alpha_,))
        chis[i] = sol.root
        estars[i] = fstar(chis[i], alpha_)

    #The RS injectivity threshold prediction
    sol = optimize.root_scalar(injectivity_threshold, bracket=[3, 10], method='brentq', xtol = 1e-7)
    alphainj_RS = sol.root

    tqdm.write(f"Found alphainj_RS = {alphainj_RS}")

    #The d-AT lower bound
    beta_MAX = 10. #The highest possible value for the inverse temperature.
    sol = optimize.root_scalar(lambda alpha: AT_stable(alpha, beta_star(alpha, beta_MAX)), bracket=[3., alphainj_RS], xtol = 1e-7)
    alpha_dAT = sol.root
    # We check that the threshold value on beta is not the value of beta^star(alpha_dAT)
    assert beta_star(alpha_dAT, beta_MAX) < beta_MAX, f"ERROR: We have reached the threshold beta_MAX = {beta_MAX} for alpha_dAT = {alpha_dAT}"

    tqdm.write(f"Found alpha_dAT = {alpha_dAT}")

    output = {'alphainj_RS':alphainj_RS, 'alpha_dAT':alpha_dAT, 'alphas_RS':alphas, 'chis_RS':chis, 'estars_RS':estars}
    filename = "Data/RS_results.pkl"
    outfile = open(filename,'wb')
    pickle.dump(output,outfile)
    outfile.close()

    tqdm.write("All done!")
