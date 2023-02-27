# pylint: disable=invalid-name, missing-class-docstring, missing-function-docstring, fixme, line-too-long, missing-module-docstring, multiple-imports, pointless-string-statement

import math
import numpy as np
import numba as nb
from numba import cfunc,carray
from numba.types import intc, CPointer, float64
from scipy import LowLevelCallable

"""
Auxiliary functions for the 1-RSB calculations, in particular the functions n, a, b, c.
Here we use Numba, for faster CPU-only calculations using low level callable C functions.
"""

#Depending on the number of variables to unpack. Here, 4
def jit_4(integrand_function):
    jitted_function = nb.njit(integrand_function)
    @cfunc(float64(intc, CPointer(float64)))
    def wrapped(n_, xx):
        values = carray(xx,n_)
        return jitted_function(values[0], values[1], values[2], values[3])
    return LowLevelCallable(wrapped.ctypes)

@nb.njit(error_model="numpy",fastmath=True)
def Gauss(u, m, v):
    return np.exp(-(u-m)**2/(2.*v))/np.sqrt(2*np.pi*v)

#H and its derivatives
@nb.njit(error_model="numpy",fastmath=True)
def H(x):
    return (1. - math.erf(x/np.sqrt(2.)))/2.

@nb.njit(error_model="numpy",fastmath=True)
def H1(x):
    return - Gauss(x, 0, 1)

@nb.njit(error_model="numpy",fastmath=True)
def H2(x):
    return x*Gauss(x, 0, 1)

@nb.njit(error_model="numpy",fastmath=True)
def Ineg(xi, q): #Function checked numerically
    return H(np.sqrt(q/(1.-q))*xi)

@nb.njit(error_model="numpy",fastmath=True)
def I2chi(xi, q, chi): #Function checked numerically
    return H((np.sqrt(2*chi) - np.sqrt(q)*xi)/np.sqrt(1.-q))

@nb.njit(error_model="numpy",fastmath=True)
def I0(xi, q, chi, chim): #Function checked numerically
    A = np.sqrt(chi / (chi+ chim*(1-q)))
    return A*np.exp(-chim*q*xi**2 / (2*(chi+ chim*(1-q))))*(H(-A*np.sqrt(q)*xi/np.sqrt(1.-q)) - H(-A*(np.sqrt(q)*xi - np.sqrt(2./chi)*(chi+chim*(1-q)) )/np.sqrt(1.-q)) )

@nb.njit(error_model="numpy",fastmath=True)
def I1(xi, q, chi, chim): #Function checked numerically
    A = np.sqrt(chi / (chi+ chim*(1-q)))
    prefactor = np.sqrt(chi)*np.exp(-chim*q*xi**2 / (2*(chi+ chim*(1-q))))/(chi+ chim*(1-q))**(3./2)
    term_1 = np.sqrt(q)*chi*xi*(H(-A*np.sqrt(q)*xi/np.sqrt(1.-q)) - H(-A*(np.sqrt(q)*xi - np.sqrt(2./chi)*(chi+chim*(1-q)) )/np.sqrt(1.-q)))
    term_2 = -np.sqrt((1-q)*chi*(chi+ chim*(1-q)))*(H1(-A*np.sqrt(q)*xi/np.sqrt(1.-q)) - H1(-A*(np.sqrt(q)*xi - np.sqrt(2./chi)*(chi+chim*(1-q)) )/np.sqrt(1.-q)))
    return prefactor*(term_1 + term_2)

@nb.njit(error_model="numpy",fastmath=True)
def I2(xi, q, chi, chim): #Function checked numerically
    A = np.sqrt(chi / (chi+ chim*(1-q)))
    prefactor = chi**(3./2)*np.exp(-chim*q*xi**2 / (2*(chi+ chim*(1-q))))/(chi+ chim*(1-q))**(5./2)
    B1 = chi*(1-q)+chim*(1.-q)**2+chi*q*xi**2
    term_1 = B1*(H(-A*np.sqrt(q)*xi/np.sqrt(1.-q)) - H(-A*(np.sqrt(q)*xi - np.sqrt(2./chi)*(chi+chim*(1-q)) )/np.sqrt(1.-q)))
    B2 = -2.*xi*np.sqrt(chi*q*(1-q)*(chi+ chim*(1-q)))
    term_2 = B2*(H1(-A*np.sqrt(q)*xi/np.sqrt(1.-q)) - H1(-A*(np.sqrt(q)*xi - np.sqrt(2./chi)*(chi+chim*(1-q)) )/np.sqrt(1.-q)))
    B3 = (1-q)*(chi+ chim*(1-q))
    term_3 = B3*(H2(-A*np.sqrt(q)*xi/np.sqrt(1.-q)) - H2(-A*(np.sqrt(q)*xi - np.sqrt(2./chi)*(chi+chim*(1-q)) )/np.sqrt(1.-q)))
    return prefactor*(term_1 + term_2+ term_3)

@nb.njit(error_model="numpy",fastmath=True)
def n(xi, q, chi, chim):
    return Ineg(xi, q) + np.exp(-chim)*I2chi(xi, q, chi) + I0(xi, q, chi, chim)

@nb.njit(error_model="numpy",fastmath=True)
def a(xi, q, chi, chim):
    return I1(xi, q, chi, chim)

@nb.njit(error_model="numpy",fastmath=True)
def b(xi, q, chi, chim):
    return I2(xi, q, chi, chim)

@nb.njit(error_model="numpy",fastmath=True)
def c(xi, q, chi, chim):
    return np.exp(-chim)*I2chi(xi, q, chi) + I2(xi, q, chi, chim)/(2.*chi)

@jit_4
def integrand_logn(xi, q, chi, chim):
    return Gauss(xi, 0, 1)*np.log(n(xi, q, chi, chim))

@jit_4
def integrand_c_n(xi, q, chi, chim):
    return Gauss(xi, 0, 1)*c(xi, q, chi, chim)/n(xi, q, chi, chim)

@jit_4
def integrand_estar(xi, q, chi, chim):
    return Gauss(xi, 0, 1)*I2chi(xi, q, chi)/n(xi, q, chi, chim)

@jit_4
def integrand_A0(xi, q, chi, chim):
    return Gauss(xi, 0, 1)*(xi*a(xi, q, chi, chim)/np.sqrt(q) - b(xi, q, chi, chim))/n(xi, q, chi, chim)

@jit_4
def integrand_A1(xi, q, chi, chim):
    return Gauss(xi, 0, 1)*b(xi, q, chi, chim)/n(xi, q, chi, chim)
