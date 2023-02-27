#output pylint: disable=invalid-name, missing-class-docstring, missing-function-docstring, fixme, line-too-long, missing-module-docstring, multiple-imports, pointless-string-statement

import pickle
import numpy as np
import jax.numpy as jnp
from jax import vmap
from scipy.optimize import root_scalar
from tqdm import tqdm
from .jitted_functions.FullRSB_GPU import gaussian_conv_sw_jax, vmap_gamma_jax, vmap_finf

"""
Solver of the Full-RSB zero-temperature equations.
In this file we often call estar = estar(beta = infty), which is called fstar in the paper.
"""

class FullRSBSolver_GPU():
    #For a given value of alpha and a large enough value of the RSB level k, we find the F-RSB solution.
    def __init__(self, parameters_):
        self.parameters = parameters_
        self.k = parameters_["k"]
        assert self.k >= 1
        self.xs = parameters_["xs"] #Sorted array, length self.k+1, with x_k = inf
        assert self.xs[0] > 0 and self.xs[-1] == jnp.inf
        self.xmax = self.xs[-2] #Since xs[-1] = jnp.inf
        self.verbosity = parameters_['verbosity']
        self.alpha = parameters_["alpha"]
        self.epsilon_cv = parameters_["epsilon_cv"]
        self.save_b = parameters_["save"] #Level of saves of the iterations
        self.MAX_ITERATIONS = parameters_["MAX_ITERATIONS"]
        self.MIN_ITERATIONS = parameters_["MIN_ITERATIONS"]
        self.dtype = 'float64'

        self.full_output = parameters_["full_output"] #Do I return the full output (with the functions f and P)

        if "q_init" in parameters_:
            self.q_init = parameters_["q_init"]
        else:
            self.q_init = None

        #Parameters for the discretization of the space in the parameter H
        self.H = parameters_["H"]
        self.N = parameters_["N"]
        self.hs, self.dh = jnp.linspace(-self.H, self.H, 2*self.N+1, endpoint=True, retstep=True, dtype = self.dtype)

        self.chi = 6e-2 #Approximate value from the 1RSB computation (for alpha around 6): we use it to initialize the iterations

        if self.q_init is None:
            #Arbitrary initialization, using again that q0 is around 0.7 for alpha around 6.
            self.qs = jnp.linspace(0.7, 1, self.k+1 , endpoint=True, dtype = self.dtype) #We must have q_k = 1 always
        else:
            self.qs = jnp.array(self.q_init.astype(self.dtype))
            assert self.qs[-1] == 1 and jnp.size(self.qs) == self.k+1

        self.index_first_q = int(-1e10) #Index of the first strictly positive q, we will compute it when updating P
        self.qm1s = jnp.zeros_like(self.qs)
        self.lambdas = np.zeros_like(self.qs) #Given the update of lambdas, they are not well adapted for use as jax arrays
        self.fis = jnp.zeros((self.k + 1, 2*self.N+1), dtype = self.dtype)
        self.Pis = jnp.zeros((self.k + 1, 2*self.N+1), dtype = self.dtype)
        self.deltaqs = jnp.zeros(self.k+1, dtype = self.dtype)
        self.compute_deltaq()
        self.set_f_boundary()

    #This function is jitted to allow for in-place change
    def set_f_boundary(self):
        self.fis = self.fis.at[-1, :].set(vmap_finf(self.hs))

    def compute_fi(self):
        for i in range(self.k):
            if self.xs[-2-i] == 0: #In this case the procedure could be singular
                self.fis = self.fis.at[-2-i, :].set(gaussian_conv_sw_jax(self.fis[-1-i], self.deltaqs[-1-i]/(2*self.chi), self.H))
            else:
                fmax = jnp.amax(self.fis[-1-i, :]) #To avoid numerical overflows in the convolution
                conv = gaussian_conv_sw_jax(jnp.exp(self.xs[-2-i]*(self.fis[-1-i] - fmax)), self.deltaqs[-1-i]/(2*self.chi), self.H)
                self.fis = self.fis.at[-2-i, :].set(fmax + jnp.log(conv)/self.xs[-2-i])

    def compute_Pi(self):
        self.index_first_q = 0 #Index of the first strictly positive q
        while self.qs[self.index_first_q] <= 1e-5:
            self.index_first_q += 1

        #Before this index, all the Pis are delta distributions
        self.Pis = self.Pis.at[self.index_first_q, :].set(vmap_gamma_jax(self.qs[self.index_first_q], jnp.sqrt(2*self.chi)*self.hs))

        #Now we do the procedure
        for i in range(self.index_first_q, self.k):
            fimin = jnp.amin(self.fis[i, :]) #To avoid numerical overflows in the convolution
            conv  = gaussian_conv_sw_jax(self.Pis[i, :]*jnp.exp(-self.xs[i]*(self.fis[i, :] - fimin)), self.deltaqs[i+1]/(2*self.chi), self.H)
            self.Pis = self.Pis.at[i+1, :].set(jnp.exp(self.xs[i]*(self.fis[i+1, :] - fimin)) * conv)

    def update_chi(self):
        h_first = self.N #h = 0
        h_last = jnp.abs(self.hs - 1.).argmin() #h = 1

        RHS = 2**(3./2) * self.alpha * jnp.trapz((self.Pis[-1]*self.hs**2)[h_first:h_last], self.hs[h_first:h_last]) #We use trapz rather than simpson

        #We compute the partial sums sum_{j=i}^k (q_j - q_{j-1}) x_{j-1}, for i = 0, ..., k with the convention q_{-1} = 0 and x_{-1} = 0
        #This sum is clearly not adapted for the use of JAX, since it is recursive, so we use a simple numpy array
        partial_sums = np.zeros(self.k+1, dtype = self.dtype)
        partial_sums[-1] = self.deltaqs[-1]*self.xs[-2]
        for i in range(1, self.k):
            partial_sums[-1-i] = partial_sums[-i] + self.deltaqs[-1-i]*self.xs[-2-i]
        partial_sums[0] = partial_sums[1] #For i = 0 and i = 1 the sum is the same because x_{-1} = 0

        rolled_partial_sums = np.roll(partial_sums, shift = -1) #rolled[i] = partial_sums[i+1]

        #The function to zero out: goes to +infty for chi -> 0, and to a negative value for chi -> infty, so it should work
        def to_zero(chi):
            #The two boundary conditions
            sum_value = self.qs[0] / ((chi + partial_sums[1])**2) + self.deltaqs[-1] / (chi*(chi + partial_sums[-1]))
            sum_value += jnp.sum(self.deltaqs[1:-1]  / ((chi + partial_sums[1:-1]) *(chi + rolled_partial_sums[1:-1]) ) )
            sum_value *= np.sqrt(chi)
            return sum_value - RHS

        sol = root_scalar(to_zero, bracket=[1e-6, 1e0] ,method='brentq')
        self.chi = sol.root

    def compute_qm1(self):
        dfis = jnp.gradient(self.fis, self.dh, axis = 1) #First derivative of f with respect to h

        def fill_qm1(i):
            #We take care of the potential problematic case of P(x_m,h) = delta(h) if ever q_0 = 0
            return jnp.where(i < self.index_first_q, dfis[i, self.N]**2, jnp.trapz(self.Pis[i, :]*(dfis[i, :]**2), self.hs)) #If i < index_first_f, then q_0 = ... = q_i = 0.Recall that f[0,N] is basically f(q0, h = 0)

        def vmap_fill_qm1(indices):
            return vmap(fill_qm1)(indices)

        self.qm1s = (- self.alpha / jnp.sqrt(2*self.chi)) * vmap_fill_qm1(jnp.arange(self.k+1)) #we vectorize this loop

    def update_lambda(self):
        #Recursive update, this is why lambdas are not well adapted to jax
        self.lambdas = np.zeros(self.k+1, dtype = self.dtype)
        self.lambdas[0] = np.sqrt(-self.qs[0]/self.qm1s[0])
        for i in range(1, self.k+1):
            self.lambdas[i] = 1./(1./self.lambdas[i-1] - self.xs[i-1]*(self.qm1s[i] - self.qm1s[i-1]))

    def update_q(self):
        new_qs = np.ones(self.k+1) #qk = 1 always, since xk = inf
        #The term i = 0 of this partial sum has no meaning
        partial_sums = np.zeros(self.k+1, dtype = self.dtype)
        partial_sums[-1] = - self.lambdas[-1]/self.xs[-2]
        for i in range(self.k-1):
            partial_sums[-2-i] = partial_sums[-1-i] + (1./self.xs[-2-i] - 1./self.xs[-3-i])*self.lambdas[-2-i]
        partial_sums = np.roll(partial_sums, shift = -1) #New[i] = old[i+1]

        new_qs[:self.k] = 1. - self.lambdas[:self.k] / self.xs[:self.k] - partial_sums[:self.k]

        error = jnp.max(jnp.abs(new_qs - self.qs))
        self.qs = jnp.array(new_qs)

        self.compute_deltaq()

        #Heuristically, we find that the following error happens when H is not large enough.
        assert jnp.amin(self.deltaqs) >= -1e-10, f"ERROR: The new qs are no longer in increasing order, the min/max difference are {jnp.amin(self.deltaqs[1:])} and {jnp.amax(self.deltaqs[1:])}"
        if jnp.amin(self.deltaqs) < 0: #Then in this case, the difference is extremely small, so we might as well sort the qs
            self.qs = jnp.sort(self.qs)
            self.compute_deltaq()

        return error

    def compute_deltaq(self):
        #Computes q[i] - q[i-1], with q[-1] = 0
        self.deltaqs = self.qs - jnp.roll(self.qs, shift = 1) #Rolled[i] = original[i-1]
        self.deltaqs = self.deltaqs.at[0].set(self.qs[0])

    def run(self):
        #We make the iterations until convergence
        converged = False
        counter, error = 0, np.inf

        with tqdm(desc=f'Iterative procedure (current error = {error})', leave = False, total = 100) as pbar:
            while (not(converged) and counter < self.MAX_ITERATIONS) or (counter < self.MIN_ITERATIONS):
                self.compute_fi()
                self.compute_Pi()

                self.compute_qm1()
                self.update_lambda()
                error = self.update_q()
                self.update_chi()

                if self.save_b >= 1: #Do we save the results after this iteration
                    self.save(counter, error)

                pbar.set_description(f"Iterative procedure (current error = {error})")
                pbar.refresh() #Show immediately the update on the progress bar

                converged = (error < self.epsilon_cv)
                counter += 1
                if counter % 100 == 0:
                    pbar.reset(total = 100)
                pbar.update(1)

        assert converged, "ERROR: the iterations did not converge !"
        estar = self.get_estar()
        if self.verbosity >= 2:
            tqdm.write(f"Ground state energy = {round(estar, 6)}.")

        output = {"chi":self.chi, "xs":self.xs, "qs":self.qs, "k":self.k, "estar": estar}
        if self.full_output:
            output['Pis']=self.Pis
            output['fis']=self.fis
            output['hs']=self.hs
            output['N']=self.N
            output['H']=self.H
        return output

    def get_estar(self):
        #We compute the integral of P(xmax = inf, h) starting from 1
        h_first = jnp.abs(self.hs - 1.).argmin() #h = 1, since it's not necessarily exactly in the interval
        return self.alpha*jnp.sqrt(2*self.chi)*jnp.trapz(self.Pis[-1][h_first:], self.hs[h_first:])

    def save(self, iteration, error):
        #Saves all the parameters and estimators at a given iteration
        estar = self.get_estar()
        output = {"chi":self.chi, "xs":self.xs, "qs":self.qs, "k":self.k, "estar":estar, "error": error}
        if self.save_b >= 2:
            output['Pis']=self.Pis
            output['fis']=self.fis
            output['hs']=self.hs
            output['N']=self.N
            output['H']=self.H
        filename = f"Data/tmp/FRSB_T0_GPU_alpha_{self.alpha}_k_{self.k}_xmax_{self.xmax}_H_{self.H}_N_{self.N}_iteration_{iteration}.pkl"
        outfile = open(filename,'wb')
        pickle.dump(output,outfile)
        outfile.close()
