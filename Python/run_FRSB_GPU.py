# pylint: disable=invalid-name, missing-class-docstring, missing-function-docstring, fixme, line-too-long, missing-module-docstring, multiple-imports, pointless-string-statement

import argparse

"""
Full-RSB computation. This file can compute:
- The threshold predicted by the Full-RSB assumption.
- The F-RSB prediction for the ground state energy over a range of alphas.
"""

#We must parse parameters before defining other functions or importing any jax functions, since it will fix which GPU is used.
parser = argparse.ArgumentParser(description='FRSB Solver (GPU)')
parser.add_argument("--verbosity", type=int)
parser.add_argument("--save", type=int, default = 0)
parser.add_argument("--full_output", type=int, default = 0)
parser.add_argument("--MAX_ITERATIONS", type=int, default = 10000)
parser.add_argument("--MIN_ITERATIONS", type=int, default = 10)
parser.add_argument("--mlog_ecv", type=float, default = 4.) #The convergence criterion on qs is Deltaq <= 10**(- mlog_ecv)
parser.add_argument("--alphas", nargs="+", type = float, default = []) #alpha_min, alpha_max, and the NB of points. If empty, we do a threshold calculation.
parser.add_argument("--Hs", nargs="+", type = float) #The dfifferent values of H considered
parser.add_argument("--xmaxs", nargs="+", type = float) #The dfifferent values of xmax considered
parser.add_argument("--ks", nargs="+", type=int) #The different values of k considered
parser.add_argument("--cs", nargs="+", type=float) #The constant N = c N0.
parser.add_argument("--device", type=int, default = 0) #The number of the GPU to use
args = parser.parse_args()

from jax import config, devices
device = devices("gpu")[args.device] #The GPU device to use
config.update('jax_enable_x64', True) #Enable 64 bit precision computation in Jax
config.update('jax_default_device', device)

import pickle
from tqdm import tqdm
import numpy as np
import jax.numpy as jnp
from scipy.optimize import brentq
from Class.FullRSBSolver_GPU import FullRSBSolver_GPU

def N0(H, k, xmax):
    return H * np.sqrt(k) * xmax

def run_scan_alpha(alphas_, parameters_, prefix_):
    qs, chis, estars = np.array([None for _ in alphas_]), np.zeros_like(alphas_), np.zeros_like(alphas_)
    c, xmax, H, N, k, xs = parameters_['c'], parameters_['xmax'], parameters_['H'], parameters_['N'], parameters_['k'], parameters_['xs']

    for i_a in tqdm(range(jnp.size(alphas_)), desc = f"(k, xmax, H, c) = ({k}, {xmax}, {H}, {c}). Iteration over alphas", leave = True):
        alpha = alphas_[i_a]

        parameters_['alpha'] = alpha

        if i_a >= 1: #Then we add the previous result as initialization
            parameters_["q_init"] = qs[i_a-1]

        Solver = FullRSBSolver_GPU(parameters_)
        solution = Solver.run()
        qs[i_a] = solution['qs']
        chis[i_a] = solution['chi']
        estars[i_a] = solution['estar']


        output = {'solution': solution, 'alpha':alpha, 'H':H, 'N':N, 'k':k, 'xs':xs}
        filename = f"Data/tmp/FRSB_T0_GPU_alpha_{alpha}_k_{k}_xmax_{xmax}_H_{H}_N_{N}.pkl"
        outfile = open(filename,'wb')
        pickle.dump(output,outfile)
        outfile.close()

    #Now we save everything
    output = {'alphas':alphas_, 'qs':qs, 'chis':chis, 'estars':estars, 'xs':xs, 'H':H, 'N':N, 'k':k, 'c':c}
    filename = prefix_+f"FRSB_T0_GPU_k_{k}_xmax_{xmax}_H_{H}_c_{c}.pkl"
    outfile = open(filename,'wb')
    pickle.dump(output,outfile)
    outfile.close()

def run_threshold(parameters_):
    #We first solve the initialization alpha = 6 to have initializer values
    c, xmax, H, k = parameters_['c'], parameters_['xmax'], parameters_['H'], parameters_['k']

    tqdm.write(f"We have (k, xmax, H, c) = ({k}, {xmax}, {H}, {c}). Starting alpha = 6 for initialization...")
    parameters_["alpha"] = 6.
    Solver_init = FullRSBSolver_GPU(parameters_)
    solution_init = Solver_init.run()
    initialization = solution_init["qs"]

    def to_zero(alpha_):
        tqdm.write(f"Starting alpha = {alpha_}")
        parameters_["alpha"] = alpha_
        parameters_["q_init"] = initialization
        Solver_ = FullRSBSolver_GPU(parameters_)
        solution_ = Solver_.run()
        return solution_['estar'] - 1.

    xtol = 1e-4
    alphainj_FRSB = brentq(to_zero, 6., 7., xtol = xtol)

    if verbosity >= 1:
        tqdm.write(f"Found alphainj_FRSB = {alphainj_FRSB} +- {xtol} !")

    parameters_["alpha"] = alphainj_FRSB
    parameters_["q_init"] = initialization
    solver_final = FullRSBSolver_GPU(parameters_)
    solution_final = solver_final.run()

    output = {'alphainj_FRSB':alphainj_FRSB, 'xtol':xtol, 'solution':solution_final, 'parameters':parameters_}
    filename = f"Data/thresholds/FRSB_T0_GPU_k_{k}_xmax_{xmax}_H_{H}_c_{c}.pkl"
    outfile = open(filename,'wb')
    pickle.dump(output,outfile)
    outfile.close()

if __name__== "__main__":

    verbosity = args.verbosity
    save = args.save
    full_output = bool(args.full_output)
    MAX_ITERATIONS = args.MAX_ITERATIONS
    MIN_ITERATIONS = args.MIN_ITERATIONS
    epsilon_cv = 10**(-args.mlog_ecv) #Threshold for the convergence
    dtype = 'float64'

    #Now we iterate over k, H, c, xmax
    for i_k in tqdm(range(np.size(args.ks)), desc = f"Iteration over ks = {args.ks}", leave = True):
        k_ = args.ks[i_k]
        for i_H in tqdm(range(np.size(args.Hs)), desc = f"Iteration over Hs = {args.Hs}", leave = True):
            H_ = args.Hs[i_H]
            for i_c in tqdm(range(np.size(args.cs)), desc = f"Iteration over cs = {args.cs}", leave = True):
                c_ = args.cs[i_c]
                for i_x in tqdm(range(np.size(args.xmaxs)), desc = f"Iteration over xmaxs = {args.xmaxs}", leave = True):
                    xmax_ = args.xmaxs[i_x]

                    N_ = int(c_*np.ceil(N0(H_, k_, xmax_))) #We take N = c N_0
                    xs_ = jnp.zeros(k_+1, dtype = dtype)
                    xs_ = xs_.at[:-1].set(np.linspace(1e-2, xmax_, k_, endpoint=True, dtype = dtype)) #List of xs
                    xs_ = xs_.at[-1].set(jnp.inf)

                    parameters = {'verbosity': verbosity, 'c':c_, 'k':k_, 'epsilon_cv':epsilon_cv, 'xmax':xmax_, 'H':H_, 'N':N_, 'xs':xs_, 'save':save, 'MAX_ITERATIONS':MAX_ITERATIONS,  'MIN_ITERATIONS':MIN_ITERATIONS, "full_output":full_output}

                    assert np.size(args.alphas) in [0, 1,3]
                    if np.size(args.alphas) == 3:
                        prefix = "Data/scan_alpha/" if not(full_output) else "Data/investigation_f_P/"
                        alphas = jnp.linspace(args.alphas[0], args.alphas[1], num = int(args.alphas[2]), endpoint=True, dtype = dtype)
                        run_scan_alpha(alphas, parameters, prefix) #Then we solve for the set of values of alpha
                    elif np.size(args.alphas) == 1:
                        prefix = "Data/scan_alpha/" if not(full_output) else "Data/investigation_f_P/"
                        alphas = jnp.array([args.alphas[0]], dtype = dtype)
                        run_scan_alpha(alphas, parameters, prefix) #Then we solve for the set of values of alpha
                    else: #Then we do a search over alpha to find the injectivity threshold
                        run_threshold(parameters)
