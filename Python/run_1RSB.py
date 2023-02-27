# pylint: disable=invalid-name, missing-class-docstring, missing-function-docstring, fixme, line-too-long, missing-module-docstring, multiple-imports, pointless-string-statement

import argparse, pickle
from tqdm import tqdm
import numpy as np
from numpy.random import default_rng
from scipy.optimize import root_scalar
from Class.OneRSB_Solver import OneRSB_Solver

"""
1-RSB computation. This file can compute:
- The threshold predicted by the 1-RSB assumption.
- The 1-RSB prediction for the ground state energy over a range of alphas.
"""

if __name__== "__main__":
    tqdm.write("1RSB Solver for the injectivity threshold calculation")

    parser = argparse.ArgumentParser(description='1RSB Solver')
    parser.add_argument("--verbosity", type=int, default = 1)
    parser.add_argument("--damping_q0", type=float, default = 0.0)
    parser.add_argument("--damping_chi", type=float, default = 0.0)
    parser.add_argument("--mlog_TOL", type=float, default = 7.)
    parser.add_argument("--mlog_TOL_chim", type=float, default = 7.)
    parser.add_argument("--alphas", nargs="+", type = float, default = []) #alpha_min, alpha_max, and the NB of points. If empty, we do a threshold calculation using a binary search
    args = parser.parse_args()

    rng = default_rng()

    damping_q0 = args.damping_q0
    damping_chi = args.damping_chi
    verbosity = args.verbosity
    TOL = 10**(-args.mlog_TOL)
    TOL_chim = 10**(-args.mlog_TOL_chim)

    assert np.size(args.alphas) in [0, 1,3]
    if np.size(args.alphas) == 3:
        alphas = np.linspace(args.alphas[0], args.alphas[1], num = int(args.alphas[2]), endpoint=True)
    elif np.size(args.alphas) == 1:
        alphas = np.array([args.alphas[0]])
    else:
        alphas = []

    if np.size(alphas) > 0: #Then we solve for the set of values of alpha

        solutions = [{} for _ in alphas]
        for i_a in tqdm(range(np.size(alphas)), desc = 'Iteration over alphas', leave = True):
            alpha = alphas[i_a]
            parameters = {'verbosity': verbosity, 'alpha':alpha, 'damping_q0':damping_q0, 'damping_chi':damping_chi, 'TOL':TOL, 'TOL_chim':TOL_chim}
            solver = OneRSB_Solver(parameters)
            solutions[i_a] = solver.solve()

        output = {'alphas':alphas, 'solutions':solutions}
        filename = "Data/1RSB_scan_alpha.pkl"
        outfile = open(filename,'wb')
        pickle.dump(output,outfile)
        outfile.close()

    else: #Then we do a binary search over alpha, to find the value of the transition in [6, 7]. 
          #We do two searches using the tqo equivalent estimates of the ground state energy estar and fstar, to check that the results are coherent.
        def to_zero(alpha_, key):
            parameters_ = {'verbosity': verbosity, 'alpha':alpha_, 'damping_q0':damping_q0, 'damping_chi':damping_chi, 'TOL':TOL, 'TOL_chim':TOL_chim}
            solver_ = OneRSB_Solver(parameters_)
            solution_ = solver_.solve()
            return solution_[key] - 1.

        keys = ["estar", "fstar"]
        alphainj_1RSB = {}
        for key in keys:
            sol = root_scalar(to_zero, bracket=[6., 7.], method='brentq', xtol = 1e-6, args = (key,))
            alphainj_1RSB[key] = sol.root

        if verbosity >= 1:
            for key in keys:
                tqdm.write(f"Using the estimate {key}, we have alphainj_1RSB = {alphainj_1RSB[key]} !")

        #To save the parameters, we take the estar solution (this is arbitrary)
        parameters = {'verbosity': verbosity, 'alpha':alphainj_1RSB['estar'], 'damping_q0':damping_q0, 'damping_chi':damping_chi, 'TOL':TOL, 'TOL_chim':TOL_chim}
        solver = OneRSB_Solver(parameters)
        solution = solver.solve()

        output = {'alphainj_1RSB':alphainj_1RSB, 'solution':solution}
        filename = "Data/1RSB_threshold.pkl"
        outfile = open(filename,'wb')
        pickle.dump(output,outfile)
        outfile.close()
