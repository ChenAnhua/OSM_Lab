#======================================================================
#
#     This routine solves an infinite horizon growth model 
#     with dynamic programming and sparse grids
#
#     The model is described in Scheidegger & Bilionis (2017)
#     https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2927400
#
#     external libraries needed:
#     - IPOPT (https://projects.coin-or.org/Ipopt)
#     - PYIPOPT (https://github.com/xuy/pyipopt)
#     - TASMANIAN (http://tasmanian.ornl.gov/)
#
#     Simon Scheidegger, 11/16 ; 07/17
#======================================================================

import nonlinear_solver_initial as solver     #solves opt. problems for terminal VF
import nonlinear_solver_iterate as solviter   #solves opt. problems during VFI
from parameters import *                      #parameters of model
import interpolation as interpol              #interface to sparse grid library/terminal VF
import interpolation_iter as interpol_iter    #interface to sparse grid library/iteration
import interpolation_exp as interpol_exp              #interface to sparse grid library/terminal VF
import interpolation_iter_exp as interpol_iter_exp    #interface to sparse grid library/iteration
import postprocessing as post                 #computes the L2 and Linfinity error of the model

import TasmanianSG                            #sparse grid library
import numpy as np


#======================================================================
# Start with Value Function Iteration

# terminal/initial value function
valnew_sub = TasmanianSG.TasmanianSparseGrid()
for iT, theta in enumerate(theta_vec):
    if (numstart==0):
        
        valnew_sub = interpol.sparse_grid(n_agents, iDepth , refinement_level, fTol, theta)
        valnew_sub.write("valnew_1." + str(numstart) + ".shock" + str(iT + 1) + ".txt") #write file to disk for restart
    
    
    
    #valnew = interpol.sparse_grid(n_agents, iDepth , refinement_level, fTol, theta)
    #valnew.write("valnew_1." + str(numstart) + ".txt") #write file to disk for restart

# value function during iteration
    else:
        valnew_sub.read("valnew_1." + str(numstart) + ".shock" + str(iT + 1) + ".txt")  #write file to disk for restart

valnew=TasmanianSG.TasmanianSparseGrid()                      # we might need to calculate 5 different Valnew now
valnew = interpol_exp.sparse_grid(n_agents, iDepth , refinement_level, fTol, theta_vec, theta_prob)    
valnew.write("valnew_l." + str(numstart) + ".txt")
valold=TasmanianSG.TasmanianSparseGrid()
valold=valnew                 # we need to define Valold differently

for i in range(numstart, numits):
    for iT ,theta in enumerate(theta_vec):
        valnew_sub=TasmanianSG.TasmanianSparseGrid()
        valnew_sub = interpol_iter.sparse_grid_iter(n_agents, iDepth , valold, refinement_level, fTol, theta)
        valnew_sub.write("valnew_1." + str(i + 1) + ".shock" + str(iT + 1) + ".txt")
    valnew = TasmanianSG.TasmanianSparseGrid()
    valnew = interpol_iter_exp.sparse_grid_iter(n_agents, iDepth , valold, refinement_level, fTol, theta_vec, theta_prob)
    valnew.write("valnew_l." + str(i + 1) + ".txt")
    valold=TasmanianSG.TasmanianSparseGrid()                  # we need to define Valold differently
    valold=valnew
    
#======================================================================
print "==============================================================="
print " "
print " Computation of a growth model of dimension ", n_agents ," finished after ", numits, " steps"
print " "
print "==============================================================="
#======================================================================

# compute errors   
avg_err=post.ls_error(n_agents, numstart, numits, No_samples)

#======================================================================
print "==============================================================="
print " "
print " Errors are computed -- see errors.txt"
print " "
print "==============================================================="
#======================================================================
