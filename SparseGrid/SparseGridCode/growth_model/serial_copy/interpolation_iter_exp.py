#======================================================================
#
#     This routine interfaces with the TASMANIAN Sparse grid
#     The crucial part is 
#
#     aVals[iI]=solveriter.iterate(aPoints[iI], n_agents)[0]  
#     => at every gridpoint, we solve an optimization problem
#
#     Simon Scheidegger, 11/16 ; 07/17
#======================================================================

import TasmanianSG
import numpy as np
from parameters import *
import nonlinear_solver_iterate as solveriter

#======================================================================

def sparse_grid_iter(n_agents, iDepth, valold, refinement_level, fTol, theta_vec, theta_prob):   
    
    grid  = TasmanianSG.TasmanianSparseGrid()

    k_range=np.array([k_bar, k_up])

    ranges=np.empty((n_agents, 2))


    for i in range(n_agents):
        ranges[i]=k_range

    iDim=n_agents
    iOut=1
    #level of grid before refinement
    grid.makeLocalPolynomialGrid(iDim, iOut, iDepth, which_basis, "localp")
    grid.setDomainTransform(ranges)

    aPoints=grid.getPoints()
    iNumP1=aPoints.shape[0]
    aVals=np.empty([iNumP1, 1])
    
    file=open("comparison1.txt", 'w')
    for iI in range(iNumP1):
        expectation = [solveriter.iterate(aPoints[iI], n_agents, valold, theta)[0] for theta in theta_vec]
        expectation = np.array(expectation)
        expectation = np.dot(expectation, theta_prob.T)
        #expectation = np.empty_like(aVals[iI])
        #for iT, theta  in enumerate(theta_vec):
        #    new_expect = solveriter.iterate(aPoints[iI], n_agents, valold, theta)[0]
        #    expectation += new_expect*theta_prob[iT]
        aVals[iI]=expectation         # We need to do something on changing this Valold
        v=aVals[iI]*np.ones((1,1))
        to_print=np.hstack((aPoints[iI].reshape(1,n_agents), v))
        np.savetxt(file, to_print, fmt='%2.16f')
    
    
    #file.close()
    grid.loadNeededPoints(aVals)
    #refinement level
    for iK in range(refinement_level):
        grid.setSurplusRefinement(fTol, 1, "fds")   #also use fds, or other rules
        aPoints = grid.getNeededPoints()
        aVals = np.empty([aPoints.shape[0], 1])
        for iI in range(aPoints.shape[0]):
            expectation = [solveriter.iterate(aPoints[iI], n_agents, valold, theta)[0] for theta in theta_vec]
            expectation = np.array(expectation)
            expectation = np.dot(expectation, theta_prob.T)
            aVals[iI]= expectation     # We need to do something on changing this Valold
            v = aVals[iI]*np.ones((1, 1))
            to_print=np.hstack(( aPoints[iI].reshape(1,n_agents), v))
            np.savetxt(file, to_print, fmt='%2.16f')

        grid.loadNeededPoints(aVals)    
    
    file.close()
    f=open("grid_iter.txt", 'w')
    np.savetxt(f, aPoints, fmt='% 2.16f')
    f.close()
    
    return grid

#======================================================================
