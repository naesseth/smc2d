"""smc.py

2D constrained channel using fully adapted SMC.

"""
import argparse
import numpy as np
from scipy import misc
from numpy.random import random_sample
import time

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(usage=__doc__)
    parser.add_argument("--nx", type=int, help="number of rows")
    parser.add_argument("--ny", type=int, help="number of columns")
    parser.add_argument("--iter", type=int, help="iterations to run")
    parser.add_argument("--particles", type=str, help="list with particles N")
    args = parser.parse_args()

    # Number of particles - N (design variable)
    particles = args.particles.split(',')
    iterations = args.iter
    CM = np.zeros( iterations )
    
    # Initialize arrays/matrices
    nx = args.nx
    ny = args.ny
    xDomain = np.arange( 2 )
    interactionPotentials = np.ones( (2, 2) )
    interactionPotentials[1, 1] = 0

    fileName = str(nx)+'x'+str(ny)+'informationRateTEST.txt'

    # New file, print initial info, first line
    f = open(fileName, 'w')
    f.write('nx ny\n')
    f.write(str(nx) + ' ' + str(ny) + '\n')
    f.write('particles informationRateEst \n')
    f.close()
    
    for iIter in particles:
        print iIter
        for i in np.arange(0,iterations):
            N = int(iIter)
            print N
            # SMC initializations
            trajectorySMC = np.zeros( (N, nx, ny) )
            tempWeights = np.ones( N )
            ancestors = np.zeros( N, np.int )
            tempParticleDist = np.zeros( len(xDomain) )
    
            # BP initializations
            messages = np.zeros( (N, nx, len(xDomain)) )

            # ---------------
            #      SMC
            # ---------------        
            # SMC first iteration, forward filtering
            for iForward in np.arange(0,nx-1):
                for iParticle in np.arange(0,N):
                    if iForward > 0 and iForward < nx-1:
                        messages[iParticle,iForward,:] = np.dot(interactionPotentials, messages[iParticle,iForward-1,:])
                    else:
                        messages[iParticle,iForward,:] = np.dot( interactionPotentials, np.ones(2) )
            tempWeights = np.sum( messages[:,nx-2,:], axis=1 )
            CM[i] += np.log2(np.mean( tempWeights ))
    
            for iBackward in np.arange(0,nx)[::-1]:
                for iParticle in np.arange(0,N):
                    tempParticleDist = np.ones( len(xDomain) )
                    if iBackward < nx-1:
                        tempParticleDist = tempParticleDist * interactionPotentials[trajectorySMC[iParticle, iBackward+1, 0],:]
                    if iBackward > 0:
                        tempParticleDist = tempParticleDist * messages[iParticle, iBackward-1,:]
                    tempParticleDist = tempParticleDist / np.sum( tempParticleDist )
                    ind = discreteSampling( tempParticleDist, xDomain, 1)
                    trajectorySMC[iParticle, iBackward, 0] = xDomain[ind]

            # SMC MAIN LOOP
            for iSMC in np.arange(1,ny):
                # Forward filtering
                for iForward in np.arange(0,nx):
                    for iParticle in np.arange(0,N):
                        if iForward > 0 and iForward < nx-1:
                            messages[iParticle,iForward,:] = np.dot(interactionPotentials, interactionPotentials[trajectorySMC[iParticle, iForward, iSMC-1],:] * messages[iParticle,iForward-1,:])
                        elif iForward == 0:
                            messages[iParticle,iForward,:] = np.dot(interactionPotentials, interactionPotentials[trajectorySMC[iParticle, iForward, iSMC-1],:])
                        else:
                            messages[iParticle,iForward,:] = interactionPotentials[trajectorySMC[iParticle, iForward, iSMC-1],:] * messages[iParticle,iForward-1,:]

                tempWeights = np.sum( messages[:,nx-1,:], axis=1)
                CM[i] += np.log2( np.mean( tempWeights ) )
        
                if np.sum( tempWeights ) == 0:
                    raw_input()
                # Sample ancestors
                ancestors = discreteSampling( tempWeights / np.sum( tempWeights ), np.arange(N), N)
                     
                # Backward sampling
                for iBackward in np.arange(0,nx)[::-1]:
                    for iParticle in np.arange(0,N):
                        tempParticleDist = interactionPotentials[trajectorySMC[ancestors[iParticle], iBackward, iSMC-1],:]
                        if iBackward < nx-1:
                            tempParticleDist = tempParticleDist * interactionPotentials[trajectorySMC[iParticle, iBackward+1, iSMC],:]
                        if iBackward > 0:
                            tempParticleDist = tempParticleDist * messages[ancestors[iParticle],iBackward-1,:]
                        tempParticleDist = tempParticleDist / np.sum(tempParticleDist)
                        ind = discreteSampling( tempParticleDist, xDomain, 1 )
                        trajectorySMC[iParticle, iBackward, iSMC] = xDomain[ind]
                    
                trajectorySMC[:, :, :iSMC] = trajectorySMC[ancestors.astype(int), :, :iSMC].reshape( (N, nx, iSMC) )

            CM[i] = CM[i] / (nx*ny)
        f = open(fileName, 'a')
        f.write(str(iIter) + ' ')
        np.savetxt(f, CM.reshape( (1,iterations) ))
        f.close()
        #print 'MI est:',CM

def discreteSampling(weights, domain, nrSamples):
    bins = np.cumsum(weights)
    return domain[np.digitize(random_sample(nrSamples), bins)]

def ravel_multi_index(coord, shape):
    return coord[0] * shape[1] + coord[1]

def unravel_index(coord, shape):
    iy = np.remainder(coord, shape[1])
    ix = (coord - iy) / shape[1]
    return ix, iy
    
if __name__ == "__main__":
    main()

