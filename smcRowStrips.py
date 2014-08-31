"""smc.py

2D constrained channel using fully adapted SMC adding several columns at a time.

"""
from optparse import OptionParser
import numpy as np
from scipy import misc
from numpy.random import random_sample
import time

def main():
    # Parse command-line arguments
    parser = OptionParser()
    parser.add_option("--nx", type=int, help="number of rows")
    parser.add_option("--ny", type=int, help="number of columns")
    parser.add_option("--iter", type=int, help="iterations to run")
    parser.add_option("--particles", type=str, help="list with particles N")
    parser.add_option("--stripWidth", type=int, help="width of strip")
    (args, options) = parser.parse_args()

    # Number of particles - N (design variable)
    particles = args.particles.split(',')
    iterations = args.iter
    CM = np.zeros( iterations )
    
    # Initialize arrays/matrices
    nx = args.nx
    ny = args.ny
    stripWidth = args.stripWidth
    nrInd = int(ny/stripWidth)
    # Interaction potentials
    interactionPotentials = np.ones( (2, 2) )
    interactionPotentials[1, 1] = 0
    intPot = np.ones( (2, 2) )
    intPot[1, 1] = 0
    nrComb = 2**stripWidth
    xDomain = np.zeros((nrComb,stripWidth), dtype=np.uint64)
    for iIter in range(nrComb):
        tempStr = np.binary_repr(iIter, width=stripWidth)
        for iStrip in range(stripWidth):
            xDomain[iIter, iStrip] = int(tempStr[iStrip])
    fileName = str(nx)+'x'+str(ny)+'informationRateROWstripW' + str(stripWidth) + '.txt'

    # New file, print initial info, first line
    f = open(fileName, 'w')
    f.write('nx ny\n')
    f.write(str(nx) + ' ' + str(ny) + '\n')
    f.write('particles timeElapsed informationRateEst \n')
    f.close()
    
    for iIter in particles:
        print iIter
        startSMC = time.time()
        CM = np.zeros( iterations )
        for i in np.arange(0,iterations):
            N = int(iIter)
            # SMC initializations
            trajectorySMC = np.zeros( (N, nx, ny) )
            tempWeights = np.ones( N )
            ancestors = np.zeros( N, np.int )
    
            # BP initializations
            messages = np.zeros( (N, nrComb, nx-1) )
            normConstMessages = np.zeros( (N, nx) )

            # ---------------
            #      SMC
            # ---------------        
            # CSMC first iteration, forward filtering
            for iParticle in range(N):
                for iRow in range(nx-1):
                    for iCur in range(nrComb):
                        tempDist = np.ones(nrComb)
                        for iPrev in range(nrComb):
                            for iStrip in range(stripWidth):
                                if iStrip != stripWidth-1:
                                    tempDist[iPrev] *= intPot[xDomain[iPrev,iStrip], xDomain[iPrev, iStrip+1]] 
                                tempDist[iPrev] *= intPot[xDomain[iPrev,iStrip], xDomain[iCur, iStrip]]
                            if iRow > 0:
                                tempDist[iPrev] *= messages[iParticle,iPrev, iRow-1]
                        messages[iParticle,iCur,iRow] = np.sum(tempDist)
                    normConstMessages[iParticle,iRow] = np.sum(messages[iParticle,:,iRow])
                    messages[iParticle,:,iRow] = messages[iParticle,:,iRow] / normConstMessages[iParticle,iRow]
                # Column sum
                tempDist = np.ones(nrComb)
                for iCur in range(nrComb):
                    for iStrip in range(stripWidth-1):
                        tempDist[iCur] *= intPot[xDomain[iCur,iStrip], xDomain[iCur, iStrip+1]] 
                    tempDist[iCur] *= messages[iParticle, iCur, nx-2]
                normConstMessages[iParticle,-1] = np.sum( tempDist )
            tempWeights = np.prod(normConstMessages, axis=1)
            CM[i] += np.log2(np.mean( tempWeights ))

            # First iteration, Backward sampling
            for iParticle in range(N):
                for iRow in range(nx)[::-1]:
                    tempDist = np.ones( nrComb )
                    for iCur in range(nrComb):
                        for iStrip in range(stripWidth):
                            if iStrip != stripWidth-1:
                                tempDist[iCur] *= intPot[xDomain[iCur,iStrip], xDomain[iCur, iStrip+1]] 
                            if iRow < nx-1:
                                tempDist[iCur] *= intPot[xDomain[iCur,iStrip], trajectorySMC[iParticle,iRow+1, iStrip]]
                        if iRow > 0:
                            tempDist[iCur] *= messages[iParticle, iCur, iRow-1]
                    tempDist = tempDist / np.sum(tempDist)
                    curInd = discreteSampling( tempDist, range(nrComb), 1 )
                    for iStrip in range(stripWidth):
                        trajectorySMC[iParticle,iRow,iStrip] = xDomain[curInd,iStrip]
          
            # SMC MAIN LOOP
            for iSMC in np.arange(1,nrInd):
                # BP initializations
                messages = np.zeros( (N, nrComb, nx-1) )
                normConstMessages = np.zeros( (N, nx) )
                # Forward filtering
                for iParticle in range(N):
                    for iRow in range(nx-1):
                        for iCur in range(nrComb):
                            tempDist = np.ones(nrComb)
                            for iPrev in range(nrComb):
                                for iStrip in range(stripWidth):
                                    if iStrip != stripWidth-1:
                                        tempDist[iPrev] *= intPot[xDomain[iPrev,iStrip], xDomain[iPrev, iStrip+1]] 
                                    tempDist[iPrev] *= intPot[xDomain[iPrev,iStrip], xDomain[iCur, iStrip]]
                                tempDist[iPrev] *= intPot[xDomain[iPrev,0], trajectorySMC[iParticle, iRow, iSMC*stripWidth-1]]
                                if iRow > 0:
                                    tempDist[iPrev] *= messages[iParticle,iPrev, iRow-1]
                            messages[iParticle,iCur,iRow] = np.sum(tempDist)
                        normConstMessages[iParticle,iRow] = np.sum(messages[iParticle,:,iRow])
                        messages[iParticle,:,iRow] = messages[iParticle,:,iRow] / normConstMessages[iParticle,iRow]
                    # Column sum
                    tempDist = np.ones(nrComb)
                    for iCur in range(nrComb):
                        for iStrip in range(stripWidth-1):
                            tempDist[iCur] *= intPot[xDomain[iCur,iStrip], xDomain[iCur, iStrip+1]]
                        tempDist[iCur] *= intPot[xDomain[iCur,0], trajectorySMC[iParticle, nx-1, iSMC*stripWidth-1]]
                        tempDist[iCur] *= messages[iParticle, iCur, nx-2]
                    normConstMessages[iParticle,nx-1] = np.sum( tempDist )
                tempWeights = np.prod(normConstMessages, axis=1)
                CM[i] += np.log2(np.mean( tempWeights ))

                # Sample ancestors
                ancestors = discreteSampling( tempWeights / np.sum( tempWeights ), np.arange(N), N)
                     
                # Backward sampling
                for iParticle in range(N):
                    for iRow in range(nx)[::-1]:
                        tempDist = np.ones( nrComb )
                        for iCur in range(nrComb):
                            for iStrip in range(stripWidth):
                                if iStrip != stripWidth-1:
                                    tempDist[iCur] *= intPot[xDomain[iCur,iStrip], xDomain[iCur, iStrip+1]] 
                                if iRow < nx-1:
                                    tempDist[iCur] *= intPot[xDomain[iCur,iStrip], trajectorySMC[iParticle,iRow+1, iSMC*stripWidth+iStrip]]
                            tempDist[iCur] *= intPot[xDomain[iCur,0], trajectorySMC[ancestors[iParticle], iRow, iSMC*stripWidth-1]]
                            if iRow > 0:
                                tempDist[iCur] *= messages[ancestors[iParticle], iCur, iRow-1]
                        tempDist = tempDist / np.sum(tempDist)
                        # Sampling
                        curInd = discreteSampling( tempDist, range(nrComb), 1 )
                        for iStrip in range(stripWidth):
                            trajectorySMC[iParticle,iRow,iSMC*stripWidth+iStrip] = xDomain[curInd,iStrip]
                trajectorySMC[:, :, :iSMC*stripWidth] = trajectorySMC[ancestors.astype(int), :, :iSMC*stripWidth]
            CM[i] = CM[i] / (nx*ny)
        f = open(fileName, 'a')
        f.write(str(iIter) + ' ' + str(time.time() - startSMC) + ' ')
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

