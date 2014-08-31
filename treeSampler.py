"""treeSampler.py

Sample from 2D constrained channel using tree sampling.

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
    parser.add_option("--chains", type=int, help="number of independent chains")
    (args, options) = parser.parse_args()

    # Initialize arrays/matrices
    nx = args.nx
    ny = args.ny
    nrChains = args.chains

    iterNr = args.iter
    output = np.zeros( (nrChains, nx, ny) )
    #output[0,:,:] = (np.random.randint(2,size=nx*ny)).reshape((nx,ny))
    fileName = str(nx) + 'x' + str(ny) + 'TSTIME.txt'
    # Block structure
    temp = np.arange(ny)
    block1 = temp[::2]
    block2 = np.setdiff1d(temp, block1)

    # Interaction potentials
    intPot = np.ones( (2,2), dtype=np.uint64 )
    intPot[1,1] = 0
    xDomain = np.arange(2)

    f = open(fileName, 'w')
    f.write('nx ny\n')
    f.write(str(nx) + ' ' + str(ny) + '\n')
    f.write('iterNr timeElapsed zA zB\n')
    f.close()
    
    # Main loop
    for iIter in np.arange(iterNr):
        print iIter
        startTS = time.time()
        fMarginalA = np.zeros( (nrChains, ny), dtype=np.uint64  )
        fMarginalB = np.zeros( (nrChains, ny), dtype=np.uint64  )
        for iChain in range(nrChains):
            # --------------
            #   First tree
            # --------------
            #print output
            for iCol in block1:
                messages = np.ones( (len(xDomain), nx), dtype=np.uint64 )
                # Forward filtering
                for iRow in range(nx):
                    tempDist = np.ones( 2, dtype=np.uint64 )
                    if iCol > 0:
                        tempDist = tempDist * intPot[:, output[iChain,iRow,iCol-1]]
                    if iCol < ny-1:
                        tempDist = tempDist * intPot[:, output[iChain,iRow,iCol+1]]
                    if iRow > 0 and iRow < nx-1:
                        messages[:,iRow] = np.dot(intPot, tempDist * messages[:,iRow-1])
                    elif iRow == 0:
                        messages[:,iRow] = np.dot(intPot, tempDist)
                    else:
                        messages[:,iRow] = tempDist * messages[:,iRow-1]
                        fMarginalA[iChain,iCol] = np.sum( messages[:,iRow] )

                # Backward sampling
                for iRow in range(nx)[::-1]:
                    tempDist = np.ones( 2 )
                    if iCol > 0:
                        tempDist = tempDist * intPot[:, output[iChain,iRow,iCol-1]]
                    if iCol < ny-1:
                        tempDist = tempDist * intPot[:, output[iChain,iRow,iCol+1]]
                    if iRow > 0:
                        tempDist = tempDist * messages[:,iRow-1]
                    if iRow < nx-1:
                        tempDist = tempDist * intPot[:,output[iChain,iRow+1, iCol]]
                    tempDist = tempDist / np.sum(tempDist)
                    output[iChain,iRow, iCol] = discreteSampling( tempDist, xDomain, 1 )
            #print output
            # --------------
            #  Second block
            # -------------
            for iCol in block2:
                messages = np.ones( (len(xDomain), nx), dtype=np.uint64  )
                # Forward filtering
                for iRow in range(nx):
                    tempDist = np.ones( 2 )
                    if iCol > 0:
                        tempDist = tempDist * intPot[:, output[iChain,iRow,iCol-1]]
                    if iCol < ny-1:
                        tempDist = tempDist * intPot[:, output[iChain,iRow,iCol+1]]
                    if iRow > 0 and iRow < nx-1:
                        messages[:,iRow] = np.dot(intPot, tempDist * messages[:,iRow-1])
                    elif iRow == 0:
                        messages[:,iRow] = np.dot(intPot, tempDist)
                    else:
                        messages[:,iRow] = tempDist * messages[:,iRow-1]
                        fMarginalB[iChain,iCol] = np.sum( messages[:,iRow] )

                # Backward sampling
                for iRow in range(nx)[::-1]:
                    tempDist = np.ones( 2 )
                    if iCol > 0:
                        tempDist = tempDist * intPot[:, output[iChain,iRow,iCol-1]]
                    if iCol < ny-1:
                        tempDist = tempDist * intPot[:, output[iChain,iRow,iCol+1]]
                    if iRow > 0:
                        tempDist = tempDist * messages[:,iRow-1]
                    if iRow < nx-1:
                        tempDist = tempDist * intPot[:,output[iChain, iRow+1, iCol]]
                    tempDist = tempDist / np.sum(tempDist)
                    output[iChain, iRow, iCol] = discreteSampling( tempDist, xDomain, 1 )
        #print fMarginal
        #raw_input()
        fMarginal = np.zeros( (nrChains,ny) , dtype=np.uint64)
        fMarginal[:,:len(block1)] = fMarginalA[:,block1]
        fMarginal[:,len(block1):len(block1)+len(block2)] = fMarginalB[:,block2]
        f = open(fileName, 'a')
        f.write(str(iIter) + ' ' + str(time.time()-startTS) + ' ')
        np.savetxt(f, fMarginal.reshape( (1,nrChains*ny) ), fmt='%u')
        f.close()

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
