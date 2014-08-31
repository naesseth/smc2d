"""treeSampler.py

Sample from 2D constrained channel using tree sampling + larger blocks/strips.

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
    parser.add_option("--stripWidth", type=int, help="width of the strips")
    (args, options) = parser.parse_args()

    # Initialize arrays/matrices
    nx = args.nx
    ny = args.ny
    nrChains = args.chains
    stripWidth = args.stripWidth
    iterNr = args.iter
    output = np.zeros( (nrChains, nx, ny) )
    fileName = str(nx) + 'x' + str(ny) + 'TSstripw' + str(stripWidth) + '.txt'
    # Block structure
    index = np.arange(ny)
    nrInd1 = int(np.ceil(float(ny)/float(stripWidth)/2))
    nrInd2 = ny/stripWidth - nrInd1
    #print nrInd1, nrInd2
    block1 = np.zeros( (nrInd1,stripWidth), dtype=np.uint64 )
    block2 = np.zeros( (nrInd2,stripWidth), dtype=np.uint64 )
    temp = 0
    for iIndex in range(nrInd1):
        block1[iIndex,:] = index[stripWidth*iIndex+temp:stripWidth*(iIndex+1)+temp]
        temp = temp + stripWidth
    temp = stripWidth
    for iIndex in range(nrInd2):
        block2[iIndex,:] = index[stripWidth*iIndex+temp:stripWidth*(iIndex+1)+temp]
        temp = temp + stripWidth    
    #print block1, block2
    
    # Interaction potentials
    intPot = np.ones( (2,2), dtype=np.uint64 )
    intPot[1,1] = 0
    nrComb = 2**stripWidth
    xDomain = np.zeros((nrComb,stripWidth), dtype=np.uint64)
    for iIter in range(nrComb):
		tempStr = np.binary_repr(iIter, width=stripWidth)
		for iStrip in range(stripWidth):
			xDomain[iIter, iStrip] = int(tempStr[iStrip])
    #nrDimArray = '(' + '2,'*stripWidth + str(nx) + ')'
    #print xDomain
	
    # Create file
    f = open(fileName, 'w')
    f.write('nx ny\n')
    f.write(str(nx) + ' ' + str(ny) + '\n')
    f.write('iterNr timeElapsed zA zB\n')
    f.close()
    
    # Main loop
    for iIter in np.arange(iterNr):
        print iIter
        startTS = time.time()
        fMarginalA = np.zeros( (nrChains, nrInd1) )
        fMarginalB = np.zeros( (nrChains, nrInd2) )
        for iChain in range(nrChains):
            # --------------
            #   First tree
            # --------------
            #print output[iChain,:,:].reshape( (nx,ny) )
            for iBlock in range(nrInd1):
                messages = np.ones( (nrComb, nx-1) )
                normConstMessages = np.zeros( nx )
                # Forward filtering
                for iRow in range(nx-1):
                    for iCur in range(nrComb):
                        tempDist = np.ones(nrComb)
                        for iPrev in range(nrComb):
                            for iStrip in range(stripWidth-1):
                                tempDist[iPrev] *= intPot[xDomain[iPrev,iStrip], xDomain[iPrev, iStrip+1]] 
                                tempDist[iPrev] *= intPot[xDomain[iPrev,iStrip], xDomain[iCur, iStrip]]
                            tempDist[iPrev] *= intPot[xDomain[iPrev,stripWidth-1], xDomain[iCur, stripWidth-1]]
                            if block1[iBlock, 0] > 0:
                                tempDist[iPrev] *= intPot[xDomain[iPrev,0], output[iChain, iRow, block1[iBlock, 0]-1]]
                            if block1[iBlock, stripWidth-1] < ny-1:
                                tempDist[iPrev] *= intPot[xDomain[iPrev,-1], output[iChain, iRow, block1[iBlock, stripWidth-1]+1]]
                            if iRow > 0:
                                tempDist[iPrev] *= messages[iPrev, iRow-1]
                        messages[iCur,iRow] = np.sum(tempDist)
                    normConstMessages[iRow] = np.sum(messages[:,iRow])
                    messages[:,iRow] = messages[:,iRow] / normConstMessages[iRow]
                # Column sum
                tempDist = np.ones(nrComb)
                for iCur in range(nrComb):
                    for iStrip in range(stripWidth-1):
                        tempDist[iCur] *= intPot[xDomain[iCur,iStrip], xDomain[iCur, iStrip+1]] 
                    if block1[iBlock, 0] > 0:
                        tempDist[iCur] *= intPot[xDomain[iCur,0], output[iChain, nx-1, block1[
iBlock, 0]-1]]
                    if block1[iBlock, -1] < ny-1:
                        tempDist[iCur] *= intPot[xDomain[iCur,-1], output[iChain, nx-1, block1[iBlock, -1]+1]]
                    tempDist[iCur] *= messages[iCur, nx-2]
                normConstMessages[-1] = np.sum( tempDist )
                fMarginalA[iChain,iBlock] = np.sum( np.log(normConstMessages) )
                #raw_input()
	     		
                # Backward sampling
                for iRow in range(nx)[::-1]:
                    tempDist = np.ones( nrComb )
                    for iCur in range(nrComb):
                        for iStrip in range(stripWidth-1):
                            tempDist[iCur] *= intPot[xDomain[iCur,iStrip], xDomain[iCur, iStrip+1]] 
                            if iRow < nx-1:
                                tempDist[iCur] *= intPot[xDomain[iCur,iStrip], output[iChain,iRow+1, block1[iBlock, iStrip]]]
                        if iRow < nx-1:
                            tempDist[iCur] *= intPot[xDomain[iCur,-1], output[iChain,iRow+1, block1[iBlock, stripWidth-1]]]
                        if block1[iBlock, 0] > 0:
                            tempDist[iCur] *= intPot[xDomain[iCur,0], output[iChain, iRow, block1[iBlock, 0]-1]]
                        if block1[iBlock, stripWidth-1] < ny-1:
                            tempDist[iCur] *= intPot[xDomain[iCur,-1], output[iChain, iRow, block1[iBlock, stripWidth-1]+1]]
                        if iRow > 0:
                            tempDist[iCur] *= messages[iCur, iRow-1]
                    tempDist = tempDist / np.sum(tempDist)
                    curInd = discreteSampling( tempDist, range(nrComb), 1 )
                    for iStrip in range(stripWidth):
                        output[iChain,iRow,block1[iBlock, iStrip]] = xDomain[curInd,iStrip]

            # --------------
            #  Second block
            # -------------
            for iBlock in range(nrInd2):
                messages = np.ones( (nrComb, nx-1) )
                normConstMessages = np.zeros( nx )
                # Forward filtering
                for iRow in range(nx-1):
                    for iCur in range(nrComb):
                        tempDist = np.ones(nrComb)
                        for iPrev in range(nrComb):
                            for iStrip in range(stripWidth-1):
                                tempDist[iPrev] *= intPot[xDomain[iPrev,iStrip], xDomain[iPrev, iStrip+1]] 
                                tempDist[iPrev] *= intPot[xDomain[iPrev,iStrip], xDomain[iCur, iStrip]]
                            tempDist[iPrev] *= intPot[xDomain[iPrev,-1], xDomain[iCur, -1]]
                            if block2[iBlock, 0] > 0:
                                tempDist[iPrev] *= intPot[xDomain[iPrev,0], output[iChain, iRow, block2[iBlock, 0]-1]]
                            if block2[iBlock, -1] < ny-1:
                                tempDist[iPrev] *= intPot[xDomain[iPrev,-1], output[iChain, iRow, block2[iBlock, -1]+1]]
                            if iRow > 0:
                                tempDist[iPrev] *= messages[iPrev, iRow-1]
                        messages[iCur,iRow] = np.sum(tempDist)
                    normConstMessages[iRow] = np.sum(messages[:,iRow])
                    messages[:,iRow] = messages[:,iRow] / normConstMessages[iRow]
                        
                #print messages[:,iRow]
						
                tempDist = np.ones(nrComb)
                for iCur in range(nrComb):
                    for iStrip in range(stripWidth-1):
                        tempDist[iCur] *= intPot[xDomain[iCur,iStrip], xDomain[iCur, iStrip+1]] 
                    if block2[iBlock, 0] > 0:
                        tempDist[iCur] *= intPot[xDomain[iCur,0], output[iChain, nx-1, block2[iBlock, 0]-1]]
                    if block2[iBlock, -1] < ny-1:
                        tempDist[iCur] *= intPot[xDomain[iCur,-1], output[iChain, nx-1, block2[iBlock, -1]+1]]
                    tempDist[iCur] *= messages[iCur, nx-2]
                normConstMessages[-1] = np.sum( tempDist )
                fMarginalB[iChain,iBlock] = np.sum( np.log(normConstMessages) )
                #print fMarginalB[iChain,iBlock]
                #raw_input()
				
                # Backward sampling
                for iRow in range(nx)[::-1]:
                    tempDist = np.ones( nrComb )
                    for iCur in range(nrComb):
                        for iStrip in range(stripWidth-1):
                            tempDist[iCur] *= intPot[xDomain[iCur,iStrip], xDomain[iCur, iStrip+1]]
                            if iRow < nx-1:
                                tempDist[iCur] *= intPot[xDomain[iCur,iStrip], output[iChain,iRow+1, block2[iBlock, iStrip]]]
                        if iRow < nx-1:
                            tempDist[iCur] *= intPot[xDomain[iCur,-1], output[iChain,iRow+1, block2[iBlock, -1]]]
                        if block2[iBlock, 0] > 0:
                            tempDist[iCur] *= intPot[xDomain[iCur,0], output[iChain, iRow, block2[iBlock, 0]-1]]
                        if block2[iBlock, -1] < ny-1:
                            tempDist[iCur] *= intPot[xDomain[iCur,-1], output[iChain, iRow, block2[iBlock, -1]+1]]
                        if iRow > 0:
                            tempDist[iCur] *= messages[iCur, iRow-1]
                    tempDist = tempDist / np.sum(tempDist)
                    curInd = discreteSampling( tempDist, range(nrComb), 1 )
                    for iStrip in range(stripWidth):
                        output[iChain,iRow, block2[iBlock, iStrip]] = xDomain[curInd,iStrip]
        #print fMarginal
        #raw_input()
        #print output[0,:,:].reshape( (nx,ny) )
        #raw_input()
        fMarginal = np.zeros( (nrChains,nrInd1+nrInd2) )
        fMarginal[:,:len(block1)] = fMarginalA
        fMarginal[:,len(block1):len(block1)+len(block2)] = fMarginalB
        f = open(fileName, 'a')
        f.write(str(iIter) + ' ' + str(time.time()-startTS) + ' ')
        np.savetxt(f, fMarginal.reshape( (1,nrChains*(nrInd1+nrInd2)) ) )
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
