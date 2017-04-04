# -*- coding: utf-8 -*-


'''

John Farmer

The code should be self-explanatory this time around.

In part (a) I give both a tabulated Poisson distribution and a sampled distribution.

* I sample the distribution by mapping the probabilities into intervals of (0,1) and generating a random number between 0 and 1.

In part (d) I resampled 10,000 times to demonstrate that the estimator is unbiased (100 times is not really enough).

In part (h), I report the percentage within one sigma. This is generally around 60%, but it varies based on the data.

Obviously if the 'data' generated is an outlier, then the percentage would be low.

It would be interesting to repeat this a great many times and histogram the result.
''' 
    
import random
import math
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stat

#Samples from the PDF and computes the mean.
#Maps random reals in (0,1) to the Poisson distribution using a Poisson lookup table
class SampleGenerator:

    
    def CalculatePoisson(self, inp):
        return ((self._mean**inp)*math.exp(-self._mean))/(math.factorial(inp))
        
    def __init__(self,mean,scaleFactor):
        self._mean = mean
        self._scaleFactor=scaleFactor
        self._maxX=self._mean*self._scaleFactor
        
        self.Entries=[]
        self.appendEntry=self.Entries.append
        self.cdf=[]        
        
        self.distMean=0
        
        self.PoissonTable = {x:  self.CalculatePoisson(x) for x in range(0, self._mean*(1+self._scaleFactor))}
        runningAverage=0
        for key in self.PoissonTable:
            runningAverage=runningAverage+self.PoissonTable[key]
            self.cdf.append(runningAverage)
        
    def GeneratePoint(self):
        randomNumber = random.random()
        index=-1
        if randomNumber < self.cdf[0]:
            index=0
        else:
            for i in range(0,len(self.cdf)-1):
                if randomNumber > self.cdf[i] and randomNumber < self.cdf[i+1]:
                    index=i+1
            #if index == -1:
                #print("Sample discarded.")
        if index != -1:    
            self.appendEntry(index)
        
    def GeneratePoints(self, numPoints):
        for i in range(1,numPoints):
            self.GeneratePoint()
        self.distMean=np.average(self.Entries)        


        
        
partA = SampleGenerator(3,3)
partA.GeneratePoints(10000)
n, bins, patches = plt.hist(partA.Entries,bins=partA._maxX, range=(0, partA._maxX))
plt.xlabel("n")
plt.ylabel("Entries")
plt.title("Poisson pdf (sampled)")
plt.savefig("part_a_sampled.png")
plt.clf()

plt.bar(partA.PoissonTable.keys(), partA.PoissonTable.values())
plt.xlabel("n")
plt.ylabel("P")
plt.title("Poisson pdf (tabulated)")
plt.savefig("part_a_tabulated.png")
plt.clf()


partB = SampleGenerator(3,3)
partB.GeneratePoints(50)
n, bins, patches = plt.hist(partB.Entries,bins=partB._maxX, range=(0, partB._maxX))
plt.xlabel("n")
plt.ylabel("Entries")
plt.title("50 Poisson samples")
plt.savefig("part_b.png")
plt.clf()

theSamples=[]
theSamples.append(partB.Entries)
theMeans=[]
theMeans.append(partB.distMean)
for i in range(0,100):
        theGen = SampleGenerator(3,3)
        theGen.GeneratePoints(50)
        theSamples.append(theGen.Entries)
        theMeans.append(theGen.distMean)
n, bins, patches = plt.hist(theMeans,bins=7*partB._maxX, range=(0, partB._maxX))
plt.axvline(x=theMeans[0], ymin=0, ymax = 30,linestyle='--', linewidth=2, color='k')
plt.xlabel("$\mu$")
plt.ylabel("Entries")
plt.title("Distribution of Poisson Means")
plt.figtext(.60,.7,"* Data = dashed line")
plt.savefig("part_d.png")
plt.clf()

biasMeans=[]
for i in range(0,10000):
    theGen=SampleGenerator(3,3)
    theGen.GeneratePoints(50)
    biasMeans.append(theGen.distMean)
n, bins, patches = plt.hist(biasMeans,bins=7*partB._maxX, range=(0, partB._maxX))
plt.xlabel("$\mu$")
plt.ylabel("Entries")
plt.title("Distribution of Poisson Means (10,000 entries)")
plt.figtext(.50,.65,"Unbiased (symmetric) as n $\dashrightarrow \infty$")
plt.savefig("part_d2.png")


plt.clf()

space=np.linspace(0,partB._maxX,num=1000)

theLogLikelihoods=[]
for dist in theSamples:
    total=sum(dist)
    #n, bins, patches = plt.hist(dist,bins=partB._maxX, range=(0, partB._maxX)))
    logLikelihood=[]
    logLikelihood=total*np.log(space)-space*len(dist)
    theLogLikelihoods.append(logLikelihood)
    #find out how to construct likelihood function

    

for index, event in enumerate(theLogLikelihoods):
    if index != 0:
        plt.plot(space,event, linestyle='--',linewidth=0.2)
    if index == 0:
        plt.plot(space,event,linestyle='solid', linewidth=5)zxc
plt.xlabel("$\mu$")
plt.ylabel("$\ln$ $\mathcal{L}$")
plt.title("Log-likelihood Functions")
plt.figtext(.50,.3,"* Data = solid blue")
plt.savefig("part_e.png")
plt.clf()

MLEGuesses=[]
for index, entry in enumerate(theLogLikelihoods):
    if index == 0:
        continue
    maxIndex=np.argmax(entry)
    MLEGuesses.append(space[maxIndex])
plt.hist(MLEGuesses, bins=5*partB._maxX, range=(0,partB._maxX),)
plt.xlabel("$\mu$")
plt.ylabel("Entries")
plt.title("MLE-derived simulation mean")
plt.savefig("part_f.png")
plt.clf()

dataDist=theLogLikelihoods[0]

plt.plot(space,dataDist)
plt.xlim(1,5)
maxData=max(dataDist)
maxIndex=np.argmax(dataDist)
plt.ylim(0.3*maxData,maxData+0.1)
plt.xlabel("$\mu$")
plt.ylabel("$\ln \mathcal{L}$")
plt.title("Graphical Variance")

SearchNumber=maxData-(1/2)
minDiff=1000
minDiffIndex=0
for i in range (maxIndex, len(dataDist)):
    diff = abs(SearchNumber-dataDist[i])
    if diff < minDiff:
        minDiff=diff
        minDiffIndex=i
varMax=minDiff
varMaxIndex=minDiffIndex
minDiff=1000

for i in range (0,maxIndex):
    diff = abs(SearchNumber-dataDist[i])
    if diff < minDiff:
        minDiff=diff
        minDiffIndex=i
varMin=minDiff
varMinIndex=minDiffIndex

maxMeanVar=space[varMaxIndex]
minMeanVar=space[varMinIndex]

plt.axvline(x=space[varMinIndex], ymin=0, ymax = 30,linestyle='--', linewidth=2, color='k')
plt.axvline(x=space[varMaxIndex], ymin=0, ymax = 30,linestyle='--', linewidth=2, color='k')
plt.figtext(.16,.7,"Black: $\ln\,\,\mathcal{L}_{max} - 1/2 $")
plt.savefig("part_g.png")
plt.clf()

count=0

theSimMeans=theMeans[1:]
for mean in theMeans:
    if mean < maxMeanVar and mean > minMeanVar:
        count=count+1

n, bins, patches = plt.hist(theMeans,bins=7*partB._maxX, range=(0, partB._maxX))
plt.xlabel("$\mu$")
plt.ylabel("Entries")
plt.title("Distribution of Poisson Means")
plt.axvline(x=space[varMinIndex], ymin=0, ymax = 30,linestyle='--', linewidth=2, color='k')
plt.axvline(x=space[varMaxIndex], ymin=0, ymax = 30,linestyle='--', linewidth=2, color='k')
fraction=round((count/len(theSimMeans)*100),3)
plt.figtext(.65,.7,str(fraction)+ " % within $\pm \sigma$")
plt.savefig("part_h.png")
    
print("Done")
