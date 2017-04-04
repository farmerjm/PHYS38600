# -*- coding: utf-8 -*-


'''

John Farmer




I ran this once with very high statistics and saved the object in a pickle.

A. Here I used theta = HA - HB.

I chose this because then calculating the resulting PDF can be done numerically very easily, without
simulation:  the joint PDFof a random variable Z = X - Y is the cross-correlation of the two variables.
So I construct X and Y from the binomial distribution and map the cross-correlation to the interval
(-1,1), and no excess computational expense is needed.

B. Said routine is absobed into the 'Trial' class.  HA and HB are fed in as parameters and
Trial.CalculateConfidenceInterval(self, sigmas) calculates the confidence interval to the specified sigma level.

C. Given in the first output plot.

Near (0,0) and (1,1) the interval is conservative, and near (-1,1) and (1,-1) the interval undercovers. Probing at the
likelihood functions reveals why:  at the extremes, the maximum of the correlated likelihood function is *not* at -1 or 1,
so none of the results fall into the region. At the other extremes, theta is 0, and if 0 or 2N heads are observed total,
that will always fall within the bound on zero.

The rest of the distrubution is essentially flat. The fluctuations are roughly consistant with what we expect, given sqrt(N)
error bars on each bin entry.

** The runtime here is absurdly large and I did not want to repeat it over and over, so I ran my experiment a huge
number of times, encapsulated all the simulation results into an object, and serialized it to disk. The plots shown are
from a large simulation whose serialization I have not included (it is a very large pickle!).

This approach lets me run the huge, computationally expensive simulation once and then play around with the
priors and analysis and plotting as much as I'd like without ever having to repeat the work.


D. Given in the final two output plots.

I used a prior quadratic in theta and the piecewise prior:

0, abs(\theta) < 0.5
1, abs(\theta) > 0.5

The results are exactly as one would expect.  For the piecewise prior, the middle line PA=PB is expectedly zero,
and it turns on sharply to normal coverage as the condition is met.

The quadratic prior has a similar behavior: around the line where theta=0, the coverage is expectedly zero, and it slowly
'turns on' as we have towards the edges.

'''



    
import random
import math
import matplotlib.pyplot as plt
import numpy as np
import pickle
import scipy.stats as stat
import scipy.special as sp

#Samples from the PDF and computes the mean.
#Maps random reals in (0,1) to the Poisson distribution using a Poisson lookup table
                                        
                                         
class Trial:
    nsamples=100
    posspace=np.linspace(0,1,10000)
    wholespace=np.linspace(-1,1,19999)
    
    def __init__(self, heads_a, heads_b):
        self.HA=heads_a
        self.HB=heads_b
        self.theta=[]
        self.likelihood=[]
        self.LmaxIndex=0
        self.Lmax=9999
        self.confLow=3000
        self.confLowIndex=3000
        self.confHigh=3000
        self.confHighIndex=3000
        self.nasamples=Trial.nsamples
        self.nbsamples=Trial.nsamples
        self.GenerateLikelihood()
        self.realTheta=(self.HA-self.HB)/Trial.nsamples
        
        self.numTestedTrials=0
        self.numSuccessfulTrials=0
        
    def CalculateConfidenceInterval(self, sigmas):
        SearchNumber=self.Lmax-(np.power(sigmas,2)/2)
        minDiff=100000
        minDiffIndex=0
        for i in range(self.LmaxIndex, len(self.theta)):
            diff = abs(SearchNumber-self.theta[i])
            if diff < minDiff:
                minDiff=diff
                minDiffIndex=i
        self.confHighIndex=minDiffIndex
        self.confHigh=Trial.wholespace[self.confHighIndex]
        minDiff=100000
        
        for i in range(0,self.LmaxIndex):
            diff = abs(SearchNumber-self.theta[i])
            if diff < minDiff:
                minDiff=diff
                minDiffIndex=i
        self.confLowIndex=minDiffIndex
        self.confLow=Trial.wholespace[self.confLowIndex]

    def TestTrial(self, testTrial):
        self.numTestedTrials=self.numTestedTrials+1
        if self.realTheta > testTrial.confLow and self.realTheta < testTrial.confHigh:
            self.numSuccessfulTrials=self.numSuccessfulTrials+1
            
    def GetCoverage(self):
        return self.numSuccessfulTrials/self.numTestedTrials
        
    def GenerateLikelihood(self):
        Acoeff=sp.binom(self.nasamples,self.HA)
        Bcoeff=sp.binom(self.nbsamples,self.HB)
    
        Alikelihood=Acoeff*np.power(Trial.posspace,self.HA)*np.power((1-Trial.posspace), self.nasamples-self.HA)
        Blikelihood=Bcoeff*np.power(Trial.posspace,self.HB)*np.power((1-Trial.posspace), self.nbsamples-self.HB)
        self.likelihood=np.correlate(Alikelihood,Blikelihood,mode='full')
        self.theta=np.log(self.likelihood)
        self.LmaxIndex=np.argmax(self.theta)
        self.Lmax=self.theta[self.LmaxIndex]
        
    def ApplyCutoffPrior(self,thetacopy):
        for index,i in enumerate(Trial.wholespace):
            if abs(Trial.wholespace[index]) < 0.5:
                thetacopy[index]=0
        return thetacopy
        
    def ApplyWeightedPrior(self,thetacopy):
        for index,entry in enumerate(thetacopy):
            thetacopy[index]=thetacopy[index]*pow(Trial.wholespace[index],2)
        return thetacopy

    def FindProbability(self,priorflag):
        thetacopy=list(self.likelihood)
        if priorflag == 1:
            thetacopy=self.ApplyCutoffPrior(thetacopy)
        if priorflag == 2:
            thetacopy=self.ApplyWeightedPrior(thetacopy)
        integral=sum(thetacopy)
        integralininterval=0
        for i in range(self.confLowIndex, self.confHighIndex):
            integralininterval=integralininterval+thetacopy[i]
        return integralininterval/integral
        
class DataContainer:
    
    def __init__(self, gran):
        self.theData=[]
        self.granularity=gran
        self.x=np.linspace(0,1,self.granularity+2)
        self.y=np.linspace(0,1,self.granularity+2)

    def FetchCoverageMatrix(self):
        CoverMatrix=[[0 for i in range(len(self.theData))] for j in range(len(self.theData[0]))]
        for i,rentry in enumerate(CoverMatrix):
            for j,centry in enumerate(CoverMatrix[0]):
                CoverMatrix[i][j]=self.theData[i][j].GetCoverage()
        return CoverMatrix
        
    def FetchProbability(self, flag):
        ProbMatrix=[[0 for i in range(len(self.theData))] for j in range(len(self.theData[0]))]
        for i,rentry in enumerate(ProbMatrix):
            for j,centry in enumerate(ProbMatrix[0]):
                ProbMatrix[i][j]=self.theData[i][j].FindProbability(flag)
        return ProbMatrix
        
        
    def PlotData(self, data, title, fname):
        plt.pcolor(self.x,self.y,data)
        plt.xlabel(r"$p_B$")
        plt.ylabel(r"$p_A$")
        plt.title(title)
        plt.colorbar()
        plt.axis([0,1, 0, 1])
        plt.savefig(fname+".png")
        plt.clf()
        
    def GenerateOutputs(self):
        for i in self.theData:
            for j in i:
                j.CalculateConfidenceInterval(1)
        CoverMat=self.FetchCoverageMatrix()
        self.PlotData(CoverMat, "Coverage", "lCoverage")
        Prior1Mat=self.FetchProbability(1)
        self.PlotData(Prior1Mat, r"P (prior $|\theta| > 0.5$)", "lprior1")
        Prior2Mat=self.FetchProbability(2)
        self.PlotData(Prior2Mat, r"P (prior $\theta^2$)", "lprior2")
plt.clf()
testTrial=Trial(100,20)
testTrial.CalculateConfidenceInterval(1)
plt.axvline(x=testTrial.confLow, ymin=0, ymax = 30,linestyle='--', linewidth=2, color='k')
plt.axvline(x=testTrial.confHigh, ymin=0, ymax = 30,linestyle='--', linewidth=2, color='k')
plt.title("Correlated PDF (p_a=1, p_b=0.2)")
plt.plot(Trial.wholespace, testTrial.theta)
plt.savefig("example_correlated_pdf.png")
plt.clf()



#The following section runs the simulation. Uncomment this and specify granularity and number of trials to
#do a simulation run and pickle the results. Uncomment all pickling to just run the simulation and
#operate on the results.

ntrials=100
container=DataContainer(10)
numsteps=container.granularity+1
for i in range(0,numsteps):
    print("i="+str(i))
    gridsample=np.zeros(numsteps)
    theGridRow=[]
    for j in range(0,numsteps):
        pa=i/container.granularity
        pb=j/container.granularity
        theDataTrial=Trial(pa*Trial.nsamples, pb*Trial.nsamples)
        for k in range(0,ntrials):  
            tHA=np.random.binomial(Trial.nsamples,i/container.granularity)
            tHB=np.random.binomial(Trial.nsamples,j/container.granularity)
            testTrial=Trial(tHA, tHB)
            testTrial.CalculateConfidenceInterval(1)
            theDataTrial.TestTrial(testTrial)
        theGridRow.append(theDataTrial)
    container.theData.append(theGridRow)

    
pickle.dump(container, open( "super_big_pickle_container.p", "wb" ) )

#These lines read the container object from the  pickle and perform the analysis.

#container=pickle.load(open("super_big_pickle_container.p","rb"))
container.GenerateOutputs()

