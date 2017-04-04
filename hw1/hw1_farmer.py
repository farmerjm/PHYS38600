# -*- coding: utf-8 -*-


'''

John Farmer

I use a Poisson distribution with mean 3.
    
    (a) I did this by creating a table of Poisson probabilties, generating a random number on the interval (0,1) and mapping that to the Poisson distribution.
    
    (b) Expected Gaussian behavior. The blip in the graph around the mean seems to just be an artifact of binning, somehow.
    
    (c) The variance follows 1/sqrt(n) behavior as expected. Since I chose a poisson distribution with lambda 3, then the expected variance at, say, N=30 is sqrt(3)/sqrt((5)=0.245. This agrees with what we see.
    
    (d) The third moment is never large and quickly levels off to just noise at around n=30.
    
    N=30 is the value usually cited as the number of samples necessary for the central limit theorem to hold for skewed distributions; this is hence consistent with my result.
    
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
        self.doPartA=0
        
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
    
    def BuildSampledDistribution(self):
        self.distMean=np.average(self.Entries)
        if self.doPartA == 1:
            n, bins, patches = plt.hist(self.Entries,bins=self._maxX, range=(0, self._maxX))
            plt.xlabel("Sample")
            plt.ylabel("Entries")
            plt.title("Example Poisson sampling")
            plt.savefig("part_a.png")
            plt.clf()
            self.doPartA==0
        
class SampleWrapper:
    
    mean=3
    scaleFactor=3

    nPDFSamples=40
    basesamples=10
   
    targetn = 100
    muDistro=0
    sigmaDistro=0
    
    Variances=[]
    Skews=[]
    nSamples=[]
    
    def Resample(self,n):
        for i in range(self.basesamples,self.targetn):
            meanDistro=[]
            for j in range(0,100):
                theGen=SampleGenerator(self.mean, self.scaleFactor)
                theGen.GeneratePoints(i)
                theGen.BuildSampledDistribution()
                meanDistro.append(theGen.distMean)
                
            variance=np.std(meanDistro)
            self.Variances.append(variance)
            self.Skews.append(stat.skew(meanDistro)*pow(variance,(3./2.)))
            self.nSamples.append(i)
        self.PrintStatistics()

    def PrintStatistics(self):
        theGen=SampleGenerator(self.mean,self.scaleFactor)
        theGen.GeneratePoints(1000000)
        theGen.doPartA=1
        theGen.BuildSampledDistribution()
        meanDistro=[]
        for j in range(0,10000):
            theGen=SampleGenerator(self.mean, self.scaleFactor)
            theGen.GeneratePoints(50)
            theGen.BuildSampledDistribution()
            meanDistro.append(theGen.distMean)
        n, bins, patches=plt.hist(meanDistro, range=(self.mean-2,self.mean+2),bins=40)
        plt.xlabel("Mean")
        plt.ylabel("Entries")
        plt.title("Distribution of Mean from 10000 Poisson Samplings")
        plt.savefig("part_b.png")
        plt.clf()
        plt.plot(self.nSamples, self.Variances)
        plt.xlabel("N")
        plt.ylabel("Variance")
        plt.title("Behavior of Poisson Variance as a function of Samples")
        plt.savefig("part_c.png")
        plt.clf()
        plt.plot(self.nSamples, self.Skews)
        plt.xlabel("Number of Resamplings")
        plt.ylabel("Skewness")
        plt.title("Behavior of Poisson Third Moment as a function of Samples")
        plt.savefig("part_d.png")
        plt.clf()        
    
theWrapper = SampleWrapper()
#theWrapper.nPDFSamples=50   #little 'n', number of samples per sampling of the pmf
theWrapper.Resample(100000)   #Big 'N', number of resamplings of the pmf with n samples each

print("Done")
