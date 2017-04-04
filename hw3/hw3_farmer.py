# -*- coding: utf-8 -*-


'''

John Farmer

The observed 'line' pattern was shocking at first and I believed it an error; however, after some thinking I can justify it.

I played around with a couple of distributions to see when 'edge effects' come in. I tinkered with varying the 
probabilities around the opposite diagonal than the one showing 1 sigma in the plot:  I saw notable edge effects and
very narrow confidence intervals until reaching around (0.7, 0.3), which is in agreement with when > 0 coverage
emerges along the main diagonal of the plot.

Mathematically one can also think that the zero-corners of the plot represent areas where two vastly disparate 
functions with are being convolved.

One possible issue that could be contributing to this is the approximation I am making for coverage. I specify a 
number of sigmas and compute the confidence interval by taking ln (L_{ma}) - \n_{sigma}^2/2, as given by Cowen.
This is useful because it generalizes to an arbitrary confidence level without doing integration, but it is an
approximation and its validity might be called into question

''' 
    
import random
import math
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stat

#Samples from the PDF and computes the mean.
#Maps random reals in (0,1) to the Poisson distribution using a Poisson lookup table


class MeasurementTaker:

    def __init__(self, resolution):
        self.theResolution=resolution
        
    def GeneratePointWithResolution(self, val):
        point=np.random.normal(loc=val,scale=self.theResolution)
        return point
    
class theLine:

    def __init__(self, degree):
        self.quadcoeff=1
        self.degree=degree
        self.m=2
        self.b=6
        self.res=2
        self.X=np.linspace(1,15, 15)
        self.Y=[]
        self.x=0
        self.residuals=0
        self.ChiSquare=0
        if self.degree == 1:
            self.BuildLine() 
        else: 
            self.BuildLineQuadratic()
        self.FitLine()
        
    def BuildLine(self):
        measurer = MeasurementTaker(2)
        for i, entry in enumerate(self.X):
            self.Y.append(measurer.GeneratePointWithResolution(self.m*entry+self.b))
    
    def BuildLineQuadratic(self):
        measurer = MeasurementTaker(2)
        for i, entry in enumerate(self.X):
            self.Y.append(measurer.GeneratePointWithResolution(self.quadcoeff*entry**2+self.m*entry+self.b))

    def FitLine(self):
        self.coeffs = np.polyfit(self.X, self.Y, 1)
        self.ChiSquare=np.sum((((self.coeffs[0]*self.X+self.coeffs[1])-self.Y)/self.res) ** 2)
        self.quadcoeffs=np.polyfit(self.X, self.Y,2)
        self.ChiSquareQuad=np.sum((((self.quadcoeffs[0]*self.X**2+self.quadcoeffs[1]*self.X+self.quadcoeffs[2])-self.Y)/self.res)**2)
        
    def PlotLine(self, title):
        plt.errorbar(self.X,self.Y,xerr=0,yerr=2)
        plt.plot(self.X,self.quadcoeffs[0]*self.X**2+self.quadcoeffs[1]*self.X+self.quadcoeffs[2])
        plt.plot(self.X,self.coeffs[0]*self.X+self.coeffs[1])
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("The Line")
        plt.savefig(title)
        plt.clf()

class theExponential:
    lookup_x=[]
    lookup_y=[]
    cdf=[]
    maxcdf=0

    def GenerateSample(self):
        randomNumber = random.uniform(theExponential.cdf[0],theExponential.maxcdf)
        index=-1
        if randomNumber < theExponential.cdf[0]:
            index=0
        else:   
            for i in range(0,len(theExponential.cdf)-1):
                if randomNumber > theExponential.cdf[i] and randomNumber < theExponential.cdf[i+1]:
                    index=i+1
        if index != -1:    
            self.samples.append(theExponential.lookup_x[index])
        
    def GenerateNSamples(self, numSamples):
        for i in range(0, numSamples):
            self.GenerateSample()
            
    def AnalyzeDistro(self, index):
        y,binEdges = np.histogram(self.samples,bins=10)
        bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
        menStd     = np.sqrt(y)
        width      = 0.20
        plt.clf()
        if index == 1:     
            plt.bar(bincenters, y, width=width, yerr=menStd, ecolor='g')
            plt.xlabel("Value")
            plt.ylabel("Entries")
            plt.title(str(len(self.samples))+" exponential samples")
            plt.savefig("3b_exp_samples.png")
            plt.clf()
        self.logsamples=np.log(y)
        logcoeffs = np.polyfit(bincenters, self.logsamples, 1)
        if index == 1:
            plt.bar(bincenters,self.logsamples,width=width, yerr=menStd/y, ecolor='g')
            plt.xlabel("Value")
            plt.ylabel("log Entries")
            plt.title(str(len(self.samples))+" exponential samples")
            theFitX=np.linspace(0,5,1000)
            theFitY=theFitX*logcoeffs[0]+logcoeffs[1]
            plt.plot(theFitX,theFitY)
            plt.savefig("3b_exp_samples_log.png")
            plt.clf()
        return -1*logcoeffs[0]

        
        
    def __init__(self, nSamples):
        self.samples=[]
        self.logbins=[]
        self.GenerateNSamples(nSamples)
        
theExponential.lookup_x=np.linspace(0, 5, 10000)
theExponential.lookup_y=np.exp(-theExponential.lookup_x)
runningAverage=0
for val in theExponential.lookup_y:
    runningAverage=runningAverage+val
    theExponential.cdf.append(runningAverage)
theExponential.maxcdf=theExponential.cdf[len(theExponential.cdf)-1]
plt.clf()

print("Running...")
plt.plot(theExponential.lookup_x, theExponential.lookup_y)
plt.xlabel("x")
plt.ylabel("$e^{-x}$")
plt.title("Exponential distribution")
plt.savefig("3_exponential_dist.png")
plt.clf()

plt.plot(theExponential.lookup_x, theExponential.cdf)
plt.xlabel("x")
plt.ylabel("cdf") 
plt.title("Exponential cdf")
plt.savefig("3_exponential_cdf.png")
plt.clf()

for i in range(0,2):
    fileEnding=0
    degree=i+1
    if i == 0:
        fileEnding=".png"
    else:
        fileEnding="_quad.png"
    Lines=[]
    slopes=[]
    intercepts=[]
    quads=[]
    chisqs=[]
    chisqquads=[]
    for j in range(0,1000):
        line = theLine(degree)
        Lines.append(line)
        if j == 1: 
            line.PlotLine("2a_line"+fileEnding)
        if i == 0:    
            slopes.append(line.coeffs[0])
            intercepts.append(line.coeffs[1])
        else:
            quads.append(line.quadcoeffs[0])
            slopes.append(line.quadcoeffs[1])
            intercepts.append(line.quadcoeffs[2])
        chisqs.append(line.ChiSquare/13)
        chisqquads.append(line.ChiSquareQuad/12)
    
    plt.hist(slopes, bins=100)
    plt.xlabel("m")
    plt.ylabel("Entries")
    plt.title("Slopes histogram")
    plt.savefig("2b_slopes"+fileEnding)
    plt.clf()
    
    plt.hist(intercepts, bins=100)
    plt.xlabel("b")
    plt.ylabel("Entries")
    plt.title("Intercepts histogram")
    plt.savefig("2b_intercepts"+fileEnding)
    plt.clf()
    
    if i == 1:
            plt.hist(intercepts, bins=100)
            plt.xlabel("a (quadratic coefficient)")
            plt.ylabel("Entries")
            plt.title("Quadratic coefficient histogram")
            plt.savefig("2b_quads"+fileEnding)
            plt.clf()
    
    plt.hist(chisqs, bins=100)
    plt.xlabel("X^2 / ndf")
    plt.ylabel("Entries")
    plt.title("Chi-square of linear fit")
    plt.savefig("2c_chisq"+fileEnding)
    plt.clf()
    
    
    plt.hist(chisqquads, bins=100)
    plt.xlabel("X^2 / ndf")
    plt.ylabel("Entries")
    plt.title("Chi-square of quadratic fit")
    plt.savefig("2d_chisq2"+fileEnding)
    plt.clf()


    
    theNdf=0
    if i ==1:
        theNdf=12
    else:
        theNdf=13
    chispace=np.linspace(0,theNdf*3,1000)
    chidist=stat.chi2(theNdf,1)
    plt.plot(chispace/theNdf, chidist.pdf(chispace))
    plt.xlabel("X^2")
    plt.ylabel("P")
    plt.title("Chi-square distribution (ndf ="+str(theNdf)+")")
    plt.savefig("2d_chisq2pdf"+fileEnding)
    plt.clf()

Taus=[]
for i in range(0,500):
    if i % 100 == 0:
        print(i)
    exp = theExponential(500)
    result=exp.AnalyzeDistro(i)
    if math.isnan(result) == False:
        Taus.append(result)
print(Taus)
plt.hist(Taus, bins=20)
plt.xlabel("Tau")
plt.ylabel("Entries")
plt.title("Estimated Tau")
plt.savefig("3c_tau_hist_500samples.png")

Taus=[]
for i in range(0,500):
    if i % 100 == 0:
        print(i)
    exp = theExponential(50)
    result=exp.AnalyzeDistro(i)
    if math.isnan(result) == False:
        Taus.append(result)
print(Taus)
plt.hist(Taus, bins=20)
plt.xlabel("Tau")
plt.ylabel("Entries")
plt.title("Estimated Tau")
plt.savefig("3c_tau_hist_50samples.png")
    



