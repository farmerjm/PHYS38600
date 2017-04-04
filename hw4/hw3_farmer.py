# -*- coding: utf-8 -*-


'''

John Farmer

1.  a. Frequentist confidence intervals do not respect the physical limitations imposed on a system, ie non-negativity of a mass.
    b. Typically, that the probability to be found outside the interval on both sides of the distribution is 16% (or (100-CL)/2 %).
        Often constructed with a likelihood function, finding where the likelihood reduces by a half.
    c. We need a prior PDF to construct the posterior PDF for \mu_t.
    d. 1/\mu_t. He justifies that this is invariant over changes of power of \mu_t.
    e. Bayesian methods fail to be objective: they must be injected with a prior PDF  to construct the posterior from the likelihood function.
       Classical intervals fail to consider physical limitations on the measured parameter.
       Classical limits also handle systematics in a counterintuitive way, such that a bad calibration leads to a tighter confidence interval.
       It seems that generally people use classical statistics except when it produces things that 'seem' wrong, in which case use Bayesian.
    f. As Cousins did, perform classical analysis on the mean and statistical error and use a Bayesian analysis of the detector sensitivity.

    
2. I repeated this entire problem for a quadratic plot. The files ending in "_quad.pdf" are from the second iteration with a quadratic dataset.

    a. The data are shown in blue, the linear fit in red, and the quadratic fit in blue.
    
    b. The symmetry of the histogram reflects unbiased estimators.
    
    c. The functional form is:
        
            1/(2^{df/2}\Gamma(df/2)) x^{df/2-1}e^{-x/2}
               
        The single parameter, df, is the number of degrees of freedom in the fit. Since we have 15 data points, this is either 12 or 13.
        For the linear fit, we have two free parameters so df=13; for the quadratic fit with three free parameters, df=12.
        
        We expected the reduced chi square to be around 1, and this is the result for both fits.
        
        * For comparison I give a normalized reduced Chi2 distribution for df=12 and df=13. Overlaying them was not obviously easy, but comparing by-eye they are identical.
        
        I plotted reduced chi squares through because of their goodness-of-fit usefulness, but the conversion between the two statistics is simple.
        
    d.  In the case of the linear data, the fit gets worse. It is difficult to predict what happens here:  if we are lucky enough that we can fit
        some noise to the new x^2 degree of freedom, the X^2 will lower. However, the ndf has reduced by 1, so if there is overall no noise we can
        fit away, then the reduced chi square will rise.
        
        In the case of the quadratic data, the linear fit is abysmal and the quadratic fit is around 1. This is also expected.
        
3.  a. I sampled the distribution using the cdf; for reference I included both the plot of the distrubution and the cdf.

    b. Transforming error bars for log data is not entirely trivial because applying the logarithm literally yields asymmetric error bars.
        Instead, I transformed to first-order (d/dx log x), using \sigma_{D,log}=\sigma_D/D
        
    c. It takes a rather long time to run this with a large number of statistics (maybe I am doing something very inefficient).
    
        From running the experiment 500 times, I can say that poor sampling of the tails of the distribution leads to underestimation:  that is,
        we can see a bias in the distribution that favors the left side. I verified this by reducing the number of samples taken
        from the distribution by a factor of 10 and re-running, giving bins that are much less well-populated.  I attached outputs for both cases.
        
        Rather than wrestle with masking or reassigning garbage datasets post-log, I discarded all results for which the fit failed.
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
    



