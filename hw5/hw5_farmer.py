# -*- coding: utf-8 -*-


'''

John Farmer


Again, I sampled the PDF using the CDF.

1.  Variances are given on the plots. For Chi2, I used 10 bins. KS is the least precise and MLE the most. This is a little surprising; I expected Chi2 to have the least, since it is binned.  However, perhaps the underlying similarity between MLE and Chi2 has something to do with this.  I suppose it is a bit strange to use KS to find the value of a parameter like this:  usually it is just used to compare relative frequencies.

2.  MLE has the highest power, acccording to the Neyman-Pearson lemma, and KS has the least (again somewhat surprising).

In blue is P=0.5 and in green P=0.0.

'''



    
import random
import math
import matplotlib.pyplot as plt
import numpy as np
import pickle
import scipy.stats as stat
import scipy.special as sp
import statistics as st

#Samples from the PDF and computes the mean.
#Maps random reals in (0,1) to the Poisson distribution using a Poisson lookup table
           

class theDistro:
    lookup_x=[]
    lookup_y=[]
    gencdf=[]
    normcdf=[]
    maxcdf=0
    thetabins=np.linspace(0,2*math.pi,10, endpoint=False)

    def GenerateSample(self):
        randomNumber = random.uniform(theDistro.gencdf[0],theDistro.maxcdf)
        index=-1
        if randomNumber < theDistro.gencdf[0]:
            index=0
        else:   
            for i in range(0,len(theDistro.gencdf)-1):
                if randomNumber > theDistro.gencdf[i] and randomNumber < theDistro.gencdf[i+1]:
                    index=i+1
        if index != -1:    
            self.samples.append(theDistro.lookup_x[index])
        
    def GenerateNSamples(self, numSamples):
        for i in range(0, numSamples):
            self.GenerateSample()
            
    def __init__(self, nSamples):
        self.samples=[]
        self.logbins=[]
        self.ksP=0;
        self.chiP=0;
        self.mlP=0;
        self.GenerateNSamples(nSamples)
        self.trialCDF=np.zeros(nSamples)
        self.ksthetaspace=np.linspace(0,2*math.pi,200)
        runningTotal=0
        self.sortedsamples=np.sort(self.samples)
        for i in range(0, nSamples):
            runningTotal+=1
            self.trialCDF[i]=runningTotal
        self.trialCDF=self.trialCDF/runningTotal
        self.loglikelihood=np.sum(np.log((1+0.5*np.cos(self.samples))))
        
        self.chi_null=0
        self.chi_zero=0
        self.ks_null=0
        self.ks_zero=0
        self.l_null=0
        self.l_zero=0
        self.testone=0

    def ChiSquareMinFit(self):        
        p_list=np.linspace(0.0,1,101)
        chi2_list=np.zeros(101)
        arr=np.cos(theDistro.thetabins)
        for i,p in enumerate(p_list):
            model=(200/10)*(1+p*arr)
            data=BinData(self.sortedsamples)
            self.testone=data[0]
            chi2_list[i]=ComputeChiSquare(data,model)
            if p == 0.0:
                self.chi_zero=chi2_list[i]
            if p == 0.5:
                self.chi_null=chi2_list[i]
            if p ==0.5:
                plt.plot(self.thetabins, data)
                plt.plot(self.thetabins, model)
                plt.xlabel(r"$\theta$")
                plt.ylabel("Entries")
                plt.title("Binned chisquare test example")
                plt.savefig("Chisquare_ex.png")
                plt.clf()
        self.chiP=p_list[np.argmin(chi2_list)]


    def KSMinFit(self):
        ksStats=np.zeros(101)
        p_list=np.linspace(0.0,1,101)
        for j,p in enumerate(p_list):
            cdf=(self.sortedsamples+p*np.sin(self.sortedsamples))/(2*math.pi)
            ksStats[j]=CalculateKsStatistic(self.trialCDF, cdf)
            #print("p, ks: " + str(p) + " " + str(ksStats[j]))
            if p == 0:
                self.ks_zero=ksStats[j]
            
            if p == 0.5:
                self.ks_null=ksStats[j]
                plt.plot(self.sortedsamples, self.trialCDF)
                plt.plot(self.sortedsamples, cdf)
                plt.xlabel(r"$\theta$")
                plt.ylabel("cdf")
                plt.title("Trial CDF and Data CDF (KS)")
                plt.savefig("samplecdf.png")
                plt.clf()
        self.ksP=p_list[np.argmin(ksStats)]
        
    def MLEfit(self):
        L_list=np.zeros(101)
        p_list=np.linspace(0.0,1,101)
        for i,p in enumerate(p_list):
            likelihood=np.sum(np.log((1+p*np.cos(self.samples))))
            L_list[i]=likelihood
            if p == 0.5:
                truelike=L_list[i]
        maxlike=max(L_list)
        #print(maxlike)
        #print(truelike)
        self.l_null=-2*(truelike-maxlike)
        self.l_zero=-2*(L_list[0]-maxlike)
        maxindex=np.argmax(L_list)
        self.mlP=p_list[maxindex]
        
    def Fit(self):
        self.ChiSquareMinFit()
        self.KSMinFit()
        self.MLEfit()
        
def ComputeChiSquare(data, model):
    diff=np.zeros(len(data))
    for i in range(0, len(data)):
        diff[i]=(data[i]-model[i])**2/(model[i])
    return np.sum(diff)
        
def CalculateKsStatistic(data, model):
    diff=0
    for i in range(0, len(data)):
        test = math.sqrt((data[i]-model[i])**2)
        if test > diff:
            diff = test
    return diff

def BinData(inpData):
    theBins=np.zeros(len(theDistro.thetabins))
    for dat in inpData:
        for i in range(0,len(theDistro.thetabins)-1):
            if dat > theDistro.thetabins[i] and dat < theDistro.thetabins[i+1]:
                theBins[i]=theBins[i]+1
        if dat > theDistro.thetabins[len(theDistro.thetabins)-1]:
            theBins[len(theDistro.thetabins)-1]=theBins[len(theDistro.thetabins)-1]+1
    return theBins

    
print("Running...")
theDistro.lookup_x=np.linspace(0, 2*math.pi, 10000)
theDistro.lookup_y=1+0.5*np.cos(theDistro.lookup_x)
theDistro.lookup_y=1+0.5*np.cos(theDistro.lookup_x)
runningAverage=0    
for val in theDistro.lookup_y:
    runningAverage=runningAverage+val
    theDistro.gencdf.append(runningAverage)
theDistro.maxcdf=theDistro.gencdf[len(theDistro.gencdf)-1]
'''
plt.plot(theDistro.lookup_x, theDistro.gencdf)
plt.title("p=0.5 CDF")
plt.savefig("0.5cdf.png")
plt.clf
'''
#theDistro.gencdf=theDistro.gencdf/theDistro.maxcdf

plt.clf()
distro = theDistro(10000)
plt.hist(distro.samples, bins=16, range=(0,2*math.pi))
plt.xlabel(r'$\theta$')
plt.ylabel("N")
plt.title("10000 samples from PDF")
plt.savefig("10000samples.png")
plt.clf()



print("Begin loop...")
distros=[]
for i in range(0,10000):
    print(i)
    newDistro=theDistro(200)
    newDistro.Fit()
    distros.append(distro)

pickle.dump(distros, open( "le_pickle.p", "wb" ) )




#distros=pickle.load(open("le_pickle.p","rb"))

ksplist=[]
chiplist=[]
mlList=[]
chi2_null=[]
chi2_zero=[]
ks_null=[]
ks_zero=[]
l_null=[]
l_zero=[]
test=[]

for distro in distros:  
    ksplist.append(distro.ksP)
    chiplist.append(distro.chiP)
    mlList.append(distro.mlP)
    
    chi2_null.append(distro.chi_null)
    chi2_zero.append(distro.chi_zero)
    ks_null.append(distro.ks_null)
    ks_zero.append(distro.ks_zero)
    l_null.append(distro.l_null)
    l_zero.append(distro.l_zero)
    test.append(distro.testone)

varml=st.pvariance(mlList)
varchi2=st.pvariance(chiplist)
varks=st.pvariance(ksplist)
plt.clf()
plt.hist(mlList)
plt.xlabel("P")
plt.ylabel("Entries")
plt.title("MLE P (variance " + str(varml) + ")")
plt.savefig("MLE_p.png")
plt.clf()

plt.hist(ksplist)
plt.xlabel("P")
plt.ylabel("Entries")
plt.title("KS P (variance " + str(varks) + ")")
plt.savefig("KSmin_p.png")
plt.clf()

plt.hist(chiplist)
plt.xlabel("P")
plt.ylabel("Entries")
plt.title("Chi-square P (variance " + str(varchi2) + ")")
plt.savefig("chi2min_p.png")
plt.clf()

chi2level=16.919
plt.hist(chi2_null,bins=50,range=(0,120))
plt.hist(chi2_zero,bins=50,range=(0,120))
plt.axvline(x=chi2level, ymin=0, ymax = 30,linestyle='--', linewidth=2, color='k')
plt.xlabel("Chi-squared")
plt.ylabel("Entries")
plt.title("Chi-Square statistic (2 hypothesis)")
chi2fp=0
for distro in distros:
    if distro.chi_zero < chi2level:
        chi2fp+=1
chi2_power=1- chi2fp/10000
plt.figtext(.65,.7,"Power: " + str(chi2_power))
plt.savefig("chi2_2hypotheses.png")
plt.clf()    
    
kslevel=1.36/math.sqrt(200)
plt.hist(ks_null)
plt.hist(ks_zero)
plt.axvline(x=kslevel, ymin=0, ymax = 30,linestyle='--', linewidth=2, color='k')
plt.xlabel("KS")
plt.ylabel("Entries")
ksfp=0
for distro in distros:
    if distro.ks_zero < kslevel:
        ksfp+=1
ks_power=1- ksfp/10000
plt.figtext(.70,.85,"Power: " + str(ks_power))
plt.title("KS statistic (2 hypothesis)")
plt.savefig("KSstat_2hypotheses.png")
plt.clf()

llevel=3.841
plt.hist(l_null,bins=50,range=(-10,100))
plt.hist(l_zero,bins=50,range=(10,100))
plt.axvline(x=llevel, ymin=0, ymax = 30,linestyle='--', linewidth=2, color='k')
plt.xlabel("Llkelihood Ratio")
plt.ylabel("Entries")
plt.title("Likelihood ratio statistic (2 hypothesis)")
lfp=0
for distro in distros:
    if distro.l_zero < llevel:
        lfp+=1
l_power=1- lfp/10000
plt.figtext(.65,.7,"Power: " + str(l_power))
plt.savefig("likelihoodratio_2hypotheses.png")
plt.clf()    
    


    
