# -*- coding: utf-8 -*-


'''

John Farmer


1.  a. Done.
    b. The second array is the frequency bins. The FFT algorithm I used arranged the bins in a different order, so I used a different freq. array in my plots.
    
    I checked the normalization by verifying Parseval's theorem. I found that a normalization factor of 1/N was needed in the frequency domain, which is what I expected from the definitions of power in lecture.
    
2.  The Fourier transform of a real function has negative frequencies that are the complex conjugate of its positive ones; for this problem I work with just the positive frequencies for clarity.

    a. Done.
    b. Done.
    c. As expected, they overlap the most where there is zero offset. There are some edge effects visible along the other endpoints.
    
3.  Again, I will work with the positive part of the function, but to preserve the power I need to be a bit careful.

    Every entry *except* zero and the Nyquist frequency has a counterpart in the negative-frequency domain. To accurately calculate the power, I need to take this into account. The easiest way is by doubling every entry of the FFt
except the endpoints (Nyquist frequency and zero).


    a. Done.
    b. The RMS of the timestream gives the RMS energy. Again, compute this as in 2b, but this time we have to be a little bit careful:  there is a factor of 2 from the overall normalization of using the 'real' fft.
    c. Done.

    

'''



    
import random
import math
import matplotlib.pyplot as plt
import numpy as np
import pickle
import scipy.stats as stat
import scipy.special as sp
import scipy.signal as sig
import statistics as st



thelinspace=np.linspace(0,1023,1024)
deltaT=1/100
freqspace=np.fft.fftfreq(1024, deltaT)
LinearData=deltaT*thelinspace
SineData=np.sin(2*math.pi*thelinspace/(1024*deltaT))

freqarr=np.zeros(1024)
for i in range(0,1023):
    freqarr[i]=(i-1024/2+1)/(1024*deltaT)

plt.plot(thelinspace, SineData)
plt.xlabel("t")
plt.ylabel("f(t)")
plt.title("Sine noise")
plt.savefig("1a_Sinenoise.png")
plt.clf()


SineDataFFT=np.fft.fft(SineData)
plt.plot(freqarr, SineDataFFT.real)
plt.xlabel("f")
plt.ylabel("F(f(t))")
plt.title("Sine wave FFT")
plt.savefig("1b_SineFFT.png")
plt.clf()

print("Sums:")
print(np.sum(np.power(np.absolute(SineDataFFT),2)/(1024)))
print(np.sum(np.power(SineData,2)))

#How to check normalization? Maybe calculate power.
#Part B
gaussfreqspace=np.fft.rfftfreq(1024,1)

gaussnoise=np.random.normal(0,1,1024)
plt.plot(thelinspace, gaussnoise)
plt.xlabel("t")
plt.ylabel("Signal")
plt.title("Gaussian time-domain noise")
plt.savefig("2a_Gausnoise.png")
plt.clf()

#Nyquist frequency is half the highest frequency... so we take the FFT and find the highest freq.

gaussnoisefft=np.fft.rfft(gaussnoise)
gaussnoisefftcopy=list(gaussnoisefft)
maxfreqarg=np.argmax(gaussfreqspace)
maxfreq=gaussfreqspace[maxfreqarg]



plt.plot(gaussfreqspace, gaussnoisefft.real)
plt.xlabel("f")
plt.ylabel("F(f(t))")
plt.title("Gaussian noise FFT")
plt.savefig("2b_gaussnosie_fft.png")
plt.clf()

for i,entry in enumerate(gaussfreqspace):
    gaussnoisefft[i]=gaussnoisefft[i]*1/(1+(10*entry/maxfreq)**(2*4))

plt.plot(gaussfreqspace, 1/(1+(10*gaussfreqspace/maxfreq)**(2*4)))
plt.xlabel("Frequency")
plt.ylabel("Filter transmittance")
plt.title("Butterworth filter transmittance")
plt.savefig("2b_filt_trans.png")
plt.clf()

plt.plot(gaussfreqspace, np.absolute(gaussnoisefft))
plt.xlabel("f")
plt.ylabel("F(f(t))")
plt.title("Gaussian noise FFT, Butterworth filtered")
plt.savefig("2b_gaussnosie_fftfilt.png")
plt.clf()      

filteredgaussnoise=np.fft.irfft(gaussnoisefft)
plt.plot(thelinspace, filteredgaussnoise)
plt.xlabel("t")
plt.ylabel("f(t)")
plt.title("Gaussian noise, Butterworth filtered")
plt.savefig("2b_gaussnoise_filt.png")
plt.clf()

autocorr=np.correlate(filteredgaussnoise, filteredgaussnoise, 'full')
plt.plot(np.linspace(-1024, 1024, 2047), autocorr)
plt.ylabel("Autocorrelation")
plt.xlabel("Offset")
plt.title("Filtered autocorrelation")
plt.savefig("2c_autocorr.png")
plt.clf()

#Part C:


for i in range(0,len(gaussnoisefft)):
    if i != 0 and i != len(gaussnoisefft)-1:
        gaussnoisefft[i]=gaussnoisefft[i]*2




psd=(1/(1*1024))*np.power(np.absolute(gaussnoisefft),2)

lsd = np.sqrt(psd)

plt.plot(gaussfreqspace,lsd)
plt.xlabel("Frequency")
plt.ylabel("LSD")
plt.title("3a:  LSD")
plt.savefig("3a_LSD")
plt.clf()


#RMS of signal is equal to its mean power

rms=math.sqrt(np.mean(np.power(np.absolute(filteredgaussnoise),2)))
print(rms)
print(math.sqrt(np.sum(np.absolute(psd))/(1024*2)))
        
plt.plot(gaussfreqspace, 10*np.log10(psd/50))
plt.xlabel("Frequency")
plt.ylabel("PSD")
plt.title("3c: PSD")
plt.savefig("3C_PSD")
plt.clf()


