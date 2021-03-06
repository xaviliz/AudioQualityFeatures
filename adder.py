# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 17:40:50 2015

@author: xavilizarraga
"""

# Script to compare vynil features with cd features

#Experiment 1

# first, we need to import our essentia module. It is aptly named 'essentia'!
import essentia
from pylab import *
from numpy import *
# as there are 2 operating modes in essentia which have the same algorithms,
# these latter are dispatched into 2 submodules:

import essentia.standard
import essentia.streaming

def play(audiofile):
    import os, sys
    # NB: this only works with linux!! mplayer crashes!
    if sys.platform == 'linux2':
        os.system('mplayer %s' % audiofile)
    if sys.platform == 'darwin':
        import subprocess
        subprocess.call(["afplay", audiofile])
    else:
        print 'Not playing audio...'        

filename1 = '../../Courses/UPF/SMC/Term 2nd/Audio and Music Analysis/Labs/Audio quality/dataset/a_d_dataset/Bonnie M/cd/1- boney-m-daddy-cool-150306_0324.wav' 
filename2 = '../../Courses/UPF/SMC/Term 2nd/Audio and Music Analysis/Labs/Audio quality/dataset/a_d_dataset/Bonnie M/vinyl/1- Daddy Cool-150306_0324.wav' 
"""
print('Playing audio excerpts...')
play(filename1)
play(filename2)
"""
namespace1 = 'lowlevelcd'
namespace2 = 'lowlevelvy'
#plot condition
pcont = 0
# spectral analysis parameters
hopSize = 128
frameSize = 2048
sampleRate = 44100

# So, first let's load audio files
# to make sure it's not a trick, let's show the original "audio" to you:

# start by instantiating the audio loader:
loader1 = essentia.standard.MonoLoader(filename = filename1)
loader2 = essentia.standard.MonoLoader(filename = filename2)
# Perform the loading:
audio1 = loader1()
audio2 = loader2()
# Normalize audio data, because vinyl versions were recorded from a mixer
audio1 = audio1/max(audio1)
audio2 = audio2/max(audio2)

# Import essentia standard algorithms
from essentia.standard import *
# Initialize Essentia Objects
w = Windowing(type = 'hann')
spectrum = Spectrum()
mfcc = MFCC()
pool = essentia.Pool()
logattacktime = LogAttackTime()
centroid = Centroid()
hfc = HFC()
energy = Energy()
lowp = LowPass(cutoffFrequency = 80) # 100 hz
crest = Crest()
envelope = Envelope(attackTime=0.003/5,releaseTime=200./5)
flatnessDB = FlatnessDB()
rolloff = RollOff()
distributionshape = DistributionShape()
zerocrossing = ZeroCrossingRate()
highp = HighPass(cutoffFrequency = 15000) # 100 hz
centralmoments = CentralMoments()
strongdecay = StrongDecay()
flatnesssfx = FlatnessSFX()
derivative = Derivative()
entropy = Entropy()
dcremoval = DCRemoval(cutoffFrequency=20)
energybandratiohf = EnergyBandRatio(sampleRate = 44100, startFrequency= 10000, stopFrequency = 22050)
energybandratiolf = EnergyBandRatio(sampleRate = 44100, startFrequency= 0, stopFrequency = 80)

# Bag Of Features for cd version
for frame in FrameGenerator(audio1, frameSize = 2048, hopSize = 128):
    frame = dcremoval(frame)
    pool.add('lowlevelcd.logattacktime', logattacktime(frame))
    pool.add('lowlevelcd.lfenergy', energy(lowp(w(frame))))
    pool.add('lowlevelcd.hfenergy', energy(highp(w(frame))))
    pool.add('lowlevelcd.crestfactor',crest(abs(w(frame))))
    pool.add('lowlevelcd.rolloff', rolloff((w(frame))))
    pool.add('lowlevelcd.derivative', std(abs((derivative(w(frame))))-median(abs(derivative(w(frame))))))
    pool.add('lowlevelcd.spec_logattacktime', logattacktime(spectrum(w(frame))))
    pool.add('lowlevelcd.centroid', (sampleRate/2.)*(centroid(spectrum(derivative(w(frame))))))
    pool.add('lowlevelcd.hfc', hfc(spectrum(w(frame)))) 
    pool.add('lowlevelcd.spectralcrest', crest(spectrum(w(frame))))
    pool.add('lowlevelcd.strongdecay', strongdecay(spectrum(w(frame))))
    pool.add('lowlevelcd.flatnesssfx', flatnesssfx(spectrum(w(frame))))
    pool.add('lowlevelcd.hfbandratio', energybandratiohf(spectrum(w(frame))))    
    pool.add('lowlevelcd.lfbandratio', energybandratiolf(spectrum(w(frame))))
    pool.add('lowlevelcd.entropy', entropy(spectrum(w(frame))))

# Bag Of Features for vinyl version
for frame in FrameGenerator(audio2, frameSize = 2048, hopSize = 128):
    frame = dcremoval(frame)
    pool.add('lowlevelvy.logattacktime', logattacktime(frame))
    pool.add('lowlevelvy.spec_logattacktime', logattacktime(spectrum(w(frame))))
    pool.add('lowlevelvy.centroid', (sampleRate/2.)*(centroid(spectrum(derivative(w(frame))))))
    pool.add('lowlevelvy.hfc', hfc(spectrum(w(frame))))
    pool.add('lowlevelvy.lfenergy', energy(lowp(w(frame))))
    pool.add('lowlevelvy.hfenergy', energy(highp(w(frame))))
    pool.add('lowlevelvy.crestfactor',crest(abs(w(frame)))) 
    pool.add('lowlevelvy.spectralcrest', crest(spectrum(w(frame))))
    pool.add('lowlevelvy.rolloff', rolloff((w(frame))))
    pool.add('lowlevelvy.strongdecay', strongdecay(spectrum(w(frame))))
    pool.add('lowlevelvy.flatnesssfx', flatnesssfx(spectrum(w(frame))))
    pool.add('lowlevelvy.derivative', std(abs((derivative(w(frame))))-median(abs(derivative(w(frame))))))
    pool.add('lowlevelvy.hfbandratio', energybandratiohf(spectrum(w(frame))))    
    pool.add('lowlevelvy.lfbandratio', energybandratiolf(spectrum(w(frame))))
    pool.add('lowlevelvy.entropy', entropy(spectrum(w(frame))))
  
nframes = float(len(audio1))/hopSize
# time-frame mapping vector
t1 = np.arange(0,float(len(audio1))/sampleRate,float(hopSize)/sampleRate)
t2 = np.arange(0,float(len(audio2))/sampleRate,float(hopSize)/sampleRate)

if (pcond != 0):
    # Drawing LLD  - HFC
    plt.figure(1, figsize=(11.5, 9))
    plt.subplot(4,1,1)
    plt.plot(np.arange(audio1.size)/float(sampleRate), audio1, 'b')
    plt.axis([0, audio1.size/float(sampleRate), min(audio1), max(audio1)])
    xlabel('Time(s)'), plt.ylabel('amplitude')
    plt.title('x (cd version.wav)')
    
    plt.subplot(4,1,2)
    plt.plot(np.arange(audio2.size)/float(sampleRate), audio2, 'b')
    plt.axis([0, audio2.size/float(sampleRate), min(audio2), max(audio2)])
    plt.ylabel('amplitude')
    xlabel('Time(s)'),plt.title('x (vinyl version.wav)')
    plt.autoscale(tight=True)
    
    plt.subplot(4,1,3)
    plot(t1,pool['lowlevelcd.hfc'].T[1:])
    axis('tight'), xlabel('Time(s)')
    title('High Frequency Content (cd version)')
    plt.autoscale(tight=True)
    
    plt.subplot(4,1,4)
    plot(t2,pool['lowlevelvy.hfc'].T[1:])
    axis('tight'), xlabel('Time(s)')
    title('High Frequency Content (vinyl version)')
    plt.autoscale(tight=True)
    plt.show()
    
    # LF energy and HFenergy
    plt.figure(2, figsize=(11.5, 9))
    plt.subplot(4,1,1)
    plot(t1,pool['lowlevelcd.lfenergy'].T[1:])
    axis('tight'), xlabel('Time(s)')
    title('LFEnergy (cd version)')
    plt.autoscale(tight=True)
    
    plt.subplot(4,1,2)
    plot(t2,pool['lowlevelvy.lfenergy'].T[1:])
    axis('tight'), xlabel('Time(s)')
    title('LFEnergy (vinyl version)')
    plt.autoscale(tight=True)
    
    plt.subplot(4,1,3)
    plot(t1,pool['lowlevelcd.hfenergy'].T[1:])
    axis('tight'), xlabel('Time(s)')
    title('HFEnergy (cd version)')
    plt.autoscale(tight=True)
    
    plt.subplot(4,1,4)
    plot(t2,pool['lowlevelvy.hfenergy'].T[1:])
    axis('tight'), xlabel('Time(s)')
    title('HFEnergy (vinyl version)')
    plt.autoscale(tight=True)
    plt.show()
    
    # log attack time - Crest Factor (Time domain)
    plt.figure(3, figsize=(11.5, 9))
    plt.subplot(4,1,1)
    plot(pool['lowlevelcd.logattacktime'].T[1:])
    axis('tight'), xlabel('Time(s)')
    title('LogattackTime (cd version)')
    plt.autoscale(tight=True)
    
    plt.subplot(4,1,2)
    plot(pool['lowlevelvy.logattacktime'].T[1:])
    axis('tight'), xlabel('Time(s)')
    title('LogAttackTime (vinyl version)')
    plt.autoscale(tight=True)
    
    plt.subplot(4,1,3)
    plot(t1,pool['lowlevelcd.crestfactor'].T[1:])
    axis('tight'), xlabel('Time(s)')
    title('Crst Factor (cd version)')
    plt.autoscale(tight=True)
    
    plt.subplot(4,1,4)
    plot(t2,pool['lowlevelvy.crestfactor'].T[1:])
    axis('tight'), xlabel('Time(s)')
    title('Crest Factor (vinyl version)')
    plt.autoscale(tight=True)
    plt.show()
    
    # Spectral centroid of derivative - Derivative (Time domain)
    plt.figure(4, figsize=(11.5, 9))
    plt.subplot(4,1,1)
    plot(t1,pool['lowlevelcd.centroid'].T[1:])
    axis('tight'), xlabel('Time(s)')
    title('Spectral dCentroid (cd version)')
    plt.autoscale(tight=True)
    
    plt.subplot(4,1,2)
    plot(t2,pool['lowlevelvy.centroid'].T[1:])
    axis('tight'), xlabel('Time(s)')
    title('Spectral dCentroid (vinyl version)')
    plt.autoscale(tight=True)
    
    plt.subplot(4,1,3)
    plot(t1,pool['lowlevelcd.derivative'].T[1:])
    axis('tight'), xlabel('Time(s)')
    title('Derivative (cd version)')
    plt.autoscale(tight=True)
    
    plt.subplot(4,1,4)
    plot(t2,pool['lowlevelvy.derivative'].T[1:])
    axis('tight'), xlabel('Time(s)')
    title('Derivative (vinyl version)')
    plt.autoscale(tight=True)
    plt.show()
    
    # Low Frequency band ratio - High Frequency band ratio
    plt.figure(5, figsize=(11.5, 9))
    plt.subplot(4,1,1)
    plot(t1,pool['lowlevelcd.lfbandratio'].T[1:])
    axis('tight'), xlabel('Time(s)')
    title('LF Band Ratio (cd version)')
    plt.autoscale(tight=True)
    
    plt.subplot(4,1,2)
    plot(t2,pool['lowlevelvy.lfbandratio'].T[1:])
    axis('tight'), xlabel('Time(s)')
    title('LF Band Ratio (vinyl version)')
    plt.autoscale(tight=True)
    
    plt.subplot(4,1,3)
    plot(t1,pool['lowlevelcd.hfbandratio'].T[1:])
    axis('tight'), xlabel('Time(s)')
    title('HF Band Ratio (cd version)')
    plt.autoscale(tight=True)
    
    plt.subplot(4,1,4)
    plot(t2,pool['lowlevelvy.hfbandratio'].T[1:])
    axis('tight'), xlabel('Time(s)')
    title('HF Band Ratio (vinyl version)')
    plt.autoscale(tight=True)
    plt.show()
    
    # Rolloff (Time Domain) - Flatness SFX
    plt.figure(6, figsize=(11.5, 9))
    plt.subplot(4,1,1)
    plot(t1,pool['lowlevelcd.rolloff'].T[1:])
    axis('tight'), xlabel('Time(s)')
    title('Roll Off (cd version)')
    plt.autoscale(tight=True)
    
    plt.subplot(4,1,2)
    plot(t2,pool['lowlevelvy.rolloff'].T[1:])
    axis('tight'), xlabel('Time(s)')
    title('Roll Off (vinyl version)')
    plt.autoscale(tight=True)
    
    plt.subplot(4,1,3)
    plot(t1,pool['lowlevelcd.flatnesssfx'].T[1:])
    axis('tight'), xlabel('Time(s)')
    title('Flatness SFX (cd version)')
    plt.autoscale(tight=True)
    
    plt.subplot(4,1,4)
    plot(t2,pool['lowlevelvy.flatnesssfx'].T[1:])
    axis('tight'), xlabel('Time(s)')
    title('Flatness SFX (vinyl version)')
    plt.autoscale(tight=True)
    plt.show()
    
    # Strong Decay (Frequency Domain) -Entropy
    plt.figure(7, figsize=(11.5, 9))
    plt.subplot(4,1,1)
    plot(t1,pool['lowlevelcd.strongdecay'].T[1:])
    axis('tight'), xlabel('Time(s)')
    title('Strong Decay (cd version)')
    plt.autoscale(tight=True)
    
    plt.subplot(4,1,2)
    plot(t2,pool['lowlevelvy.strongdecay'].T[1:])
    axis('tight'), xlabel('Time(s)')
    title('Strong Decay (vinyl version)')
    plt.autoscale(tight=True)
    
    plt.subplot(4,1,3)
    plot(t1,pool['lowlevelcd.entropy'].T[1:])
    axis('tight'), xlabel('Time(s)')
    title('Entropy (cd version)')
    plt.autoscale(tight=True)
    
    plt.subplot(4,1,4)
    plot(t2,pool['lowlevelvy.entropy'].T[1:])
    axis('tight'), xlabel('Time(s)')
    title('Entropy (vinyl version)')
    plt.autoscale(tight=True)
    plt.show()
    
    # Histogram for cd version and vynil version
    figure(8)
    plt.subplot(2,1,1)
    plt.hist(audio1,50), xlim([-1,1]),title('cd version')
    plt.subplot(2,1,2)
    plt.hist(audio2,50), xlim([-1,1]),title('vinyl version')

# Computing the kurtosis or skewness. CD version looks a normal 
# distribution whereas vinyl version is a skewed normal distribution with a short deviation or IQR.
stats = ['mean','median','var','min','max','dmean','dmean2','dvar','dvar2' ]

megalopool = PoolAggregator(defaultStats=stats)(pool)
# save to output file
YamlOutput(filename='outputFeatures.yaml')(megalopool)
pathGiven = '../../Courses/UPF/SMC/Term 2nd/Audio and Music Analysis/Labs/Audio quality/dataset/a_d_dataset/'
clusters = ['cd','vinyl']
parts = pathGiven.rstrip("/").split("/")
parentDir = "/".join(parts[:-1])
arffFilename = "/".join([parentDir, parts[-1]+".arff"])
wekafile=file(arffFilename, "w+")

relation_name = "quality"
wekafile.write("@RELATION quality\n\n")
