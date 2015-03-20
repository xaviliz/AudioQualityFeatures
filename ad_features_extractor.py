# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 01:20:16 2015

@author: xavilizarraga

This function was implemented to extract some features recursively
from a dataset. It is inspired in the streaming_extractor.py from Essentia Library.
So, extract_batch.py will call this function to compute the descriptors.
"""

#! /usr/bin/env python

import sys, os
from os.path import join
import numpy
import pylab

import essentia
import essentia.standard as standard
import essentia.streaming as streaming
from essentia import Pool, INFO

from metadata     import readMetadata, getAnalysisMetadata

# maybe I can do the same with my script, create another function like replaygain and call it in the 
# process of this function. It will be a good alternative

#import replaygain
#import lowlevel
#import midlevel
#import highlevel
#import panning
#import segmentation


essentia_usage = "usage: \'essentia_extractor [options] config_file input_soundfile output_results\'"

# global defines:
analysisSampleRate = 44100.0

def parse_args():

    essentia_version = '%s\n'\
    'python version: %s\n'\
    'numpy version: %s' % (essentia.__version__,       # full version
                           sys.version.split()[0],     # python major version
                           numpy.__version__)          # numpy version

    from optparse import OptionParser
    parser = OptionParser(usage=essentia_usage, version=essentia_version)

    parser.add_option("-v","--verbose",
      action="store_true", dest="verbose", default=False,
      help="verbose mode")

    parser.add_option("-s","--segmentation",
      action="store_true", dest="segmentation", default=False,
      help="do segmentation")

    parser.add_option("-p","--profile",
      action="store", type="string", dest="profile", default="music",
      help="computation mode: 'music', 'sfx' or 'broadcast'")

    parser.add_option("--start",
      action="store", dest="startTime", default="0.0",
      help="time in seconds from which the audio is computed")

    parser.add_option("--end",
      action="store", dest="endTime", default="1.0e6",
      help="time in seconds till which the audio is computed, 'end' means no time limit")

    parser.add_option("--svmpath",
      action="store", dest="svmpath", default=join('..', 'svm_models'),
      help="path to svm models")

    (options, args) = parser.parse_args()

    return options, args
    
def computeAggregation(pool, segments_namespace=''):
    stats = ['mean','median','var','min','max','dmean','dmean2','dvar','dvar2' ]

    exceptions={'lowlevel.mfcc' : ['mean', 'cov', 'icov']}
    for namespace in segments_namespace:
        exceptions[namespace+'.lowlevel.mfcc']=['mean', 'cov', 'icov']

    if segments_namespace: exceptions['segmentation.timestamps']=['copy']
    return standard.PoolAggregator(defaultStats=stats,
                                   exceptions=exceptions)(pool)
                                                                      
def addSVMDescriptors(pool, pathToSvmModels):
    #svmModels = [] # leave this empty if you don't have any SVM models
    svmModels = ['BAL', 'CUL', 'GDO', 'GRO', 'GTZ', 'PS', 'VI','MAC', 'MAG', 'MEL', 'MHA', 'MPA', 'MRE', 'MSA']

    for model in svmModels:
        modelFilename = join(pathToSvmModels, model+'.model')
        svm = standard.SvmClassifier(model=modelFilename)(pool)
        pool.merge(svm)


def computeADLowLevel(input_file, pool, startTime, endTime, namespace=''):
    #sampleRate, downmix = getAnalysisMetadata(pool)
    sampleRate = 44100
    loader = standard.MonoLoader(filename = input_file,sampleRate = sampleRate)
    audio = loader()
    # Normalize audio data, because vinyl versions were recorded from a mixer
    audio = audio/max(audio)
    #lowlevel.compute(loader.audio, loader.audio, pool, startTime, endTime, namespace)
    # Initialize Essentia Objects
    w = standard.Windowing(type = 'hann')
    spectrum = standard.Spectrum()
    #pool = Pool()
    logattacktime = standard.LogAttackTime()
    #centroid = Centroid()
    hfc = standard.HFC()
    energy = standard.Energy()
    lowp = standard.LowPass(cutoffFrequency = 80) # 100 Hz - Low pass filter
    crest = standard.Crest()
    #envelope = streaming.Envelope(attackTime=0.003/5,releaseTime=200./5)
    #flatnessDB = streaming.FlatnessDB()
    rolloff = standard.RollOff()
    #distributionshape = streaming.DistributionShape()
    #zerocrossing = streaming.ZeroCrossingRate()
    highp = standard.HighPass(cutoffFrequency = 15000) # 100 hz
    #centralmoments = streaming.CentralMoments()
    strongdecay = standard.StrongDecay()
    flatnesssfx = standard.FlatnessSFX()
    derivative = standard.Derivative()
    entropy = standard.Entropy()
    dcremoval = standard.DCRemoval(cutoffFrequency=20)
    energybandratiohf = standard.EnergyBandRatio(sampleRate = sampleRate, startFrequency= 10000, stopFrequency = 22050)
    energybandratiolf = standard.EnergyBandRatio(sampleRate = sampleRate, startFrequency= 0, stopFrequency = 80)

    # Bag Of Features for cd version
    for frame in standard.FrameGenerator(audio, frameSize = 2048, hopSize = 128):
        frame = dcremoval(frame)
        pool.add('lowlevel.logattacktime', logattacktime(frame))
        pool.add('lowlevel.lfenergy', energy(lowp(w(frame))))
        pool.add('lowlevel.hfenergy', energy(highp(w(frame))))
        pool.add('lowlevel.crestfactor',crest(abs(w(frame))))
        pool.add('lowlevel.rolloff', rolloff((w(frame))))
        pool.add('lowlevel.derivative', numpy.std(abs((derivative(w(frame))))-numpy.median(abs(derivative(w(frame))))))
        pool.add('lowlevel.spec_logattacktime', logattacktime(spectrum(w(frame))))
        #pool.add('lowlevelcd.centroid', (sampleRate/2.)*(centroid(spectrum(derivative(w(frame))))))
        pool.add('lowlevel.hfc', hfc(spectrum(w(frame)))) 
        pool.add('lowlevel.spectralcrest', crest(spectrum(w(frame))))
        pool.add('lowlevel.strongdecay', strongdecay(spectrum(w(frame))))
        pool.add('lowlevel.flatnesssfx', flatnesssfx(spectrum(w(frame))))
        pool.add('lowlevel.hfbandratio', energybandratiohf(spectrum(w(frame))))    
        pool.add('lowlevel.lfbandratio', energybandratiolf(spectrum(w(frame))))
        pool.add('lowlevel.entropy', entropy(spectrum(w(frame))))

"""
def computeMidLevel(input_file, pool, startTime, endTime, namespace=''):
    rgain, sampleRate, downmix = getAnalysisMetadata(pool)
    loader = streaming.EqloudLoader(filename = input_file,
                                    sampleRate = sampleRate,
                                    startTime = startTime,
                                    endTime = endTime,
                                    replayGain = rgain,
                                    downmix = downmix)
    midlevel.compute(loader.audio, pool, startTime, endTime, namespace)
    essentia.run(loader)
"""

if __name__ == '__main__':

    opt, args = parse_args()

    if len(args) != 2: #3:
        print "Incorrect number of arguments\n", essentia_usage
        sys.exit(1)

    #profile = args[0]
    input_file = args[0]
    output_file = args[1]

    pool = Pool()
    startTime = float(opt.startTime)
    endTime = float(opt.endTime)

    # compute descriptors

    #readMetadata(input_file, pool)
    #INFO('Process step 1: Replay Gain')
    #replaygain.compute(input_file, pool, startTime, endTime)

    segments_namespace=[]
    INFO('Process step 1: Low Level Features')
    computeADLowLevel(input_file, pool, startTime, endTime)
    #INFO('Process step 2: Mid Level')
    #computeMidLevel(input_file, pool, startTime, endTime)
    #INFO('Process step 3: High Level')
    #highlevel.compute(pool)

    # compute statistics
    INFO('Process step 4: Aggregation')
    stats = computeAggregation(pool, segments_namespace)

    # output results to file
    INFO('writing results to ' + output_file)
    standard.YamlOutput(filename=output_file)(stats)

