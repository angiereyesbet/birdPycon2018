
# coding: utf-8

# ### Extract mfcc features and save data

# Import the libraries


import os
import speechpy
import numpy as np
import scipy.io.wavfile as wav
from sklearn.externals import joblib
from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import fbank
from python_speech_features import logfbank


# Functions


# function for extract mfcc features
def extractFeatures(audio_path):
    
    mfcc = None
    mfcc_cmvn = None
    mfcc_feature_cube = None
    
    result = True
    
    # verify if exist audio file
    if os.path.isfile(audio_path):      
        
        fs, signal = wav.read(audio_path)
        
        try:
            
            # mfcc features
            mfcc = speechpy.feature.mfcc(signal, sampling_frequency=fs, frame_length=0.020, frame_stride=0.01,
                                         num_filters=40, fft_length=512, low_frequency=0, high_frequency=None)
            
            # mfcc(mean + variance normalized) features
            mfcc_cmvn = speechpy.processing.cmvnw(mfcc, win_size=301, variance_normalization=True)
            
            # mfcc feature cube
            mfcc_feature_cube = speechpy.feature.extract_derivative_feature(mfcc)
            
        except:
            
            result = False
            
    return result, mfcc, mfcc_cmvn, mfcc_feature_cube


# Process


# read metadata file
data = joblib.load('data.gz')



# Main process
item = 1
species = {}
data_features = {}

for key in data:

    audio_path = data[key]['silDir']

    class_id = data[key]['ClassId']

    # process for specie ID
    if class_id in species:
        
        specie_id = species[class_id]
        
    else:
        
        item+=1
        specie_id = item
        species[class_id] = item

    # process for extract features
    result, mfcc, mfcc_cmvn, mfcc_feature_cube = extractFeatures(audio_path)
    
    if result is True:
        
        data_tmp = {}
        
        data_tmp['mfcc'] = mfcc
        data_tmp['mfcc_cmvn'] = mfcc_cmvn
        data_tmp['mfcc_feature_cube'] = mfcc_feature_cube
        data_tmp['label'] = specie_id
        
        data_features[key] = data_tmp



# Dump data with compression
print('Dump data...')
joblib.dump(data_features, 'mfcc_features.gz', compress=('gzip', 3)) 
print("Files for process:", len(data_features))
print("Done.")

