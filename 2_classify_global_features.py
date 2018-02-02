
# coding: utf-8

# ### Global MFCC and classification

# Import the libraries

# In[ ]:

import numpy as np
import collections
from sklearn.svm import SVC
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split


# Functions

# In[ ]:

# Support Vector Machine for classification
def classifier(features_train, features_test, labels_train, labels_test):
    
    clf = SVC(C=2**0, cache_size=300, class_weight=None, coef0=0.0,
              decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
              probability=False, random_state=None, shrinking=True,
              tol=0.001, verbose=False)
    
    clf.fit(features_train, labels_train)
    
    score_test = clf.score(features_test, labels_test)
    
    return score_test


# In[ ]:

# Extract global features
def globalFeatures(features):
    
    return features.mean(0)


# In[ ]:

# Classifier and model evaluation
def main(dataset, labels):
    
    # Split data into two set (training set, test set)
    features_train, features_test, labels_train, labels_test = train_test_split(
        dataset, labels, test_size=0.3)

    print('Train set shape:', np.shape(features_train))
    print('Test set shape:', np.shape(features_test))

    counter = collections.Counter(labels_train)
    counter = dict(counter)

    print('Distribution labels (train set):', counter)

    counter=collections.Counter(labels_test)
    counter = dict(counter)

    print('Distribution labels (test set):', counter)

    score = classifier(features_train, features_test, labels_train, labels_test)

    print('Done!', 'Score:', score)


# Process

# In[ ]:

# Read data file
data = joblib.load('mfcc_features.gz')


# In[ ]:

# Extract new features

mfcc_data = list()
mfcc_cmvn_data = list()
mfcc_feature_cube_data = list()
labels_data = list()

for key in data:
    
    mfcc = data[key]['mfcc']
    mfcc_cmvn = data[key]['mfcc_cmvn']
    mfcc_feature_cube = data[key]['mfcc_feature_cube']
    mfcc_feature_cube = mfcc_feature_cube.reshape((len(mfcc_feature_cube), 39))
    
    label = data[key]['label']    
    
    mfcc_global = globalFeatures(mfcc)
    
    mfcc_cmvn_global = globalFeatures(mfcc_cmvn)
    
    mfcc_feature_global = globalFeatures(mfcc_feature_cube)
    
    
    mfcc_data.append(mfcc_global)    
    
    mfcc_cmvn_data.append(mfcc_cmvn_global)    
    
    mfcc_feature_cube_data.append(mfcc_feature_global)
    
    
    labels_data.append(label)


# In[ ]:

# Process for mfcc
main(mfcc_data, labels_data)

print('\n')

# Process for mfcc (mean + variance normalized)
main(mfcc_cmvn_data, labels_data)

print('\n')

# Process for mfcc (cube)
main(mfcc_feature_cube_data, labels_data)


# In[ ]:



