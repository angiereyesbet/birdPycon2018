
# coding: utf-8

# ### Bag of MFCC-based word and classification

# Import the libraries

# In[ ]:

import numpy as np
import collections
from sklearn.svm import SVC
from sklearn.externals import joblib
from scipy.spatial.distance import cdist
from sklearn.cluster import MiniBatchKMeans
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

# Normalize data
def normalize(data):
    
    std_dev = np.std(data, axis=0)
    zero_std_mask = std_dev == 0
    
    if zero_std_mask.any():
    
        std_dev[zero_std_mask] = 1.0
        
    result = data / std_dev
    
    return result, std_dev


# In[ ]:

# K-Means clustering
def k_means(data, k_guess):
    
    batch_size = 45 
    
    mbk = MiniBatchKMeans(init='k-means++', n_clusters=k_guess, batch_size=batch_size,
                          n_init=10, max_no_improvement=10, verbose=0)
    
    codebook = mbk.fit(data)
    
    return codebook.cluster_centers_


# In[ ]:

# Create histogram
def histogram(std_dev, features, codebook, index):    
    
    histogram_= [0] * index
    
    distance = cdist(features,  codebook, 'euclidean')
    
    short = ((np.argsort(distance)).transpose()[0]).tolist()
    
    counter = dict(collections.Counter(short))
    
    for key in counter:        
        
        histogram_[key] = int(counter[key])
          
    return histogram_


# In[ ]:

# Bag of features (visual words)
def bagOfWords(features, n_clusters):
    
    features, std_dev = normalize(mfcc)
    
    codebook = k_means(features, n_clusters)
    
    histogram_ = histogram(std_dev, features, codebook, n_clusters)
    
    return histogram_


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

n_clusters = 100

for key in data:
    
    mfcc = data[key]['mfcc']
    mfcc_cmvn = data[key]['mfcc_cmvn']
    mfcc_feature_cube = data[key]['mfcc_feature_cube']
    mfcc_feature_cube = mfcc_feature_cube.reshape((len(mfcc_feature_cube), 39))
    
    label = data[key]['label'] 
    
    n_clusterster = int ( len(mfcc) / 2) 
    
    mfcc_histogram = bagOfWords(mfcc, n_clusters)
    
    mfcc_cmvn_histogram = bagOfWords(mfcc_cmvn, n_clusters)
    
    mfcc_feature_cube_histogram = bagOfWords(mfcc_feature_cube, n_clusters)
        
    
    mfcc_data.append(mfcc_histogram)    
    
    mfcc_cmvn_data.append(mfcc_cmvn_histogram)    
    
    mfcc_feature_cube_data.append(mfcc_feature_cube_histogram)
    
    
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

