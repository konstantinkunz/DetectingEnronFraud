#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit

### Load the dictionary containing the dataset
data_dict = pickle.load(open("final_project_dataset.pkl", "r"))

### Task 1: Explore the data
keys = data_dict.keys()
features = data_dict[keys[0]].keys()

### How does an element look like?
print "How does an element look like?"
print keys[0], data_dict[keys[0]]

### How many persons (elements) are there in the data set?
print "How many persons (elements) are there in the data set?"
print len(data_dict)

### How many features are there?
print "How many features are there?"
print len(data_dict[keys[10]])

### How many of those are persons of interest?
print "How many of those are persons of interest?"
i = 0
for key in keys:
    if data_dict[key]['poi'] == True:
        i += 1
print i

### How many valid data points are there?
print "How many valid data points are there?"
i = 0
for key in keys:
    for feature in features:
                     if data_dict[key][feature] != 'NaN':
                         i += 1
print i

### How many valid data points are from POIs?
print "How many valid data points are from POIs?"
i = 0
for key in keys:
    for feature in features:
                     if (data_dict[key][feature] != 'NaN' and
                         data_dict[key]['poi'] == True):
                         i += 1
print i

### What is the distribution of NaN value data over the features?
print "What is the distribution of NaN value data over the features?"
NonValidData = {}
for feature in features:
    NonValidData[feature] = 0
    for key in keys:
        if data_dict[key][feature] == 'NaN':
            NonValidData[feature] += 1
print NonValidData

### How many people dont send and receive any messages?
print "How many people dont send or receive any messages?"
feature = ('to_messages', 'from_messages')
i = 0
for key in keys:
        if (data_dict[key][feature[0]] == 'NaN' and
            data_dict[key][feature[1]] == 'NaN'):
            i += 1
print i

### How many of these are POIs?
print "How many of these are POIs?"
feature = ('to_messages', 'from_messages', 'poi')
i = 0
for key in keys:
        if (data_dict[key][feature[0]] == 'NaN' and
            data_dict[key][feature[1]] == 'NaN'):
            if data_dict[key][feature[2]] == True:
                i += 1
print i

### What are the top 5 features with most values?
print "What are the top 5 features with most values?"
import operator
sorted_NonValidData = sorted(NonValidData.items(), key=operator.itemgetter(1))
print sorted_NonValidData[1:6]

### What features that are non-zero are most shared with POIs what are most shared with non POIs
print "What features do POIs share the most"
ValidData = {}
for feature in features:
    ValidData[feature] = 0
    for key in keys:
        if (data_dict[key][feature] != 'NaN' and
            data_dict[key]['poi'] == True):
            ValidData[feature] += 1
    ValidData[feature] = ValidData[feature]/18.0        
sorted_ValidData = sorted(ValidData.items(), key=operator.itemgetter(1))
print sorted_ValidData

### What features that are non-zero are most shared with non-POIs what are most shared with non POIs
print "What features do POIs share the most"
ValidData = {}
for feature in features:
    ValidData[feature] = 0
    for key in keys:
        if (data_dict[key][feature] != 'NaN' and
            data_dict[key]['poi'] == False):
            ValidData[feature] += 1
    ValidData[feature] = ValidData[feature]/128.0       
sorted_ValidData = sorted(ValidData.items(), key=operator.itemgetter(1))
print sorted_ValidData
