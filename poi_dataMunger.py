#!/usr/bin/python
from itertools import izip
from itertools import tee

""" The module preprocess the enron data so that it is ready for
    machine learning
"""

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return izip(a, b)

def removeElements(data_dict, keys):
    """removes the specified keys from the data_dict dictionary
    """
    for key in keys:
        data_dict.pop(key,0)
    return data_dict

def fixDataShift(data_dict, key, features, value, shift):
    """ shifts features values in data_dict one position left according to
        order of features
        data_dict   - the data to be shifted
        key         - the data point affected
        features    - the features to shift along
        value       - the value of the first feature
    """
    ### go through all elements and copy them from the source to the target
    if shift == 'right':
        first = features[0]
        features = reversed(features)
    elif shift == 'left':
        first = features[len(features)-1]
        features = features
    for featureTarget, featureSource in pairwise(features):
        data_dict[key][featureTarget] = data_dict[key][featureSource]
    ### this is the first feature and has to be set manually
    data_dict[key][first] = 0.0
    return data_dict

def fixNanToZero(data_dict):
    """ transformed all NaN values in the data_dict to 0.0
    """
    for person in data_dict.keys():
        for feature in data_dict[person].keys():
            if (data_dict[person][feature] == 'NaN'):
                data_dict[person][feature] = 0.0
    return data_dict

def sortSelection(selection):
    """ sorts an array of selectKBest scores
    """
    sortedSelection = []
    i = 0
    for score in selection:
        sortedSelection.append((score,i))
        i += 1
    sortedSelection = sorted(sortedSelection, key=lambda tup:tup[0], reverse=True)
    return sortedSelection
