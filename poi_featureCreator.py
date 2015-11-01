#!/usr/bin/python

""" implements functions that create new features for the enron dataset
"""

def createSumFeature(data_dict, features, newFeature):
    """ creates new features
    """
    for people in data_dict.keys():
        s = 0.0
        for feature in features:
            s += data_dict[people][feature]
        data_dict[people][newFeature] = s
    return data_dict

def createDifferenceFeature(data_dict, features, newFeature):
    """ creates new features
    """
    for people in data_dict.keys():
        data_dict[people][newFeature] = data_dict[people][features[0]]-data_dict[people][features[1]]
    return data_dict

def createRatioFeature(data_dict, features, newFeature):
    """ creates a new feature equal to the ratio of features[0]/feature[1]
    """
    for people in data_dict.keys():
        if data_dict[people][features[1]] != 0:
            data_dict[people][newFeature] = float(data_dict[people][features[0]])/float(data_dict[people][features[1]])
        else:
            data_dict[people][newFeature] = 0.0
    return data_dict
