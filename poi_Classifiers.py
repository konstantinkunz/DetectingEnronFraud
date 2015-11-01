#!/usr/bin/python

""" This module implements the classifiers for the enron poi identification
    All functions return the pipeline of the classifier and its parameters
"""

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import numpy as np


def getDecisionTree(gridSearch=True):
    """ Returns the parameters and pipeline for a decision tree.
    """
    pipe = Pipeline(steps=[('select', SelectKBest(score_func=f_classif)),
                           ('pca', PCA()),
                           ('classifier', DecisionTreeClassifier(max_features=None))])
    if gridSearch == True:
        params = {'select__k':range(3,15,1),
                  'pca__n_components':np.linspace(0.1,0.9,9),
                  'pca__whiten':[False, True],
                  'classifier__min_samples_split':range(5,20,2),
                  'classifier__max_depth':range(5,20,2),
                  'classifier__class_weight':['auto',None]}
    else:
        params = {'select__k':[14],
                  'pca__n_components':[0.5],
                  'pca__whiten':[True],
                  'classifier__min_samples_split':[8],
                  'classifier__max_depth':[None],
                  'classifier__class_weight':[None]}
        
    return pipe, params

def getSVM(gridSearch = True):
    """ Return the parameters and pipeline for a SVM classifier
    """
    pipe = Pipeline(steps=[('scaler', MinMaxScaler()),
                           ('select', SelectKBest(score_func=f_classif)),
                           ('pca', PCA()),
                           ('classifier', SVC())])

    if gridSearch == True:
        params = {'select__k':range(3,15,1),
                  'pca__n_components':np.linspace(0.1,0.9,9),
                  'pca__whiten':[True],
                  'classifier__C':[0.1,1,5,10,50,100,500,1000],
                  'classifier__kernel':['rbf'],
                  'classifier__class_weight':[None]}
    else:
        params = {'select__k':[9],
                  'pca__n_components':[0.8],
                  'pca__whiten':[True],
                  'classifier__C':[500],
                  'classifier__kernel':['rbf'],
                  'classifier__class_weight':[None]}

    return pipe, params

def getkNeighbors(gridSearch = True):
    """ Returns the parameters and pipeline for a kNearestNeighbors classifier
    """
    pipe = Pipeline(steps=[('scaler', MinMaxScaler()),
                           ('select', SelectKBest(score_func=f_classif)),
                           ('pca', PCA()),
                           ('classifier', KNeighborsClassifier())])

    if gridSearch == True:
        params = {'select__k':range(3,15,1),
                  'pca__n_components':np.linspace(0.1,0.9,9),
                  'pca__whiten':[True],
                  'classifier__n_neighbors':range(1,20,1),
                  'classifier__weights':['uniform','distance']}
    else:
        params = {'select__k':[3],
                  'pca__n_components':[0.7],
                  'pca__whiten':[True],
                  'classifier__n_neighbors':[3],
                  'classifier__weights':['distance']}

    return pipe, params

