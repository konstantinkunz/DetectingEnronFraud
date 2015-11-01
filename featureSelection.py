#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

def printFeatureRank(selectionAlg, importances, features):
    indices = np.argsort(importances)[::-1]
    print ""
    print  selectionAlg + " feature scores:"
    for i in indices:
        ind = int(i)
        print features[ind] + ": " + str(importances[ind])

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi',
                 'salary','to_messages','deferral_payments',
                 'total_payments','exercised_stock_options','bonus',
                 'restricted_stock','shared_receipt_with_poi',
                 'restricted_stock_deferred','total_stock_value','expenses',
                 'loan_advances','from_messages','other',
                 'from_this_person_to_poi','director_fees',
                 'deferred_income','long_term_incentive',
                 'from_poi_to_this_person']

# You will need to use more features

### Load the dictionary containing the dataset
data_dict = pickle.load(open("final_project_dataset.pkl", "r") )

########################################################################
### Task 2: Remove outliers

### Take out the TOTAL data row from the data set
data_dict.pop('TOTAL',0)

### Take out the Travel Agency data row from the data set
data_dict.pop('THE TRAVEL AGENCY IN THE PARK',0)

### Fix the left shifted data of BHATNAGAR SANJAY by shifting it all back
## these are the features that need to be fixed in the order they occur
featuresFix = ['other', 'expenses', 'director_fees', 'total_payments',
                   'exercised_stock_options', 'restricted_stock',
                   'restricted_stock_deferred', 'total_stock_value']
## go from right to left as to not delete any values
## copy x to x+1 and count down x to the first element in the feature list
k = len(featuresFix)-1
for i in  range(0, k):
    data_dict['BHATNAGAR SANJAY'][featuresFix[k-i]] = data_dict['BHATNAGAR SANJAY'][featuresFix[k-i-1]]
## the first feature 'other' is actually zero and has to be set manually
data_dict['BHATNAGAR SANJAY']['other'] = 0
########################################################################

### Extract features and labels from dataset for local testing
my_dataset = data_dict
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)


########################################################################
### Feature selection
import numpy as np

### selectKBest
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

clf = SelectKBest(k= 'all')
features_new = clf.fit_transform(features, labels)
importances = clf.scores_
indices = np.argsort(importances)[::-1]

printFeatureRank('selectKBest', importances, features_list[1:])


### DecisionTree feature selection
from sklearn.ensemble import ExtraTreesClassifier
clf = ExtraTreesClassifier(n_estimators=1000)
features_new = clf.fit(features, labels).transform(features)
importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]

printFeatureRank('DecisionTree', importances, features_list[1:])
