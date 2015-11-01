#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from poi_dataMunger import removeElements
from poi_dataMunger import fixDataShift
from poi_dataMunger import fixNanToZero
from poi_featureCreator import createSumFeature
from poi_featureCreator import createDifferenceFeature
from poi_featureCreator import createRatioFeature
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn import grid_search
from poi_dataMunger import sortSelection
from poi_Classifiers import getDecisionTree
from poi_Classifiers import getSVM
from poi_Classifiers import getkNeighbors

########################################################################
### Helper functions

    
########################################################################

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

financial_features = ['salary',
                     'bonus',
                     'long_term_incentive',
                     'deferred_income',
                     'deferral_payments',
                     'loan_advances',
                     'other',
                     'expenses',
                     'director_fees',
                     'total_payments',
                     'exercised_stock_options',
                     'restricted_stock',
                     'restricted_stock_deferred',
                     'total_stock_value'
                      ]

                 
email_features = ['to_messages',
                  'from_poi_to_this_person',
                  'shared_receipt_with_poi',
                  'from_messages',
                  'from_this_person_to_poi']
              
features_list = ['poi'] + financial_features + email_features

# You will need to use more features

### Load the dictionary containing the dataset
data_dict = pickle.load(open("final_project_dataset.pkl", "r") )

########################################################################

### Task 2.1: Remove outliers
keys = ['TOTAL','THE TRAVEL AGENCY IN THE PARK','LOCKHART EUGENE E']
data_dict = removeElements(data_dict, keys)

### Task 2.2: Fix the data
featuresFix = ['other', 'expenses', 'director_fees', 'total_payments',
                   'exercised_stock_options', 'restricted_stock',
                   'restricted_stock_deferred', 'total_stock_value']

key = 'BHATNAGAR SANJAY'
data_dict = fixDataShift(data_dict, key, featuresFix, value=0.0, shift = 'right')

key = 'BELFER ROBERT'
featuresFix = financial_features
data_dict = fixDataShift(data_dict,key,featuresFix,value=0.0,shift='left')

### Task 2.3 Remove all NaN for further processing
data_dict = fixNanToZero(data_dict)

########################################################################

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict


### create the total poi interaction of the people
features = ['from_poi_to_this_person','shared_receipt_with_poi','from_this_person_to_poi']
newFeature = 'total_poi_interact'
my_dataset = createSumFeature(my_dataset, features, newFeature)
features_list.append(newFeature)

### create the bonus salary ratio
features = ['bonus','salary']
newFeature = 'bonus_salary_ratio'
my_dataset = createRatioFeature(my_dataset, features, newFeature)
features_list.append(newFeature)

### create the total incentives feature
features = ['bonus','long_term_incentive']
newFeature = 'total_incentives'
my_dataset = createSumFeature(my_dataset, features, newFeature)
features_list.append(newFeature)

### create total compensation feature
features = ['total_payments','total_stock_value']
newFeature = 'total_compensation'
my_dataset = createSumFeature(my_dataset, features, newFeature)
features_list.append(newFeature)

### create to/from poi email percentage of total to/from mails
features = ['from_poi_to_this_person','to_messages']
newFeature = 'from_poi_ratio'
my_dataset = createRatioFeature(my_dataset, features, newFeature)
features_list.append(newFeature)
features = ['from_this_person_to_poi','from_messages']
newFeature = 'to_poi_ratio'
my_dataset = createRatioFeature(my_dataset, features, newFeature)
features_list.append(newFeature)

### create the from non-poi to this person feature
features = ['to_messages','from_poi_to_this_person']
newFeature = 'from_non_poi_to_this_person'
my_dataset = createDifferenceFeature(my_dataset, features, newFeature)
features_list.append(newFeature)

### create the total payments, stock value ratio
features = ['total_payments','total_stock_value']
newFeature = 'payments_stock_ratio'
my_dataset = createRatioFeature(my_dataset, features, newFeature)
features_list.append(newFeature)

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)


########################################################################
### Feature selection

""" Lets check the features for importance via automated selection algorithms
    such as SelectKBest.
    This is just a stage to understand the features a bit better.
"""

"""
### scale the features in order to have comparable variances
minMax = MinMaxScaler()
features_scaled = minMax.fit_transform(features)

### Univariate feature selection by SelectKBest
selector = SelectKBest(f_classif, k=len(features_list)-1)
features_new = selector.fit_transform(features_scaled, labels)

### sort the results
sortedSelection = sortSelection(selector.scores_)

### show the results
print "The weights of the features with SelectKBest are:"
i=0
for feature in sortedSelection:
    print "{0}: {1}".format(features_list[feature[1]+1], feature[0])
"""

"""
########################################################################
### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
target_names = ['not-poi', 'poi']

### create a basic classifier GaussianNB
clf = GaussianNB()
clf.fit(features, labels)
### test the classifier
labels_pred = clf.predict(features)
##print "The results of a basic GaussianNB classifier are:"
##print(classification_report(labels, labels_pred, target_names=target_names))


### create a SVM classifier with scaled data
features_scaled = preprocessing.scale(features)
clf = svm.SVC()
clf.fit(features_scaled, labels)
### test the classifier
labels_pred = clf.predict(features_scaled)
##print "The results of a basic SVM classifier are:"
##print(classification_report(labels, labels_pred, target_names=target_names))


### create a random forest classifier
clf = RandomForestClassifier(n_estimators=10)
clf = clf.fit(features, labels)
### test the classifier
labels_pred = clf.predict(features)
##print "The results of the basic forest of decision trees classifier is:"
##print(classification_report(labels, labels_pred, target_names=target_names))
"""
#########################################################################
### Task 4: Try a variety of classifiers
### Task 5: Tune your classifier
### Task 6: Validation

""" All these tasks will be performed in this step with a pipeline
    for different machinelearning algorithms
"""

### achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

### stratified shuffle splits
sk_fold = StratifiedShuffleSplit(labels, n_iter=1000, test_size=0.1)

### create the classifiers and compare them
pipelines = []

""" Uncomment the classifier that you want to analyse or the
    you want to compare.
"""

### Decision Tree

pipe, params = getDecisionTree(gridSearch = False)
pipelines.append((pipe, params))


### SVM
"""
pipe, params = getSVM(gridSearch = False)
pipelines.append((pipe, params))
"""

### kNearestNeighbors
"""
pipe, params = getkNeighbors(gridSearch = False)
pipelines.append((pipe, params))
"""

## gridsearch on the classifier parameters
scores = ['f1', 'recall', 'precision']
for score in scores:
    print "start gridsearch"
    for pipe, params in pipelines:
        clf = grid_search.GridSearchCV(estimator=pipe , param_grid=params,
                                   cv=sk_fold, scoring=score, verbose=0)
        clf.fit(features, labels)
        print "{0} score: {1}".format(score, clf.best_score_)
        print "Best estimator:{0}".format(clf.best_estimator_)
    print "end gridsearch"

## print the selected features and their score
"""
sortedSelection = sortSelection(clf.best_estimator_.named_steps['select'].scores_)
print "The weights of the features with SelectKBest are:"
i=0
for feature in sortedSelection:
    print "{0}: {1}".format(features_list[feature[1]+1], feature[0])
"""

### create classifier for tester.py
clf = clf.best_estimator_

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
