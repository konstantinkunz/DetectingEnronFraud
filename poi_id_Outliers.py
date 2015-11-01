#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")
import matplotlib.pyplot
from feature_format import featureFormat, targetFeatureSplit

### Load the dictionary containing the dataset
data_dict = pickle.load(open("final_project_dataset.pkl", "r"))

## Do some initialization stuff
peoples = data_dict.keys()
financial_features = ['salary', 'deferral_payments','total_payments',
                      'loan_advances', 'bonus', 'restricted_stock_deferred',
                      'deferred_income', 'total_stock_value', 'expenses',
                      'exercised_stock_options', 'other', 'long_term_incentive',
                      'restricted_stock', 'director_fees']


### helper functions
def printFeature(people, feature):
    ## this function prints out the people specific feature
    print feature + ": " + str(data_dict[people][feature])

### Take out all NaN and transform them to zeroes
for people in peoples:
    for feature in financial_features:
        if (data_dict[people][feature] == 'NaN'):
            data_dict[people][feature] = 0

### Screen for people according to financial criteria
features_analyse = ['total_stock_value', 'salary', 'poi']
"""
limits = [0, 0]
bandits = []
for people in peoples:
    ## filter for features to analyze
    if (data_dict[people][features_analyse[0]] > limits[0] and
        data_dict[people][features_analyse[1]] <= limits[1]):
        bandits.append(people)
        ## print out the filter results
        print people
        for feature in features_analyse:
            printFeature(people, feature)
"""
### Select two features to visualize
data = featureFormat(data_dict, features_analyse)


### Visualize the data
for point in data:
    dataPointX = point[0]
    dataPointY = point[1]
    if point[2] == 0:
        matplotlib.pyplot.scatter( dataPointX, dataPointY )
    else:
        matplotlib.pyplot.scatter( dataPointX, dataPointY , c = 'r')

matplotlib.pyplot.xlabel(features_analyse[0])
matplotlib.pyplot.ylabel(features_analyse[1])
matplotlib.pyplot.show()
