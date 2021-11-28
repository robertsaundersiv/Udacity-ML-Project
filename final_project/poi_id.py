import pickle
import pprint
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
# BaggingClassifier, RandomForestClassifier, GradientBoostingClassifier were all
# attempted but did nothing to improve classification of data, I settled on VotingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
# import matplotlib.pyplot as plt
import final_project.tester

pd.options.mode.chained_assignment = None  # default='warn'

# 1: Select what features you'll use.
# features_list is a list of strings, each of which is a feature name.
# The first feature must be "poi".
# broke data into 3 points of interest: email data, pay data, and stock data, combined them all into
# one with poi as the first element for project requirements
email_data = ['to_messages', 'from_messages', 'from_poi_to_this_person', 'from_this_person_to_poi',
              'shared_receipt_with_poi']
pay_data = ['salary', 'bonus', 'long_term_incentive', 'deferred_income', 'deferral_payments', 'loan_advances',
            'other', 'expenses', 'director_fees', 'total_payments']
stock_data = ['exercised_stock_options', 'restricted_stock', 'restricted_stock_deferred', 'total_stock_value']

features_list = ['poi'] + pay_data + stock_data + email_data

# Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "rb") as data_file:
    data_dict = pickle.load(data_file)

# 2: Remove outliers
# I looked through the data and found 2 rows that seemed wrong, the total row and the travel agency in the park row.
# I dropped the total row but left the travel agency in the park in case it has something of interest in an already
# sparse dataset.
# pprint.pprint(data_dict)
del data_dict['TOTAL']
# del data_dict['THE TRAVEL AGENCY IN THE PARK']
# number of points
print('Number of data points: ' + str(len(data_dict)))
# number of poi vs nonpoi
poi = 0
nonpoi = 0
for person in data_dict:
    if data_dict[person]['poi']:
        poi += 1
    else:
        nonpoi += 1
print('Number of POI:         ' + str(poi) + '\nNumber of non-POI:     ' + str(nonpoi))

# Cleaned the data as a dataframe, first replacing all NaN data with numpy nans then ensuring it is ordered in the same
# way as the features list.
df = pd.DataFrame.from_dict(data_dict, orient='index')
df = df.replace('NaN', np.nan)
df = df[features_list]
pprint.pprint(df.info())
pprint.pprint(df.describe())

# According to the documentation provided with this dataset (enron61072insiderpay.pdf) the money and stock values
# listed as nan are actually 0 and the email data is unknown. filling money and stock values with 0 and email values
# with the average grouped by poi status
df[pay_data] = df[pay_data].fillna(0)
df[stock_data] = df[stock_data].fillna(0)
df_poi = df[df['poi'] == True]
df_nonpoi = df[df['poi'] == False]
df_poi.fillna(round(df_poi[email_data].mean()), inplace=True)
df_nonpoi.fillna(round(df_nonpoi[email_data].mean()), inplace=True)
df = df_poi.append(df_nonpoi)

# found that some of the pay data was imported improperly, also checked stock data but it had 0 errors
badData = (df[df[pay_data[:-1]].sum(axis='columns') != df['total_payments']])
print(badData)
print("\nCorrecting them")
# Belfer is shifted 1 to the right
belfer = df.loc['BELFER ROBERT'][1:15].tolist()
belfer.pop(0)
belfer.append(0)
n = 0
fin_data = pay_data + stock_data
for ele in belfer:
    df.at['BELFER ROBERT', fin_data[n]] = ele
    n += 1

# Bhatnagar is shifted 1 to the left
bhatnagar = df.loc['BHATNAGAR SANJAY'][1:15].tolist()
bhatnagar.pop(-1)
bhatnagar = [0] + bhatnagar
n = 0
for ele in bhatnagar:
    df.at['BHATNAGAR SANJAY', fin_data[n]] = ele
    n += 1
# the next 2 lines were used to make sure it comes back empty
# badData = (df[df[pay_data[:-1]].sum(axis='columns') != df['total_payments']])
# print(badData)

# 3: Create new feature(s)
# Store to my_dataset for easy export below.

df['expense_income_ratio'] = df['expenses'] / (df['salary'] + df['bonus'])
df['inbound_poi_ratio'] = df['from_poi_to_this_person'] / df['to_messages']
df['outbound_poi_ratio'] = df['from_this_person_to_poi'] / df['from_messages']
df['listed_with_poi_ratio'] = df['shared_receipt_with_poi'] / df['to_messages']
df['bonus_to_salary_ratio'] = df['bonus'] / df['salary']
df['bonus_to_total_income_ratio'] = df['bonus'] / df['total_payments']
features_list.append('expense_income_ratio')
features_list.append('inbound_poi_ratio')
features_list.append('outbound_poi_ratio')
features_list.append('listed_with_poi_ratio')
features_list.append('bonus_to_salary_ratio')
features_list.append('bonus_to_total_income_ratio')
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.fillna(value=0, inplace=True)

# 4: Try a variety of classifiers
# Please name your classifier clf for easy export below.
# Note that if you want to do PCA or other multi-stage operations,
# you'll need to use Pipelines. For more info:
# http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
# Scale the dataset and send it back to a dictionary
scaled_df = df.copy()
scaler = MinMaxScaler()
scaled_df.iloc[:, 1:] = scaler.fit_transform(scaled_df.iloc[:, 1:])
my_dataset = scaled_df.to_dict(orient='index')

# Create and test the Gaussian Naive Bayes Classifier
print('\nNaive Bayes')
clf = GaussianNB()
final_project.tester.dump_classifier_and_data(clf, my_dataset, features_list)
final_project.tester.main()

# Create and test the Decision Tree Classifier
print('\nDecision Tree')
clf = DecisionTreeClassifier()
final_project.tester.dump_classifier_and_data(clf, my_dataset, features_list)
final_project.tester.main()

# Create and test the Support Vector Classifier
print('\nSupport Vector Classifier')
clf = SVC(kernel='poly')
final_project.tester.dump_classifier_and_data(clf, my_dataset, features_list)
final_project.tester.main()

# Create and test the k-Mean Cluster Classifier experimented with different cluster values but 2 was the best performing
print('\nk-mean Clustering')
clf = KMeans(n_clusters=2)
final_project.tester.dump_classifier_and_data(clf, my_dataset, features_list)
final_project.tester.main()

# 5: Tune your classifier to achieve better than .3 precision and recall
# using our testing script. Check the tester.py script in the final project
# folder for details on the evaluation method, especially the test_classifier
# function. Because of the small size of the dataset, the script uses
# stratified shuffle split cross validation. For more info:
# http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# GaussianNB does not have parameters to improve its results

# Create and test the Decision Tree Classifier (0.86667, .5, .65) to (.89333, .59091, .65)
print('\nDecision Tree improved')
clf = DecisionTreeClassifier(min_samples_leaf=2)
final_project.tester.dump_classifier_and_data(clf, my_dataset, features_list)
final_project.tester.main()

# Create and test the Support Vector Classifier() to (.86667, .5, .15) to (.87333, .52174, .6)
print('\nSupport Vector Classifier improved')
clf = SVC(kernel='poly', degree=11)
final_project.tester.dump_classifier_and_data(clf, my_dataset, features_list)
final_project.tester.main()

# Create and test the k-Mean Cluster Classifier experimented with different cluster values but 2 was the best performing
# changing the initialization parameters and then running and rerunning the data set gives drastically different results
# every time so further improvement is luck not actual improvement I received a perfect score with the settings how they
# are now but then ran again and did not meet minimum specs for any values
print('\nk-mean Clustering improved')
clf = KMeans(n_clusters=2, n_init=30)
final_project.tester.dump_classifier_and_data(clf, my_dataset, features_list)
final_project.tester.main()

print('\nVoting Classifier using new data and Naive Bias and Decision Tree')
# left off svc as it actually made the results less accurate and k-means is
# not a classifier so does not work for this purpose
# average scores for this are Accuracy: 0.86000	Precision: 0.47826	Recall: 0.55000	F1: 0.51163	F2: 0.53398
# Total predictions:  150	True positives:   11	False positives:   12	False negatives:    9	True negatives:  118
# Best scores for this are Accuracy: 0.87333	Precision: 0.52381	Recall: 0.55000	F1: 0.53659	F2: 0.54455
# 	Total predictions:  150	True positives:   11	False positives:   10	False negatives:    9	True negatives:  120
clf1 = GaussianNB()
clf2 = DecisionTreeClassifier()
# clf3 = SVC(kernel='poly')
# clf4 = KMeans(n_clusters=2)
estimators = []
estimators.append(('Naive Bias', clf1))
estimators.append(('Decision Tree', clf2))
# estimators.append(('SVC(poly)', clf3))
# estimators.append(('k-means', clf4))
clf = VotingClassifier(estimators)
final_project.tester.dump_classifier_and_data(clf, my_dataset, features_list)
final_project.tester.main()

# TODO 6: Dump your classifier, dataset, and features_list so anyone can
# check your results. You do not need to change anything below, but make sure
# that the version of poi_id.py that you submit can be run on its own and
# generates the necessary .pkl files for validating your results.

final_project.tester.dump_classifier_and_data(clf, my_dataset, features_list)
