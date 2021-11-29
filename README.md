# Udacity-ML-Project
Udacity nanodegree Machine Learning Project

Enron Submission Free-Response Questions

### Summarize for us the goal of this project and how machine learning is useful in trying to accomplish it. As part of your answer, give some background on the dataset and how it can be used to answer the project question. Were there any outliers in the data when you got it, and how did you handle those?  [relevant rubric items: “data exploration”, “outlier investigation”]

The goal of this project was to use machine learning to analyze whether a person in the Enron email dataset provided should be a person of interest or not regarding the court case and fraud charges. This dataset provides information regarding income, stock valuations, and email comings and goings of employees at Enron at the brink of its collapse. It does already include who was and was not a person of interest for this company, but this information may be useful in other company investigations after it has been normalized. There was a total row that isn't an outlier per se but is extraneous data. There was another one called The Travel Agency in The Park, but I left this one in the data in case it was being utilized to launder money or otherwise useful in identifying fraudulent activity. There were also 2 users whose financial data was shifted one space over (Robert Belfer and Sanjay Bhatnagar) and their data required corrective action to shift back into place.
There were quite a few empty values in the data provided, regarding the 2 types of financial information all null values were to be interpreted as 0 per the pdf provided with the data, and the null information in the email data I set equal to the mean of the column for their respective poi or non-poi subsets.  In the dataset, I used there were 145 lines with 20 built-in features there were 18 labeled as known POI and 127 labeled as non-POI. The following is a table showing all the features non-null counts in the data and as you can see the data is sparse. For all the financial features, null is a 0 according to the documentation so that can be fleshed out but the email features null is unknown values.
| #   | Column                      | Non-Null  |
|----:|:---------------------------:|:----------|
| 0   | poi                         | 145       |
| 1   | salary                      | 94        |
| 2   | bonus                       | 81        |
| 3   | long_term_incentive         | 65        |
| 4   | deferred_income             | 48        |
| 5   | deferral_payments           | 38        |
| 6   | loan_advances               | 3         |
| 7   | other                       | 92        |
| 8   | expenses                    | 94        |
| 9   | director_fees               | 16        |
| 10  | total_payments              | 124       |
| 11  | exercised_stock_options     | 101       |
| 12  | restricted_stock            | 109       |
| 13  | restricted_stock_deferred   | 17        |
| 14  | total_stock_value           | 125       |
| 15  | to_messages                 | 86        |
| 16  | from_messages               | 86        |
| 17  | from_poi_to_this_person     | 86        |
| 18  | from_this_person_to_poi     | 86        |
| 19  | shared_receipt_with_poi     | 86        |

### What features did you end up using in your POI identifier, and what selection process did you use to pick them? Did you have to do any scaling? Why or why not? As part of the assignment, you should attempt to engineer your own feature that does not come ready-made in the dataset -- explain what feature you tried to make, and the rationale behind it. (You do not necessarily have to use it in the final analysis, only engineer and test it.) In your feature selection step, if you used an algorithm like a decision tree, please also give the feature importances of the features that you use, and if you used an automated feature selection function like SelectKBest, please report the feature scores and reasons for your choice of parameter values.  [relevant rubric items: “create new features”, “intelligently select features”, “properly scale features”]

I used 'poi', 'salary', 'bonus', 'long_term_incentive', 'deferred_income', 'deferral_payments', 'loan_advances', 'other', 'expenses', 'director_fees', 'total_payments', 'exercised_stock_options', 'restricted_stock', 'restricted_stock_deferred', 'total_stock_value', 'to_messages', 'from_messages', 'from_poi_to_this_person', 'from_this_person_to_poi', 'shared_receipt_with_poi' as the base features and added the following with their respective formulas listed
'expense_income_ratio' = 'expenses'/('salary' + 'bonus'), 'inbound_poi_ratio' = 'from_poi_to_this_person' / 'to_messages', 'outbound_poi_ratio' = 'from_this_person_to_poi' / 'from_messages', 'listed_with_poi_ratio' = 'shared_receipt_with_poi' / 'to_messages', 'bonus_to_salary_ratio' = 'bonus' / 'salary', 'bonus_to_total_income_ratio' = 'bonus' / 'total_payments'
The expense_income_ratio was made because I believe a common way people commit fraud is overblown expenses to pocket the extra cash. If their expenses outclassed their normal income it would show that they potentially were committing fraud, inbound_poi_ratio was created because this shows the amount of mail from known poi's to their total mail coming in and if they are receiving a lot of mail from known poi then they may be one as well, the reverse is true as well if they are sending a lot of mail to poi's then they are probably one as well, also if they are included in chains with the poi they are more likely to be one hence listed_with_poi_ratio. The bonus to salary and bonus to total also seemed like a good location to find people who received too large of a bonus and were potentially committing fraud. Whether or not scaling was used was determined by which algorithm I was testing on the data set. My final algorithm takes an average of a 0.5% deduction in precision but Recall increases by 20% after using these new features so there is a marked overall improvement after adding these features. I did attempt using selectKBest to find the best columns to use in the decision tree. This did find that 4 of the 6 new features were in the most effective range at determining poi status. Eliminating any of the columns causes the precision and recall to drop drastically, anywhere from 10-50% lower depending on the run. This is even more pronounced when the new features are included with all the original features in the final run. the individual algorithms were improved with different tuning systems, the first was for Decision Trees, the only tuning required was changing min_samples_leaf to 2 from the default of 1. Changing the svc degree to degree 11 yielded the best results. For k-mean clustering I changed the number of n_init or the number of random seeds it runs with to 30, on more than 1 occasion it ran a perfect test, but it did run many very bad tests as well. I found that for this system k-means was just not reliable enough to use. I did find that when I used the Decision tree tuning that worked the best on the original algorithm by itself in the voting system that it lowered the effectiveness of the voting system. I attempted tuning the other variables for decision trees as well for the voting algorithm but found that any tuning that I performed dropped the precision and Recall by a minimum of 5%

### What algorithm did you end up using? What other one(s) did you try? How did model performance differ between algorithms?  [relevant rubric item: “pick an algorithm”]

My final algorithm is an ensemble algorithm called a voting classifier that combined the Gaussian Naive Bias and Decision Tree systems to get the most accurate results. I tried Gaussian Naive Bayes (bad precision with a lot of false positives), Decision Tree(passing before tuning, only slight improvement made with tuning), Support Vector Classifier(failed due to recall before tuning, lowered precision slightly but still passing and quadrupled recall well within passing after tuning), k-mean clustering (any changes made to k-mean clustering changed nothing in its output, running it repeatedly had it all over the place including 1 perfect run and 1 complete fail with no accurate predictions), I also tried other ensemble methods like BaggingClassifier, RandomForestClassifier, GradientBoostingClassifier but none of them improved the output of the single methods let alone the voting classifier method.

### What does it mean to tune the parameters of an algorithm, and what can happen if you don’t do this well?  How did you tune the parameters of your particular algorithm? What parameters did you tune? (Some algorithms do not have parameters that you need to tune -- if this is the case for the one you picked, identify and briefly explain how you would have done it for the model that was not your final choice or a different model that does utilize parameter tuning, e.g. a decision tree classifier).
[relevant rubric items: “discuss parameter tuning”, “tune the algorithm”]

Tuning your parameters allows you to choose how it behaves so you do not over or under classify the data. Over classification is where you get too specific for what you are trying to find, this can look like a scatter plot, and only if it falls on a known point do you classify it as something but outside of a known point it is classified incorrectly, under classifying the data more along the lines of too broad of a classification, imagine if you had to pick whether something is a car, truck or van, under classification, would be if it's small it’s a car, if it is big it’s a truck. leaving out the van data entirely. The voting classifier uses multiple methods of classifiers, I attempted the 3 basic methods and found the best results were when I dropped SVC entirely and just used Gaussian NB which has no tunable parameters and decision trees with their default values, every attempted change I made to decision trees caused the results to lower. Even if I used the improved results I found on the tuning of the individual algorithm. I tested the decision tree criterion changing it to entropy which made the results less precise by an average of 8% and recall by 4%, I changed splitter to random which made the results vary by as much as 34% between runs, the one parameter that worked on the decision tree by itself but not on the voting combined methods was min_samples_leaf which improved the individual method by about 4% on precision but had no major change in recall or accuracy.

### What is validation, and what’s a classic mistake you can make if you do it wrong? How did you validate your analysis?  [relevant rubric items: “discuss validation”, “validation strategy”]

Validation is checking your test set against known data to see if its predictions are accurate or not, a classic mistake is checking data against itself when you have used that same data to train with. I used the method built into the tester that was provided called StratifiedShuffleSplit. This provided tester returned accuracy metrics for each algorithm that it was run on.

### Give at least 2 evaluation metrics and your average performance for each of them.  Explain an interpretation of your metrics that says something human-understandable about your algorithm’s performance. [relevant rubric item: “usage of evaluation metrics”]

the following is the output from the tester function as the best I have seen the voting classifier output

Accuracy: 0.87333          
Precision: 0.52381          
Recall: 0.55000 F1: 0.53659        
F2: 0.54455

Total predictions:
150   True positives:   11         
False positives:   10         False negatives:    9        
True negatives:  120

Accuracy is how many true predictions there are compared to the total, so 87.33% accurate, precision is how many true positives there are compared to all marked as positive regardless of whether it's correct or not so 52.4% precise, recall is how many true positives there were compared to how many actual positives there are or 55% recalled correctly. More informally, accuracy is not a very good metric for this small sample size but shows how many entities (87.3%) were identified correctly as POI and non-POI out of the total population provided. Precision measures how many of the entities (52.4%) were labeled as a POI correctly compared to how many were labeled as a POI total. Recall is how many entities (55%) were labeled as POI correctly compared to how many there were supposed to be.
