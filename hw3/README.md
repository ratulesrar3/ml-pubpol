In this homework, you'll you've continue to build the Machine Learning pipeline by combining what you have been doing in labs and your previous homework(s). The goal is to improve the pipeline based on the feedback from previous assignments, and add a few components based on what we've covered in the past few lectures. More specifically, you need to:

 Coding Assignment:

1. Fix and improve the pipeline code you submitted for the last assignment based on the feedback from the TA. if something critical was pointed out in the feedback, you need to fix it. 

2. Add more classifiers to the pipeline on the code you've written in lab. I’d recommend at least having Logistic Regression, K-Nearest Neighbor, Decision Trees, SVM, Random Forests, Boosting, and Bagging. The code should have a parameter for running one or more of these classifiers and your analysis should run all of them.

3. Experiment with different parameters for these classifiers (different values of k for example, as well as parameters that other classifiers have). You should look at the sklearn documentation to see what parameter each classifier can take and what the default values sklearn selects.

4. Add additional evaluation metrics that we've covered in class to the pipeline (accuracy, precision at different levels, recall at different levels, F1, area under curve, and precision-recall curves).

5. Create temporal validation function in your pipeline that can create training and test sets over time. You can choose the length of these splits based on analyzing the data. For example, the test sets could be six months long and the training sets could be all the data before each test set.

Analysis:

5. Once you've set up the improved pipeline, you can use it to solve the problem at https://www.kaggle.com/c/kdd-cup-2014-predicting-excitement-at-donors-choose/data (Links to an external site.)Links to an external site.

instead of using all the data, please only use the data from projects and outcomes files (and to make things simpler use data from years 2011-2013 The goal is to predict, at posting time of a project,  if a project will not get fully funded so we can intervene and help them improve the project listing. 

The code should produce a table with results across train test splits over time and performance metrics (baseline, precision and recall at different thresholds 1%, 2%, 5%, 10%, 20%, 30%, 50% and AUC_ROC)

Report:

You should also write a short report (~2 pages) that compares the performance of the different classifiers across all the metrics for the data set used in the last assignment. Which classifier does better on which metrics? How do the results change over time? What would be your recommendation to someone who's working on this model on what model to go forward with?

The report should not be a list of graphs and numbers. It needs to explain to a policy audience the implications of your analysis and your recommendations as a memo you would send to a business/policy audience.
