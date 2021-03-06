### adapted from rayid's mlfunctions

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier, OrthogonalMatchingPursuit, RandomizedLogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn.metrics import *
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve
import random
import pylab as pl
import matplotlib.pyplot as plt
import seaborn as sns
import time
import sys
from datetime import datetime, timedelta


# Below are configurations for Rayid's Magic Loop functions
# all classifiers and their default params
CLASSIFIERS = {
            'RF': RandomForestClassifier(n_estimators=50, n_jobs=-1),
            'ET': ExtraTreesClassifier(n_estimators=10, n_jobs=-1, criterion='entropy'),
            'AB': AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), algorithm="SAMME", n_estimators=200),
            'LR': LogisticRegression(penalty='l1', C=1e5),
            'SVM': SVC(kernel='linear', probability=True, random_state=0),
            'GB': GradientBoostingClassifier(learning_rate=0.05, subsample=0.5, max_depth=6, n_estimators=10),
            'NB': GaussianNB(),
            'DT': DecisionTreeClassifier(),
            'SGD': SGDClassifier(loss="hinge", penalty="l2"),
            'KNN': KNeighborsClassifier(n_neighbors=3)
            }

# grid sizes
LARGE_GRID = {
            'RF': {'n_estimators': [1,10,100,1000,10000], 'max_depth': [1,5,10,20,50,100], 'max_features': ['sqrt','log2'],'min_samples_split': [2,5,10]},
            'LR': {'penalty': ['l1','l2'], 'C': [0.00001,0.0001,0.001,0.01,0.1,1,10]},
            'SGD': {'loss': ['hinge','log','perceptron'], 'penalty': ['l2','l1','elasticnet']},
            'ET': {'n_estimators': [1,10,100,1000,10000], 'criterion' : ['gini', 'entropy'] ,'max_depth': [1,5,10,20,50,100], 'max_features': ['sqrt','log2'], 'min_samples_split': [2,5,10]},
            'AB': {'algorithm': ['SAMME', 'SAMME.R'], 'n_estimators': [1,10,100,1000,10000]},
            'GB': {'n_estimators': [1,10,100,1000,10000], 'learning_rate': [0.001,0.01,0.05,0.1,0.5],'subsample': [0.1,0.5,1.0], 'max_depth': [1,3,5,10,20,50,100]},
            'NB': {},
            'DT': {'criterion': ['gini', 'entropy'], 'max_depth': [1,5,10,20,50,100], 'max_features': ['sqrt','log2'], 'min_samples_split': [2,5,10]},
            'SVM': {'C':[0.00001,0.0001,0.001,0.01,0.1,1,10], 'kernel':['linear']},
            'KNN': {'n_neighbors': [1,5,10,25,50,100],'weights': ['uniform','distance'],'algorithm': ['auto','ball_tree','kd_tree']}
       }

SMALL_GRID = {
            'RF': {'n_estimators': [10,100], 'max_depth': [5,50], 'max_features': ['sqrt','log2'],'min_samples_split': [2,10]},
            'LR': {'penalty': ['l1','l2'], 'C': [0.001,0.1,1]},
            'SGD': {'loss': ['hinge','log','perceptron'], 'penalty': ['l2','l1','elasticnet']},
            'ET': {'n_estimators': [10,100], 'criterion' : ['gini', 'entropy'] ,'max_depth': [5,50], 'max_features': ['sqrt','log2'],'min_samples_split': [2,10]},
            'AB': {'algorithm': ['SAMME', 'SAMME.R'], 'n_estimators': [10,100,1000]},
            'GB': {'n_estimators': [10,100], 'learning_rate' : [0.001,0.1,0.5],'subsample' : [0.1,0.5,1.0], 'max_depth': [5,50]},
            'NB': {},
            'DT': {'criterion': ['gini', 'entropy'], 'max_depth': [10,20,50], 'max_features': ['sqrt','log2'],'min_samples_split': [2,5,10]},
            'SVM': {'C': [0.00001,0.0001,0.001,0.01,0.1,1,10], 'kernel': ['linear']},
            'KNN': {'n_neighbors': [1,5,10],'weights': ['uniform','distance'],'algorithm': ['auto','ball_tree','kd_tree']}
       }

TEST_GRID = {
            'RF': {'n_estimators': [1], 'max_depth': [1], 'max_features': ['sqrt'],'min_samples_split': [10]},
            'LR': {'penalty': ['l1'], 'C': [0.01]},
            'SGD': { 'loss': ['perceptron'], 'penalty': ['l2']},
            'ET': {'n_estimators': [1], 'criterion' : ['gini'] ,'max_depth': [1], 'max_features': ['sqrt'],'min_samples_split': [10]},
            'AB': {'algorithm': ['SAMME'], 'n_estimators': [1]},
            'GB': {'n_estimators': [1], 'learning_rate' : [0.1], 'subsample' : [0.5], 'max_depth': [1]},
            'NB': {},
            'DT': {'criterion': ['gini'], 'max_depth': [1], 'max_features': ['sqrt'],'min_samples_split': [10]},
            'SVM': {'C':[0.01],'kernel':['linear']},
            'KNN': {'n_neighbors': [5],'weights': ['uniform'],'algorithm': ['auto']}
       }

TO_RUN = ['GB','RF','DT','KNN','LR','NB']

def generate_binary_at_k(y_scores, k):
    '''
    helper function to get binary predictions at thershold k
    '''
    cutoff_index = int(len(y_scores) * (k / 100.0))
    test_predictions_binary = [1 if x < cutoff_index else 0 for x in range(len(y_scores))]
    return test_predictions_binary


def scores_at_k(y_true, y_scores, k):
    '''
    Returns precision, recall, and accuracy at threshold k
    '''
    preds_at_k = generate_binary_at_k(y_scores, k)
    precision = precision_score(y_true, preds_at_k)
    accuracy = accuracy_score(y_true, preds_at_k)
    recall = recall_score(y_true, preds_at_k)
    return precision, accuracy, recall

def plot_precision_recall_n(y_true, y_prob, model_name):
    '''
    Plots precision and recall for population on one chart
    '''
    y_score = y_prob
    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_true, y_score)
    precision_curve = precision_curve[:-1]
    recall_curve = recall_curve[:-1]
    pct_above_per_thresh = []
    number_scored = len(y_score)
    for value in pr_thresholds:
        num_above_thresh = len(y_score[y_score>=value])
        pct_above_thresh = num_above_thresh / float(number_scored)
        pct_above_per_thresh.append(pct_above_thresh)
    pct_above_per_thresh = np.array(pct_above_per_thresh)

    plt.clf()
    fig, ax1 = plt.subplots()
    ax1.plot(pct_above_per_thresh, precision_curve, 'b')
    ax1.set_xlabel('percent of population')
    ax1.set_ylabel('precision', color='b')
    ax2 = ax1.twinx()
    ax2.plot(pct_above_per_thresh, recall_curve, 'r')
    ax2.set_ylabel('recall', color='r')
    ax1.set_ylim([0,1])
    ax1.set_ylim([0,1])
    ax2.set_xlim([0,1])

    name = model_name
    plt.title(name)
    plt.show()

    
def get_feature_importance(clf, model_name):
    clfs = {'RF':'feature_importances',
            'LR': 'coef',
            'SVM': 'coef',
            'DT': 'feature_importances',
            'KNN': None,
            'AB': 'feature_importances',
            'GB': None,
            'linear.SVC': 'coef',
            'ET': 'feature_importances'
            }

    if clfs[model_name] == 'feature_importances':
        return  list(clf.feature_importances_)
    elif clfs[model_name] == 'coef':
        return  list(clf.coef_.tolist())
    else:
        return None