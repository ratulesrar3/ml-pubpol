from methods import *
from preprocess import *

# which parameter grid do we want to use (test, small, large)
GRID = TEST_GRID

#read the csv data
#DATA = df

# which variable to use for prediction_time
PREDICTION_DATE = 'date_posted'

# outcome
OUTCOME = 'label'

# list of classifier models to run
TO_RUN = ['GB','RF','DT','KNN','LR','NB']

CUTOFF_VAL_PAIRS = [('2011-06-30', '2011-12-31'), ('2011-12-31', '2012-06-30'),
                    ('2012-06-30', '2012-12-31'), ('2012-12-31', '2013-06-30'),
                    ('2013-06-30', '2013-12-31')]

def temporal_split(df, col, cutoff_date, validation_date):
    '''
    Returns train, test pairs
    '''
    train_end = datetime.strptime(cutoff_date, '%Y-%m-%d')
    test_end = datetime.strptime(validation_date, '%Y-%m-%d')
    train = df[df[col] <= train_end]
    test = df[(df[col] > train_end) & (df[col] <= test_end)]

    return train, test


def prep_data(clean_train, clean_test, features):
    '''
    Gets X, y train/test vectors based on features list
    '''
    features = list(features)
    X_train = clean_train.filter(features)
    y_train = clean_train.filter(OUTCOME)
    X_test = clean_test.filter(features)
    y_test = clean_test.filter(OUTCOME)
    return X_train, X_test, y_train, y_test


# define dataframe to write results to
RESULTS_DF =  pd.DataFrame(columns=('model_type','clf', 'parameters', 'validation_date',
                                    'train_set_size', 'validation_set_size', 'baseline',
                                    'precision_at_5','precision_at_10','precision_at_20',
                                    'recall_at_5','recall_at_10','recall_at_20','auc-roc'))



def temporal_validation_loop(df, grid_size=TEST_GRID):
    '''
    Loops through dates for temporal validation
    '''
    for c, v in CUTOFF_VAL_PAIRS:
        train, test = temporal_split(df, PREDICTION_DATE, c, v)
        X_train, X_test, y_train, y_test = prep_data(*pre_process(train, test))

        for i, clf in enumerate([CLASSIFIERS[x] for x in TO_RUN]):
            print(TO_RUN[i])
            params = GRID[TO_RUN[i]]
            for p in ParameterGrid(params):
                try:
                    clf.set_params(**p)
                    clf.fit(X_train, y_train)
                    y_pred_probs = clf.fit(X_train, y_train.values.ravel()).predict_proba(X_test)[:,1]
                    y_pred_probs_sorted, y_test_sorted = zip(*sorted(zip(y_pred_probs, y_test.values.ravel()), reverse=True))

                    precision_5, accuracy_5, recall_5 = scores_at_k(y_test_sorted,y_pred_probs_sorted,5.0)
                    precision_10, accuracy_10, recall_10 = scores_at_k(y_test_sorted,y_pred_probs_sorted,10.0)
                    precision_20, accuracy_20, recall_20 = scores_at_k(y_test_sorted,y_pred_probs_sorted,20.0)

                    RESULTS_DF.loc[len(RESULTS_DF)] = [TO_RUN[i], clf, p, v,
                                                        y_train.shape[0], y_test.shape[0],
                                                        scores_at_k(y_test_sorted,y_pred_probs_sorted,100.0),
                                                        precision_5, precision_10, precision_20,
                                                        recall_5, recall_10, recall_20,
                                                        roc_auc_score(y_test, y_pred_probs)]

                    plot_precision_recall_n(y_test,y_pred_probs,clf)

                except IndexError:
                    print('Error')
                    continue

    return RESULTS_DF



### original classfifier loop ###

def classifiers_loop(X_train, X_test, y_train, y_test, grid_size=TEST_GRID):
    results =  pd.DataFrame(columns=('model_type','clf', 'parameters', 'auc-roc',
                                     'precision_5', 'accuracy_5', 'recall_5',
                                     'precision_10', 'accuracy_10', 'recall_10',
                                     'precision_20', 'accuracy_20', 'recall_20',
                                     'runtime', 'y_pred_probs'))
    for i, clf in enumerate([CLASSIFIERS[x] for x in TO_RUN]):
        #print(TO_RUN[i])
        params = grid_size[TO_RUN[i]]
        for p in ParameterGrid(params):
            try:
                start_time = time.time()
                clf.set_params(**p)

                y_pred_probs = clf.fit(X_train, y_train.values.ravel()).predict_proba(X_test)[:,1]
                y_pred_probs_sorted, y_test_sorted = zip(*sorted(zip(y_pred_probs, y_test.values.ravel()), reverse=True))
                end_time = time.time()
                tot_time = end_time - start_time
                #print(p)
                precision_5, accuracy_5, recall_5 = scores_at_k(y_test_sorted,y_pred_probs_sorted,5.0)
                precision_10, accuracy_10, recall_10 = scores_at_k(y_test_sorted,y_pred_probs_sorted,10.0)
                precision_20, accuracy_20, recall_20 = scores_at_k(y_test_sorted,y_pred_probs_sorted,20.0)
                results.loc[len(results)] = [TO_RUN[i], clf, p,
                                                       roc_auc_score(y_test, y_pred_probs),
                                                       precision_5, accuracy_5, recall_5,
                                                       precision_10, accuracy_10, recall_10,
                                                       precision_20, accuracy_20, recall_20,
                                                       tot_time, y_pred_probs]
                #plot_precision_recall_n(y_test,y_pred_probs,clf)
            except IndexError:
                print('Error')
                continue

    return results
