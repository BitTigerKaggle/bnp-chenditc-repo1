#!/usr/bin/python -u
import pandas
import zipfile
from time import time
from sklearn import preprocessing
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.naive_bayes import BernoulliNB
from sklearn import ensemble
from sklearn import cross_validation

# use a full grid over all parameters
param_grid = {
#                  "n_estimators": [20, 200, 600, 1000],
              "n_estimators": [600],
              "max_features": [80],
              "min_samples_split": [1, 5, 10, 20],
              "max_depth": [30, 40, 50],
#		  "min_samples_leaf": [2, 5, 10],
#		  "criterion": ["gini", "entropy"]
}



def addMissingPattern(df):
    # Add missing data pattern identifier.
    df['missing_pattern'] = df.apply(lambda row: str(row.isnull().tolist()), axis=1)
    return df

def fillNull(df):
    return df.fillna(-999, inplace=True)

def getModel():
    extc = ExtraTreesClassifier(n_estimators=850,max_features= 60,criterion= 'entropy',min_samples_split= 4,
                                        max_depth= 40, min_samples_leaf= 2, n_jobs = -1)  
    return extc

def labelEncode(df, columnName):
    # train label encoder
    le = preprocessing.LabelEncoder()
    df[columnName] = le.fit_transform(df[columnName])


def convertToNumeric(df):
    features = df.columns[2:]
    for col in features:
        if((df[col].dtype == 'object')):
            print "Converting {0} to numerical data".format(col)
            labelEncode(df, col)
            nb = BernoulliNB()
            nb.fit(df[[col]], df['target'])
            new_col = col + "_binarized"
            df[new_col] = nb.predict_proba(df[[col]])[:, 1]

def cross_validate(data, target):
    clf = getModel()
    scores = cross_validation.cross_val_score(clf, data, target, cv = 10, scoring='log_loss')
    return scores

def fit_and_print_importance(data, target):
    clf = getModel()
    clf.fit(data, target)

    importance_pairs = []
    i = 0
    for column in data.columns:
        importance_pairs.append((column, clf.feature_importances_[i]))
        i += 1

    importance_pairs = sorted(importance_pairs, key=lambda x: x[1], reverse=True)
    for item in importance_pairs:
        print item

def grid_search(data, target):
    # run grid search
    clf = getModel()
    grid_search = GridSearchCV(clf, param_grid=param_grid, scoring='log_loss', n_jobs=-1, pre_dispatch=2, verbose=6)
    start = time()
    grid_search.fit(data, target)

    print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
	  % (time() - start, len(grid_search.grid_scores_)))
    print(grid_search.grid_scores_)

def get_neighbor_values(best_value, alpha):
    lower_value = int(best_value * (1-alpha))
    higher_value = int(best_value * (1+alpha))  

    new_values = []
    if (lower_value is not 0):
        new_values.append(lower_value)

    new_values.append(best_value)

    if (higher_value is best_value):
        new_values.append(best_value + 1)
    else:
        new_values.append(higher_value)
    return new_values

def auto_grid_search(data, target):
    start_param_grid = {
            "n_estimators": [1121],
            "max_features": [60],
            "min_samples_split": [4],
            "max_depth": [40],
            "min_samples_leaf": [2],
    }
    temp_param_grid = start_param_grid
    best_param_grid = {
            "n_estimators": 1121,
            "max_features": 60,
            "min_samples_split": 4,
            "max_depth": 40,
            "min_samples_leaf": 2,
    }


    best_score = -999
    new_best_score = -99    
    alpha = 0.06

    param_queue = start_param_grid.keys()

    while (True): 
        best_score = new_best_score 

        alpha = alpha * 0.9

        # run grid search and compute new best param
        print "Seaching params:", temp_param_grid
        clf = getModel()
        searcher = GridSearchCV(clf, param_grid=temp_param_grid, scoring='log_loss', n_jobs=-1, pre_dispatch=2, verbose=6, cv=2)
        start = time()
        searcher.fit(data, target)
        print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
              % (time() - start, len(searcher.grid_scores_)))
        print "Best parameter is: ", searcher.best_params_
        print "Best score is:", searcher.best_score_
        new_best_score = searcher.best_score_

        # Generate new param grid from the best one.
        new_param_grid = {}

        key = param_queue[0]
        print "Changing ", key , "from", best_param_grid[key], "to", searcher.best_params_[key]
        if str(searcher.best_params_[key]) != str(best_param_grid[key]):
            print type(searcher.best_params_[key])
            print type(best_param_grid[key])
            # 1. If this was improved, keep searching
            best_value = searcher.best_params_[key]
            new_values = get_neighbor_values(best_value, alpha)
            new_param_grid[key] = new_values
        else:
            # 2. find the next one to tune
            param_queue = param_queue[1:] + param_queue[:1]
            key = param_queue[0] 
            best_value = searcher.best_params_[key]
            new_values = get_neighbor_values(best_value, alpha)
            new_param_grid[key] = new_values

        print "Tuning key:", key

        # 3. Fill in best params for the rest 
        for key in best_param_grid:
            if key not in new_param_grid:
                new_param_grid[key] = [best_param_grid[key]]

        temp_param_grid = new_param_grid
        best_param_grid = searcher.best_params_


