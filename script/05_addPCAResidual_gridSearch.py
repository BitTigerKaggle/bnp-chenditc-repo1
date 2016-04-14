#!/usr/bin/python -u

from sklearn.decomposition import PCA

from base import *


def addPCAFeature(df):
    features = df.select_dtypes(['float64'])

    print "input feature shape:", features.shape

    component_number = 80

    pca = PCA(n_components=component_number)
    pca.fit(features)

    pca_feature = pca.transform(features)
    pca_residual = features - pca.inverse_transform(pca_feature)

    print "pca_feature shape:", pca_feature.shape
    print "pca_residual shape:", pca_residual.shape

    for column in features.columns:
        df['pca_residual_' + column] = pca_residual[column]


print "reading data"
df = pandas.read_csv('./train.csv')

print "reprocessing"

df = addMissingPattern(df)

fillNull(df)

convertToNumeric(df)

addPCAFeature(df)

def auto_grid_search(data, target):
    start_param_grid = {
            "n_estimators": [850],
            "max_features": [60],
            "min_samples_split": [4],
            "max_depth": [40],
            "min_samples_leaf": [2],
    }
    temp_param_grid = start_param_grid

    best_score = -999
    new_best_score = -99    

    while (new_best_score - best_score > 0.0001): 
        best_score = new_best_score 

        # run grid search and compute new best param
        clf = getModel()
        searcher = GridSearchCV(clf, param_grid=temp_param_grid, scoring='log_loss', n_jobs=-1, pre_dispatch=2, verbose=6)
        start = time()
        searcher.fit(data, target)
        print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
              % (time() - start, len(searcher.grid_scores_)))
        print "Best parameter is: ", searcher.best_params_
        print "Best score is:", searcher.best_score_
        new_best_score = searcher.best_score_

        # Generate new param grid from the best one.
        new_param_grid = {}
        for key in searcher.best_params_:
            best_value = searcher.best_params_[key]
            lower_value = int(best_value * 0.9)
            higher_value = int(best_value * 1.1)  

            new_values = []
            if (lower_value is not 0):
                new_values.append(lower_value)

            new_values.append(best_value)
     
            if (higher_value is best_value):
                new_values.append(best_value + 1)
            else:
                new_values.append(higher_value)

            new_param_grid[key] = new_values

        temp_param_grid = new_param_grid


auto_grid_search(df.drop(['ID','target'], axis=1), df['target']) 
