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

start_param_grid = {
        "n_estimators": [850],
        "max_features": [40],
        "min_samples_split": [4],
        "max_depth": [40],
        "min_samples_leaf": [2],
}

auto_grid_search(df.drop(['ID','target'], axis=1), df['target']) 
