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

#    for i in range(component_number):
#        df['pca_feature' + str(i)] = pca_feature[:,i]
    for column in features.columns:
        df['pca_residual_' + column] = pca_residual[column]


print "reading data"
df = pandas.read_csv('./train.csv')

print "reprocessing"

df = addMissingPattern(df)

fillNull(df)

convertToNumeric(df)

addPCAFeature(df)

print "cross_validating"
scores = cross_validate(df.drop(['ID','target'], axis=1), df['target']) 
# fit_and_print_importance(df.drop(['ID','target'], axis=1), df['target']) 

print "Cross validation scores:"
print scores
print "Average: {0}".format(sum(scores) / 10.0)
