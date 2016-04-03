#!/usr/bin/python -u
import pandas
import zipfile
from sklearn import preprocessing
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn import ensemble
from sklearn import cross_validation

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
