import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sqlalchemy import Table, create_engine, MetaData
import sys
import os
import json

data = pd.read_csv('csv/testdateimitlabels.csv',low_memory=False)

DATASET = json.loads(sys.argv[1])
run_id = sys.argv[2]
inserts = []

col_names = data.columns
col_list = col_names.tolist()
#Get all keys except years, rics and labels(all, relevant, relevant5%)
keys_X = []
for x in range(5,839):
    keys_X.append(col_list[x])

# get all data except years,rics and labels
X = data[keys_X]

#Delete all boolean type or object type(datum) features
X = X.select_dtypes(exclude=['bool','object'])

#Replace all the NAN with mean
imp = SimpleImputer(strategy="mean")
X = imp.fit_transform(X)

#Transform the data as data frame
X = pd.DataFrame(X)
print(X)

X.info()
#get all the labels
y1 = data[['all']]
y2 = data[['relevant']]
y3 = data[['relevant5%']]


########################################################################################################################
#    prediction for label "all"
########################################################################################################################

#Using XGBoost for featureboost, each feature get a score.
#Using SelectFromModel, we can extract features, whose score is higher than average score.
model = XGBClassifier()
model.fit(X, y1.values.ravel())
print('feature_importances of all')
print(model.feature_importances_)
selection = SelectFromModel(model,prefit=True,threshold="mean")
select_X = selection.transform(X)
select_X = pd.DataFrame(select_X)
print(select_X)

y1 = pd.DataFrame(y1)

# connect label and the data after applying FeatureBoost
sel_X = pd.concat([y1,select_X],axis=1)
sel_X.info()
print("selX")
print(sel_X)

#get all datas with label "all" is 1
X_all1 = sel_X.loc[sel_X["all"] == 1]

#get all datas with label "all" is 0
X_all0 = sel_X.loc[sel_X["all"] == 0]

#get all data with label "all" is 1 except the column of label
X_all_1 = X_all1.iloc[:, 1:]
print("X_all_1##########################################")
print(X_all_1)
#get all data with label "all" is 0 except the column of label
X_all_0 = X_all0.iloc[:, 1:]
print("X_all_0###########################################")
print(X_all_0)

X_all_1.info()
X_all_0.info()

#set training data, isolation Forest is a semi-supervised algorithm, all training data is normal, and we set 4/5 data set
#as training data
X0_train = X_all_0.loc[0:109196]
print("X0_train############################################")
print(X0_train)
#set test data, the rest of normal data as test set
X0_test = X_all_0.loc[109196:]
print("X0_test############################################")
print(X0_test)

#create a classifier of Isolation Forest
clf = IsolationForest(**DATASET)
clf.fit(X0_train)

#use this classifier to predict outliers and test data
y_pred_test = clf.predict(X0_test)
y_pred_outliers = clf.predict(X_all_1)


# print for a confusion matrix and report.
print("amount of target is  0 and prediction is also 0:")
a00 = list(y_pred_test).count(1)
print(a00)

print("amount of target is  0 and prediction is 1:")
a01 = list(y_pred_test).count(-1)
print(a01)

print("amount of target is  1 and prediction is also 1:")
a11 = list(y_pred_outliers).count(-1)
print(a11)

print("amount of target is  1 and prediction is 0:")
a10 = list(y_pred_outliers).count(1)
print(a10)

print("amount of normal test data")
anormal =y_pred_test.shape[0]
print(anormal)

print("amount of test outliers ")
aoutlier = y_pred_outliers.shape[0]
print(aoutlier)



print("accuracy is")
print((a00+a11)/(anormal+aoutlier))

print("precision of 1 is")
precision_1 = a11/(a11+a01)
print(precision_1)

print("recall of 1 is")
recall_1 = a11/(a11+a10)
print(recall_1)

print("f1score of 1 is")
print((2*recall_1*precision_1)/(recall_1 + precision_1))

print("precision of 0 is")
precision_0 = a00/(a10+a00)
print(precision_0)

print("recall of 0 is")
recall_0 = a00/(a01+a00)
print(recall_0)

print("f1score of 0 is")
print((2*recall_0*precision_0)/(recall_0 + precision_0))

inserts.append(
        dict(
            label="all",
            run_id=run_id,
            accuracy=float((a00+a11)/(anormal+aoutlier)),
            precision00=float(a00/(a10+a00)),
            recall00=float(a00/(a01+a00)),
            f1_score00=float((2*recall_0*precision_0)/(recall_0 + precision_0)),
            precision01=float(a11/(a11+a01)),
            recall01=float(a11/(a11+a10)),
            f1_score01=float((2*recall_1*precision_1)/(recall_1 + precision_1))
            )
        )

########################################################################################################################
#     prediction for label "relevant"
########################################################################################################################


#Using XGBoost for featureboost, each feature get a score.
#Using SelectFromModel, we can extract features, whose score is higher than average score.
model = XGBClassifier()
model.fit(X, y2.values.ravel())
print('feature_importances of relevant')
print(model.feature_importances_)
selection = SelectFromModel(model,prefit=True,threshold="mean")
select_X = selection.transform(X)
select_X = pd.DataFrame(select_X)
print(select_X)

y2 = pd.DataFrame(y2)

sel_X = pd.concat([y2,select_X],axis=1)

#get all datas with label "relevant" is 1
X_rel1 = sel_X.loc[sel_X["relevant"] == 1]
#get all datas with label "relevant" is 0
X_rel0 = sel_X.loc[sel_X["relevant"] == 0]

#get all data with label "relevant" is 1 except the column of label
X_rel_1 = X_rel1.iloc[:, 1:]
print(X_rel_1)
#get all data with label "relevant" is 0 except the column of label
X_rel_0 = X_rel0.iloc[:, 1:]
print(X_rel_0)

print(X_rel_1)
X_rel1.info()
print(X_rel_0)
X_rel0.info()

#set training data, isolation Forest is a semi-supervised algorithm, all training data is normal, and we set 4/5 data set
#as training data
X0_train = X_rel_0.loc[0:109196]
print(X0_train)

#set test data, the rest of normal data as test set
X0_test = X_rel_0.loc[109196:]
print(X0_test)

#create a classifier of Isolation Forest
clf = IsolationForest(**DATASET)
clf.fit(X0_train)

#use this classifier to predict outliers and test data
y_pred_test = clf.predict(X0_test)
y_pred_outliers = clf.predict(X_rel_1)

# print for a confusion matrix and report.
print("amount of target is  0 and prediction is also 0:")
a00 = list(y_pred_test).count(1)
print(a00)

print("amount of target is  0 and prediction is 1:")
a01 = list(y_pred_test).count(-1)
print(a01)

print("amount of target is  1 and prediction is also 1:")
a11 = list(y_pred_outliers).count(-1)
print(a11)

print("amount of target is  1 and prediction is 0:")
a10 = list(y_pred_outliers).count(1)
print(a10)

print("amount of normal test data")
anormal =y_pred_test.shape[0]
print(anormal)

print("amount of test outliers ")
aoutlier = y_pred_outliers.shape[0]
print(aoutlier)



print("accuracy is")
print((a00+a11)/(anormal+aoutlier))

print("precision of 1 is")
precision_1 = a11/(a11+a01)
print(precision_1)

print("recall of 1 is")
recall_1 = a11/(a11+a10)
print(recall_1)

print("f1score of 1 is")
print((2*recall_1*precision_1)/(recall_1 + precision_1))

print("precision of 0 is")
precision_0 = a00/(a10+a00)
print(precision_0)

print("recall of 0 is")
recall_0 = a00/(a01+a00)
print(recall_0)

print("f1score of 0 is")
print((2*recall_0*precision_0)/(recall_0 + precision_0))

inserts.append(
        dict(
            label="relevant",
            run_id=run_id,
            accuracy=float((a00+a11)/(anormal+aoutlier)),
            precision00=float(a00/(a10+a00)),
            recall00=float(a00/(a01+a00)),
            f1_score00=float((2*recall_0*precision_0)/(recall_0 + precision_0)),
            precision01=float(a11/(a11+a01)),
            recall01=float(a11/(a11+a10)),
            f1_score01=float((2*recall_1*precision_1)/(recall_1 + precision_1))
            )
        )

########################################################################################################################
#     prediction for label"relevant5%"
########################################################################################################################

#Using XGBoost for featureboost, each feature get a score.
#Using SelectFromModel, we can extract features, whose score is higher than average score.
model = XGBClassifier()
model.fit(X, y3.values.ravel())
print('feature_importances of relevant5%')
print(model.feature_importances_)
selection = SelectFromModel(model,prefit=True,threshold="10*mean")
select_X = selection.transform(X)
select_X = pd.DataFrame(select_X)
print(select_X)

y3 = pd.DataFrame(y3)

sel_X = pd.concat([y3,select_X],axis=1)

#get all datas with label "relevant5%" is 1
X_5rel1 = sel_X.loc[sel_X["relevant5%"] == 1]
#get all datas with label "relevant5%" is 0
X_5rel0 = sel_X.loc[sel_X["relevant5%"] == 0]

#get all data with label "relevant5%" is 1 except the column of label
X_5rel_1 = X_5rel1.iloc[:, 1:]
print(X_5rel_1)
#get all data with label "relevant5%" is 0 except the column of label
X_5rel_0 = X_5rel0.iloc[:, 1:]
print(X_5rel_0)

print(X_5rel_1)
X_5rel1.info()
print(X_5rel_0)
X_5rel0.info()

#set training data, isolation Forest is a semi-supervised algorithm, all training data is normal, and we set 4/5 data set
#as training data
X0_train = X_5rel_0.loc[0:109196]
print(X0_train)
#set test data, the rest of normal data as test set
X0_test = X_5rel_0.loc[109196:]
print(X0_test)

#create a classifier of Isolation Forest
clf = IsolationForest(**DATASET)
clf.fit(X0_train)

#use this classifier to predict outliers and test data
y_pred_test = clf.predict(X0_test)
y_pred_outliers = clf.predict(X_5rel_1)


# print for a confusion matrix and report.
print("amount of target is  0 and prediction is also 0:")
a00 = list(y_pred_test).count(1)
print(a00)

print("amount of target is  0 and prediction is 1:")
a01 = list(y_pred_test).count(-1)
print(a01)

print("amount of target is  1 and prediction is also 1:")
a11 = list(y_pred_outliers).count(-1)
print(a11)

print("amount of target is  1 and prediction is 0:")
a10 = list(y_pred_outliers).count(1)
print(a10)

print("amount of normal test data")
anormal =y_pred_test.shape[0]
print(anormal)

print("amount of test outliers ")
aoutlier = y_pred_outliers.shape[0]
print(aoutlier)



print("accuracy is")
print((a00+a11)/(anormal+aoutlier))

print("precision of 1 is")
precision_1 = a11/(a11+a01)
print(precision_1)

print("recall of 1 is")
recall_1 = a11/(a11+a10)
print(recall_1)

print("f1score of 1 is")
print((2*recall_1*precision_1)/(recall_1 + precision_1))

print("precision of 0 is")
precision_0 = a00/(a10+a00)
print(precision_0)

print("recall of 0 is")
recall_0 = a00/(a01+a00)
print(recall_0)

print("f1score of 0 is")
print((2*recall_0*precision_0)/(recall_0 + precision_0))

inserts.append(
        dict(
            label="relevant5%",
            run_id=run_id,
            accuracy=float((a00+a11)/(anormal+aoutlier)),
            precision00=float(a00/(a10+a00)),
            recall00=float(a00/(a01+a00)),
            f1_score00=float((2*recall_0*precision_0)/(recall_0 + precision_0)),
            precision01=float(a11/(a11+a01)),
            recall01=float(a11/(a11+a10)),
            f1_score01=float((2*recall_1*precision_1)/(recall_1 + precision_1)
            )
        )
engine = create_engine('mysql+pymysql://'\
        + os.environ['MYSQL_USER'] + ":" \
        + os.environ['MYSQL_PASSWORD'] + "@" \
        + os.environ['MYSQL_IP'] + "/" \
        + os.environ['MYSQL_DB']
        )
conn = engine.connect()
metadata = MetaData()
metadata.reflect(engine)
algorithm_table = Table(os.environ["MYSQL_DB_TABLE"],metadata,autoload=True,autoload_with=engine)
result = conn.execute(algorithm_table.insert(), inserts)
