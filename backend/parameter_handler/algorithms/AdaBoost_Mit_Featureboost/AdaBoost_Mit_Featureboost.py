import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn import metrics
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier as adaBoost
from xgboost import XGBClassifier
from sklearn.feature_selection import SelectFromModel
from sqlalchemy import Table, MetaData, create_engine
import os
import sys
import json

inserts = []

DATASET = json.loads(sys.argv[1])
run_id = sys.argv[2]

for key in DATASET:
    DATASET[key] = [DATASET[key]]

param_dist = {'algorithm':["SAMME","SAMME.R"]}
param_dist.update(DATASET)
print(param_dist)

data = pd.read_csv('csv/testdateimitlabels.csv',low_memory=False)

col_names = data.columns
col_list = col_names.tolist()
# Get all keys except years, rics and labels(all, relevant, relevant5%)
keys_X = []
for x in range(6,839):
    keys_X.append(col_list[x])

# Get all data except years, rics and labels
X = data[keys_X]

# Delete all boolean type or object type(datum) features
X = X.select_dtypes(exclude=['bool','object'])

# Replace all the NAN with mean
imp = SimpleImputer(strategy="mean")
X = imp.fit_transform(X)

# Transform the data as data frame
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

y1.info()
print(y1)
#split training set and test set, The ratio is 4: 1
X_train, X_test, y_train, y_test = train_test_split(select_X, y1, test_size=0.2, random_state=0)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

#cross validation and grid search for hyperparameter estimation

#param_dist = {'algorithm':["SAMME","SAMME.R"]}

cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
clf = GridSearchCV(adaBoost(),param_grid=param_dist, cv=cv)
clf = clf.fit(X_train, y_train.values.ravel())

print("Best estimator found by grid search:")
print(clf.best_estimator_)

#apply the classifier on the test data and show the accuracy of the model
print('the acuracy for all is:')
print(clf.score(X_test, y_test.values.ravel()))
prediction = clf.predict(X_test)

#use the metrics.classification to report.
print("Confusion matrix:\n%s" % metrics.confusion_matrix(y_test, prediction))
print("Classification report:\n %s\n"% metrics.classification_report(y_test, prediction,output_dict=True))
classification_report_dict = metrics.classification_report(y_test, prediction,output_dict=True)
inserts.append(
        dict(
            label="all",
            run_id=run_id,
            accuracy=float(clf.score(X_test, y_test.values.ravel())),
            precision00=float(classification_report_dict["0"]["precision"]),
            recall00=float(classification_report_dict["0"]["recall"]),
            f1_score00=float(classification_report_dict["0"]["f1-score"]),
            support00=classification_report_dict["0"]["support"],
            precision01=float(classification_report_dict["1"]["precision"]),
            recall01=float(classification_report_dict["1"]["recall"]),
            f1_score01=float(classification_report_dict["1"]["f1-score"]),
            support01=classification_report_dict["1"]["support"],
            macro_avg_precision=float(classification_report_dict["macro avg"]["precision"]),
            macro_avg_recall=float(classification_report_dict["macro avg"]["recall"]),
            macro_avg_f1_score=float(classification_report_dict["macro avg"]["f1-score"]),
            macro_avg_support=classification_report_dict["macro avg"]["support"],
            weighted_avg_precision=float(classification_report_dict["weighted avg"]["precision"]),
            weighted_avg_recall=float(classification_report_dict["weighted avg"]["recall"]),
            weighted_avg_f1_score=float(classification_report_dict["weighted avg"]["f1-score"]),
            weighted_avg_support=classification_report_dict["weighted avg"]["support"]
            

            )
        )

            

########################################################################################################################
#     prediction for label "relevant"
########################################################################################################################

#Using XGBoost for featureboost, each feature get a score.
#Using SelectFromModel, we can extract features, whose score is higher than average score.
model = XGBClassifier()
model.fit(X, y2.values.ravel())
print('feature_importances of all')
print(model.feature_importances_)
selection = SelectFromModel(model,prefit=True,threshold="10*mean")
select_X = selection.transform(X)
select_X = pd.DataFrame(select_X)
print(select_X)

y2.info()
print(y2)
#split training data and test data, The ratio is 4: 1
X_train, X_test, y_train, y_test = train_test_split(select_X, y2, test_size=0.2, random_state=0)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

#cross validation and grid search for hyperparameter estimation

#param_dist = {'algorithm':["SAMME","SAMME.R"]}

cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
clf = GridSearchCV(adaBoost(),param_grid=param_dist, cv=cv)
clf = clf.fit(X_train, y_train.values.ravel())

print("Best estimator found by grid search:")
print(clf.best_estimator_)

#apply the classifier on the test data and show the accuracy of the model
print('the acuracy for relevant is:')
print(clf.score(X_test, y_test.values.ravel()))

prediction = clf.predict(X_test)
#use the metrics.classification to report.
print("Confusion matrix:\n%s" % metrics.confusion_matrix(y_test, prediction))
print("Classification report:\n %s\n"% metrics.classification_report(y_test, prediction))

classification_report_dict = metrics.classification_report(y_test, prediction,output_dict=True)

inserts.append(
        dict(
            label="relevant",
            run_id=run_id,
            accuracy=float(clf.score(X_test, y_test.values.ravel())),
            precision00=float(classification_report_dict["0"]["precision"]),
            recall00=float(classification_report_dict["0"]["recall"]),
            f1_score00=float(classification_report_dict["0"]["f1-score"]),
            support00=classification_report_dict["0"]["support"],
            precision01=float(classification_report_dict["1"]["precision"]),
            recall01=float(classification_report_dict["1"]["recall"]),
            f1_score01=float(classification_report_dict["1"]["f1-score"]),
            support01=classification_report_dict["1"]["support"],
            macro_avg_precision=float(classification_report_dict["macro avg"]["precision"]),
            macro_avg_recall=float(classification_report_dict["macro avg"]["recall"]),
            macro_avg_f1_score=float(classification_report_dict["macro avg"]["f1-score"]),
            macro_avg_support=classification_report_dict["macro avg"]["support"],
            weighted_avg_precision=float(classification_report_dict["weighted avg"]["precision"]),
            weighted_avg_recall=float(classification_report_dict["weighted avg"]["recall"]),
            weighted_avg_f1_score=float(classification_report_dict["weighted avg"]["f1-score"]),
            weighted_avg_support=classification_report_dict["weighted avg"]["support"]
            

            )
        )
########################################################################################################################
#     prediction for label"relevant5%"
########################################################################################################################

#Using XGBoost for featureboost, each feature get a score.
#Using SelectFromModel, we can extract features, whose score is higher than average score.
model = XGBClassifier()
model.fit(X, y3.values.ravel())
print('feature_importances of all')
print(model.feature_importances_)
selection = SelectFromModel(model,prefit=True,threshold="10*mean")
select_X = selection.transform(X)
select_X = pd.DataFrame(select_X)
print(select_X)

y3.info()
print(y3)
#split training data and test data, The ratio is 4: 1
X_train, X_test, y_train, y_test = train_test_split(select_X, y3, test_size=0.2, random_state=0)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

#cross validation and grid search for hyperparameter estimation

#param_dist = {'algorithm':["SAMME","SAMME.R"]}

cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
clf = GridSearchCV(adaBoost(),param_grid=param_dist, cv=cv)
clf = clf.fit(X_train, y_train.values.ravel())

print("Best estimator found by grid search:")
print(clf.best_estimator_)

#apply the classifier on the test data and show the accuracy of the model
print('the acuracy for relevant5% is:')
print(clf.score(X_test, y_test.values.ravel()))

prediction = clf.predict(X_test)
#use the metrics.classification to report
print("Confusion matrix:\n%s" % metrics.confusion_matrix(y_test, prediction))
print("Classification report:\n %s\n"% metrics.classification_report(y_test, prediction))

classification_report_dict = metrics.classification_report(y_test, prediction,output_dict=True)

inserts.append(
        dict(
            run_id=run_id,
            label="relevant5%",
            accuracy=float(clf.score(X_test, y_test.values.ravel())),
            precision00=float(classification_report_dict["0"]["precision"]),
            recall00=float(classification_report_dict["0"]["recall"]),
            f1_score00=float(classification_report_dict["0"]["f1-score"]),
            support00=classification_report_dict["0"]["support"],
            precision01=float(classification_report_dict["1"]["precision"]),
            recall01=float(classification_report_dict["1"]["recall"]),
            f1_score01=float(classification_report_dict["1"]["f1-score"]),
            support01=classification_report_dict["1"]["support"],
            macro_avg_precision=float(classification_report_dict["macro avg"]["precision"]),
            macro_avg_recall=float(classification_report_dict["macro avg"]["recall"]),
            macro_avg_f1_score=float(classification_report_dict["macro avg"]["f1-score"]),
            macro_avg_support=classification_report_dict["macro avg"]["support"],
            weighted_avg_precision=float(classification_report_dict["weighted avg"]["precision"]),
            weighted_avg_recall=float(classification_report_dict["weighted avg"]["recall"]),
            weighted_avg_f1_score=float(classification_report_dict["weighted avg"]["f1-score"]),
            weighted_avg_support=classification_report_dict["weighted avg"]["support"]
            

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
