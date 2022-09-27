import time
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn import neighbors, svm
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from skmultilearn.problem_transform import BinaryRelevance

def score(y_test, y_pred):
    print(accuracy_score(y_test,y_pred))
    print(classification_report(y_test, y_pred))

data_file='contract_labled.csv'
df = pd.read_csv(data_file)
df = df.fillna(0)
data = df.values

X = df.iloc[:,6:].values
y = df[['arithmetic', 'reentrancy', 'time_manipulation', 'TOD', 'tx_origin']].values

X_train, X_test, y_train, y_test = train_test_split(X, y,random_state = 0, test_size=0.2)

sm = SMOTE(random_state = 2)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)



y_tmp = []
for row in range(y.shape[0]):
    for col in range(y.shape[1]):
        if y[row,col] == 1:
            y_tmp.append(col + 1)
            continue
y_c = np.array(y_tmp)

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y_c,random_state = 0, test_size=0.2)

X_train_res_c, y_train_res_c = sm.fit_resample(X_train_c, y_train_c)

#KNN
knn = neighbors.KNeighborsClassifier(n_neighbors = 7, p = 2,metric='euclidean')
knn_start = time.time()
knn.fit(X_train_res, y_train_res)
knn_end = time.time()
y_pred_knn = knn.predict(X_test)

knn_start1 = time.time()
knn.fit(X_train_res_c, y_train_res_c)
knn_end1 = time.time()
y_pred_knn1 = knn.predict(X_test_c)

#AdaBoost
ada = OneVsRestClassifier(AdaBoostClassifier(n_estimators=100, random_state=0))
ada_start = time.time()
ada.fit(X_train_res, y_train_res)
ada_end = time.time()
y_pred_ada = ada.predict(X_test)

ada_start1 = time.time()
ada.fit(X_train_res_c, y_train_res_c)
ada_end1 = time.time()
y_pred_ada1 = ada.predict(X_test_c)

#XGBoost
xg = OneVsRestClassifier(xgb.XGBRFClassifier(n_estimators=100, random_state=0))
xgb_start = time.time()
xg.fit(X_train_res, y_train_res)
xgb_end = time.time()
y_pred_xgb = xg.predict(X_test)

xgb_start1 = time.time()
xg.fit(X_train_res_c, y_train_res_c)
xgb_end1 = time.time()
y_pred_xgb1 = xg.predict(X_test_c)

#MultinomialNB
clf = BinaryRelevance(MultinomialNB())
clf_start = time.time()
clf.fit(X_train_res, y_train_res)
clf_end = time.time()
y_pred_clf = clf.predict(X_test)

#SVM
clf2 = BinaryRelevance(svm.SVC())
clf2_start = time.time()
clf2.fit(X_train_res, y_train_res)
clf2_end = time.time()
y_pred_clf2 = clf2.predict(X_test)

print(f"{'Algorithms':<25} {'accuracy_score':<25} Training time")
print(f"{'KNN':<25} {accuracy_score(y_test,y_pred_knn):<25} {knn_end - knn_start}")
print(f"{'KNN_1':<25} {accuracy_score(y_test_c,y_pred_knn1):<25} {knn_end1 - knn_start1}")

print(f"{'AdaBoost':<25} {accuracy_score(y_test,y_pred_ada):<25} {ada_end - ada_start}")
print(f"{'AdaBoost_1':<25} {accuracy_score(y_test_c,y_pred_ada1):<25} {ada_end1 - ada_start1}")

print(f"{'XGBoost':<25} {accuracy_score(y_test,y_pred_xgb):<25} {xgb_end - xgb_start}")
print(f"{'XGBoost_1':<25} {accuracy_score(y_test_c,y_pred_xgb1):<25} {xgb_end1 - xgb_start1}")

print(f"{'MultinomialNB':<25} {accuracy_score(y_test,y_pred_clf):<25} {clf_end - clf_start}")
print(f"{'SVM':<25} {accuracy_score(y_test,y_pred_clf2):<25} {clf2_end - clf2_start}")

print('\n\n\nKNN:', accuracy_score(y_test,y_pred_knn))
print(classification_report(y_test, y_pred_knn, zero_division=1))
print('\n\n\nKNN:', accuracy_score(y_test_c,y_pred_knn1))
print(classification_report(y_test_c, y_pred_knn1, zero_division=1))

print('\n\n\nAdaBoost:', accuracy_score(y_test,y_pred_ada))
print(classification_report(y_test, y_pred_ada, zero_division=1))
print('\n\n\nAdaBoost:', accuracy_score(y_test_c,y_pred_ada1))
print(classification_report(y_test_c, y_pred_ada1, zero_division=1))

print('\n\n\nXGBoost:', accuracy_score(y_test,y_pred_xgb))
print(classification_report(y_test, y_pred_xgb, zero_division=1))
print('\n\n\nXGBoost:', accuracy_score(y_test_c,y_pred_xgb1))
print(classification_report(y_test_c, y_pred_xgb1, zero_division=1))

print('\n\n\nMultinomialNB:', accuracy_score(y_test,y_pred_clf))
print(classification_report(y_test, y_pred_clf, zero_division=1))

print('\n\n\nSVM:', accuracy_score(y_test,y_pred_clf2))
print(classification_report(y_test, y_pred_clf2, zero_division=1))