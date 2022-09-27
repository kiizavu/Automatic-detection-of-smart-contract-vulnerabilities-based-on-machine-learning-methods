ORIGIN = 1

import numpy as np
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE  
from sklearn.metrics import classification_report
from joblib import dump
import xgboost as xgb
from sklearn.multiclass import OneVsRestClassifier

data_file='contract_labled.csv'
df = pd.read_csv(data_file)
df = df.fillna(0)
data = df.values

X = df.iloc[:,6:].values
y = df[['arithmetic', 'reentrancy', 'time_manipulation', 'TOD', 'tx_origin']].values

if not ORIGIN:
    y_tmp = []
    for row in range(y.shape[0]):
        for col in range(y.shape[1]):
            if y[row,col] == 1:
                y_tmp.append(col + 1)
                continue
    y = np.array(y_tmp)

X_train, X_test, y_train, y_test = train_test_split(X, y,random_state = 0, test_size=0.2)

sm = SMOTE(random_state = 2)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

clf = OneVsRestClassifier(xgb.XGBRFClassifier(n_estimators=100, random_state=0))
clf.fit(X_train_res, y_train_res)
y_pred = clf.predict(X_test)

for i in range(len(y_pred)):
    print(f"{list(y_pred)[i]:<25}{list(y_test)[i]}")

print('accuracy:', accuracy_score(y_test,y_pred))

print(classification_report(y_test, y_pred, zero_division=1))

if not ORIGIN:
    dump(clf, 'model/XGBoost.joblib')
else:
    dump(clf, 'model/XGBoost_origin.joblib')