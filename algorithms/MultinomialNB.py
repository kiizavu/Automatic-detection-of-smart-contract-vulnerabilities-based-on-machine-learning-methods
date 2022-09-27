import numpy as np
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE  
from sklearn.metrics import classification_report
from joblib import dump
from sklearn.naive_bayes import MultinomialNB
from skmultilearn.problem_transform import BinaryRelevance


data_file='contract_labled.csv'
df = pd.read_csv(data_file)
df = df.fillna(0)
data = df.values

X = df.iloc[:,6:].values
y = df[['arithmetic', 'reentrancy', 'time_manipulation', 'TOD', 'tx_origin']].values

X_train, X_test, y_train, y_test = train_test_split(X, y,random_state = 0, test_size=0.2)

sm = SMOTE(random_state = 2)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

clf = BinaryRelevance(MultinomialNB())
clf.fit(X_train_res, y_train_res)
y_pred = clf.predict(X_test)

print('accuracy:', accuracy_score(y_test,y_pred))

print(classification_report(y_test, y_pred, zero_division=1))

dump(clf, 'model/MultinomialNB.joblib')