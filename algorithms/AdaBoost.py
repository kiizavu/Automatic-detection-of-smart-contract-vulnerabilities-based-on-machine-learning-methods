from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE  
from sklearn.metrics import classification_report
from joblib import dump
from sklearn.ensemble import AdaBoostClassifier
from sklearn.multiclass import OneVsRestClassifier

for i in range(2):
    data_file = f"contract_labled_{i}.csv"
    df = pd.read_csv(data_file)
    df = df.fillna(0)

    X = df.iloc[:,6:].values
    if i == 0:
        y = df[['arithmetic', 'clean', 'reentrancy', 'time_manipulation', 'TOD']].values
    else:
        y = df[['Label']]

    sm = SMOTE(random_state = 2)
    X_res, y_res = sm.fit_resample(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res,random_state = 0, test_size=0.2)


    clf = OneVsRestClassifier(AdaBoostClassifier(n_estimators=100, random_state=0))
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print(f"----------------------{i}---------------------")
    print('accuracy:', accuracy_score(y_test,y_pred))

    print(classification_report(y_test, y_pred, zero_division=1))

    dump(clf, f"model/AdaBoost_{i}.joblib")