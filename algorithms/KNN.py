import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE  
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, accuracy_score
from joblib import dump
from sklearn import neighbors

for i in range(2):
    data_file = f"csv/contract_labled_{i}_1-gram.csv"
    df = pd.read_csv(data_file)
    df = df.fillna(0)
    data = df.values

    X = df.iloc[:,6:].values
    if i == 0:
        y = df[['arithmetic', 'clean', 'reentrancy', 'time_manipulation', 'TOD']].values
    else:
        y = df[['Label']]

    X_train, X_test, y_train, y_test = train_test_split(X, y,random_state = 0, test_size=0.2)
    
    sm = SMOTE(random_state = 2)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
    
    clf = neighbors.KNeighborsClassifier(n_neighbors = 7, p = 2,metric='euclidean')
    if i == 1:
        y_train_res = y_train_res.values.flatten()
    clf.fit(X_train_res, y_train_res)
    y_pred = clf.predict(X_test)

    print(f"----------------------{i}---------------------")
    print('accuracy:', accuracy_score(y_test,y_pred))

    print(classification_report(y_test, y_pred, zero_division=1))

    dump(clf, f"model/KNN_{i}.joblib")