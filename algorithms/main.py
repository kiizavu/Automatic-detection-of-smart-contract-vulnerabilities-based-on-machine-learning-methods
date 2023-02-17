import time
import json
import pandas as pd
import xgboost as xgb
from sklearn import neighbors, svm
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble  import RandomForestClassifier

def save_score(y_test, y_pred, time_end, time_start):
    score = dict()
    score['accuracy'] = accuracy_score(y_test,y_pred)
    score['precision'] = precision_score(y_test, y_pred, average='macro' ,zero_division=1)
    score['recall'] = recall_score(y_test, y_pred, average='macro' ,zero_division=1)
    score['f1'] = f1_score(y_test, y_pred, average='macro' ,zero_division=1)
    score['train time'] = time_end - time_start
    return score

# có cân bằng dữ liệu ko
is_Smote = False

result = dict()
for i in range(1, 11):
    dataset = dict()
    print(f'{i}-gram')
    for j in range(2):
        data_file=f"csv/contract_labled_{j}_{i}-gram.csv"
        df = pd.read_csv(data_file)
        df = df.fillna(0)\

        X = df.iloc[:,6:].values
        if j == 0:
            y = df[['arithmetic', 'clean', 'reentrancy', 'time_manipulation', 'TOD']].values
        else:
            y = df[['Label']]

        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0, test_size=0.2)

        if is_Smote:
            sm = SMOTE(random_state = 2)
            X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
        else:
            X_train_res, y_train_res = X_train, y_train
        algorithms = dict()

        #RF
        rf = OneVsRestClassifier(RandomForestClassifier(n_estimators=100, random_state=0))
        rf_start = time.time()
        rf.fit(X_train_res, y_train_res)
        rf_end = time.time()
        y_pred_rf = rf.predict(X_test)

        #KNN
        knn = OneVsRestClassifier(neighbors.KNeighborsClassifier(n_neighbors = 7, p = 2,metric='euclidean'))
        if j == 1:
            y_train_res_knn = y_train_res.values.flatten()
            knn_start = time.time()
            knn.fit(X_train_res, y_train_res_knn)
            knn_end = time.time()
        else:
            knn_start = time.time()
            knn.fit(X_train_res, y_train_res)
            knn_end = time.time()
        y_pred_knn = knn.predict(X_test)

        #AdaBoost
        ada = OneVsRestClassifier(AdaBoostClassifier(n_estimators=100, random_state=0))
        ada_start = time.time()
        ada.fit(X_train_res, y_train_res)
        ada_end = time.time()
        y_pred_ada = ada.predict(X_test)

        #XGBoost
        xg = OneVsRestClassifier(xgb.XGBRFClassifier(n_estimators=100, random_state=0))
        xgb_start = time.time()
        xg.fit(X_train_res, y_train_res)
        xgb_end = time.time()
        y_pred_xgb = xg.predict(X_test)

        algorithms['Random Forest'] = save_score(y_test, y_pred_rf, rf_end, rf_start)
        algorithms['KNN'] = save_score(y_test, y_pred_knn, knn_end, knn_start)
        algorithms['AdaBoost'] = save_score(y_test, y_pred_ada, ada_end, ada_start)
        algorithms['XGBoost'] = save_score(y_test, y_pred_xgb, xgb_end, xgb_start)

        dataset[f'dataset {j}'] = algorithms
        
    result[f'{i}-gram'] = dataset

if is_Smote:
    result_file_name = "Training result Smote.json"
else:
    result_file_name = "Training result no Smote.json"

with open(result_file_name, "w") as outfile:
    json.dump(result, outfile)