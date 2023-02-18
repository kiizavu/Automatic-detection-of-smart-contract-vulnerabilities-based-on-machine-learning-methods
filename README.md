# Automatic detection of smart contract vulnerabilities based on machine learning methods
In this project, we apply KNN, AdaBoost, XGBoost, RandomForest, DNN to detect and SMOTE to balance dataset.

## How to run
1. Put vulnerable contracts to dataset/sourcecode
2. Run [prepare_dataset/source_to_bytecode.py](prepare_dataset/source_to_bytecode.py) to convert these contracts to bytecode
3. Run [prepare_dataset/extract_feature_label.py](prepare_dataset/extract_feature_label.py) to extract feature of contracts and label them to `contract_labled_x_y-gram.csv` in [csv directory](csv/).
4. Run all algorithm in algorithms directory to train model, the trained model will dump to model directory. File [main.py](algorithms/main.py) in algorithms directory will get scores of all algorithms we used and save them to [Training result.json](Training result.json) or [Training result no Smote.json](Training result no Smote.json).
5. Run [statistic.py](statistic.py) to compare scores between algorithms and extract figures to [figures directory](figures/).
6. Run [main.py](main.py) to detect vulnerable in specific contract.

## Conclusion
>- If you need high performance and don't worry about training time, you should use the **AdaBoost model** with __decimal label__.
>- If you need a model has short training time and don't worry about performance, you should use the **KNN model** with __binary label__.
>- If you need a model which has both high performance and short training time, we suggest to use the **Random Forest model** with __decimal label__.