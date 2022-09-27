#Automatic detection of smart contract vulnerabilities based on machine learning methods
1. Put vulnerable contracts to dataset/sourcecode
2. Run prepare_dataset/source_to_bytecode.py to convert these contracts to bytecode
3. Run prepare_dataset/extract_feature_label.py to extract feature of contracts and label them to `contract_labled.csv`.
4. Run all algorithm in algorithms directory to train model, the trained model will dump to model directory
5. Run main.py to detect vulnerable in specific contract 
