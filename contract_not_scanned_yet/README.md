The `contract_before_2017.json` is get on [Google Bigquery](https://console.cloud.google.com/bigquery?page=table&t=contracts&d=crypto_ethereum&p=bigquery-public-data) with script
```SQL
SELECT contracts.address, contracts.bytecode,contracts.block_timestamp
FROM `bigquery-public-data.crypto_ethereum.contracts` AS contracts
JOIN `bigquery-public-data.crypto_ethereum.transactions` AS transactions ON (transactions.to_address = contracts.address)
where DATE(contracts.block_timestamp) <= "2016-12-31"
GROUP BY contracts.address, contracts.bytecode, contracts.block_timestamp
HAVING count(contracts.address) > 1
ORDER BY contracts.address ASC
```
Then download result as JSONL and extract it into separate file in folder `contract_before_2017`
