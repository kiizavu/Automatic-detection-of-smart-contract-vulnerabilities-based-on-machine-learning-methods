import numpy as np  
from solcx import compile_standard, install_solc
import collections
from nltk import ngrams 
import itertools
import xgboost as xgb
from joblib import load


ARITHMETIC_OP = ['ADD', 'MUL', 'SUB', 'DIV', 'SDIV', 'SMOD', 'MOD', 'ADDMOD', 'MULMOD', 'EXP']
CONSTANT1 = ['BLOCKHASH', 'TIMESTAMP', 'NUMBER', 'DIFFICULTY', 'GASLIMIT', 'COINBASE']
CONSTANT2 = ['ADDRESS', 'ORIGIN', 'CALLER']
COMPARASION = ['LT', 'GT', 'SLT', 'SGT']
LOGIC_OP = ['AND', 'OR', 'XOR', 'NOT']
DUP = ['DUP1', 'DUP2', 'DUP3', 'DUP4', 'DUP5', 'DUP6', 'DUP7','DUP8', 'DUP9','DUP10','DUP11','DUP12','DUP13','DUP14', 'DUP15','DUP16']
SWAP = ['SWAP1', 'SWAP2','SWAP3','SWAP4','SWAP5','SWAP6','SWAP7','SWAP8','SWAP9','SWAP10','SWAP11','SWAP12','SWAP13','SWAP14','SWAP15','SWAP16']
PUSH = ['PUSH5', 'PUSH6', 'PUSH7', 'PUSH8', 'PUSH9', 'PUSH10', 'PUSH11', 'PUSH12', 'PUSH13', 'PUSH14', 'PUSH15', 'PUSH16', 'PUSH17', 'PUSH18', 'PUSH19', 'PUSH20', 'PUSH21', 'PUSH22', 'PUSH23', 'PUSH24', 'PUSH25', 'PUSH26', 'PUSH27', 'PUSH28', 'PUSH29', 'PUSH30', 'PUSH31', 'PUSH32']
LOG = ['LOG1', 'LOG2', 'LOG3', 'LOG4']
all_opcode = ['KECCAK256', 'GETPC', 'LOG0', 'INVALID', 'EQ', 'STOP', 'ARITHMETIC_OP', 'SIGNEXTEND', 'COMPARASION', 'ISZERO', 'LOGIC_OP', 'BYTE', 'SHL', 'SHR', 'SAR', 'SHA3', 'CONSTANT2', 'BALANCE', 'CALLVALUE', 'CALLDATALOAD', 'CALLDATASIZE', 'CALLDATACOPY', 'CODESIZE', 'CODECOPY', 'GASPRICE', 'EXTCODESIZE', 'EXTCODECOPY', 'RETURNDATASIZE', 'RETURNDATACOPY', 'EXTCODEHASH', 'CONSTANT1', 'CHAINID', 'SELFBALANCE', 'BASEFEE', 'POP', 'MLOAD', 'MSTORE', 'MSTORE8', 'SLOAD', 'SSTORE', 'JUMP', 'JUMPI', 'PC', 'MSIZE', 'GAS', 'JUMPDEST', 'PUSH1', 'PUSH2', 'PUSH3', 'PUSH4', 'PUSH', 'DUP', 'SWAP', 'LOG', 'CREATE', 'CALL', 'CALLCODE', 'RETURN', 'DELEGATECALL', 'CREATE2', 'STATICCALL', 'REVERT', 'SELFDESTRUCT']

def getOpcode(filename):
    with open(filename, 'r') as file:                   # doc file sol
        file_content = file.read()

    # ham doc version cua contract
    def getVersionPragma(file_content):                     # param la path cua contract          
        for line in file_content:                               # duyet tung dong trong contract do
            if 'pragma' in line:                        # neu dong do chua chu "pragma"
                temp = line.split()                     # chuyen dong do thanh list ['pragma', 'solidity', '^0.4.19;']
                if len(temp) == 3 and temp[2][0].isnumeric() == True:       # ['pragma', 'solidity', '0.4.19;']
                    return temp[2][0:-1]
                elif len(temp) == 3 and temp[2][1].isnumeric() == True:       # ['pragma', 'solidity', '^0.4.19;']
                    return temp[2][1:-1]
                elif len(temp) == 3 and temp[2][1].isnumeric() == False:    # ['pragma', 'solidity', '<=0.4.19;']
                    return temp[2][2:-1]
                elif len(temp) > 3 and temp[2][1].isnumeric() == True:      # ['pragma', 'solidity', '>0.4.22', '<0.6.0]
                    return temp[2][2:]
                elif len(temp) > 3 and temp[2][1].isnumeric() == False:     # ['pragma', 'solidity', '>=0.4.22', '<0.6.0]
                    return temp[2][2:]

    # Ham simplify opcode de giam so luong cot khi train
    def simplified_rule(opcode):
        if opcode in ARITHMETIC_OP:
            opcode = 'ARITHMETIC_OP'
        elif opcode in CONSTANT1:
            opcode = 'CONSTANT1'
        elif opcode in CONSTANT2:
            opcode = 'CONSTANT2'
        elif opcode in COMPARASION:
            opcode = 'COMPARASION'
        elif opcode in LOGIC_OP:
            opcode = 'LOGIC_OP'
        elif opcode in DUP:
            opcode = 'DUP'
        elif opcode in SWAP:
            opcode = 'SWAP'
        elif opcode in PUSH:
            opcode = 'PUSH'
        elif opcode in LOG:
            opcode = 'LOG'
        return opcode


    version = getVersionPragma(file_content.split('\n'))
    
    install_solc(version)
    compiled_sol = compile_standard(
        {
            "language": "Solidity",
            "sources": {filename: {"content": file_content}},
            "settings": {
                "outputSelection": {
                    "*": {
                        "*": ["evm.bytecode.opcodes"]
                    }
                }
            },
        },
        solc_version=version,
    )

    list_opcodes = []
    for contract_name in compiled_sol['contracts'][filename].keys():
        cc = compiled_sol['contracts'][filename][contract_name]["evm"]["bytecode"]["opcodes"]
        list_opcode = [simplified_rule(opcode) for opcode in cc.split() if opcode[0] != '0']
        list_opcodes.extend(list_opcode)
    return list_opcodes

opcode = getOpcode("test.sol")

bigramPattern = [p for p in itertools.product(all_opcode, repeat=2)]
bigram_dict = dict.fromkeys(bigramPattern, 0)


n_grams = ngrams(opcode, 2)
count_bigram = collections.Counter(n_grams)
for grams in count_bigram:
    bigram_dict[grams] = count_bigram[grams]

cc = [bigram_dict[key] for key in bigram_dict.keys()]

# model = 'model/XGBoost.json'
# clf = xgb.XGBRFClassifier()
# clf.load_model(model)
model = 'model/AdaBoost.joblib'
clf = load(model)

pred = clf.predict(np.expand_dims(cc, axis=0))

print(pred)
# vulns = {
#     0: 'arithmetic',
#     1: 'reentrancy',
#     2: 'time_manipulation',
#     3: 'TOD',
#     4: 'tx_origin'
# }

# result = list(pred[0])
# k = 0
# for i in range(len(result)):
#     if result[i] != 0:
#         print(vulns[i], 'found in contract')
#         k += 1
# if k == 0:
#     print('Not found vulnerable in this contract')