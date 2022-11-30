import binascii
from pyevmasm import disassemble_all 
import os
import collections
from nltk import ngrams 
import csv
import re

#ngram = 1
label = {'clean': 0, 'arithmetic': 1, 'reentrancy': 2, 'time_manipulation': 3, 'TOD': 4}
regex = r"Binary of the runtime part: \s(.*)"

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


dataset_bytecode = 'dataset/bytecode'       # folder chua bytecode
vulns = os.listdir(dataset_bytecode)        # cac loi trong folder do

dataset = dict.fromkeys(vulns)              # luu loai lo hong, dia chi, opcode - {vuln: {address: opcode}}

### Doc file bytecode cua tung contract sau do chuyen thanh opcode roi luu vao bien dataset
for vuln in vulns:
    vuln_dir_byte = os.path.join(dataset_bytecode, vuln)                # lay path cua folder lo hong
    contracts = os.listdir(vuln_dir_byte)                               # lay contract trong folder do
    contracts_dict = {}                                                 # luu dia chi, opcode - {address: opcode}
    for contract in contracts:
        with open(os.path.join(vuln_dir_byte, contract), "r") as f:     # doc tung contract
            content = f.read()

        bytecode = re.findall(regex, content, re.MULTILINE)
        
        bincontent = ''.join(bytecode)                                   # noi tat ca bytecode trong contract
        try:
            rawbytes = binascii.unhexlify(bincontent)                   # chuyen sang dang hex, neu loi thi bo qua, vi co the co loi khi chuyen tu source code sang bytecode
        except:
            continue

        opcodes = disassemble_all(rawbytes)                             # chuyen tu bytecode sang opcode

        tmp = ""
        for opcode in opcodes:
            tmp += simplified_rule(str(opcode).split()[0]) + " "        # simplify opcode va bo 0x trong opcode
        tmp = tmp[:-1]

        contracts_dict[contract] = tmp                                  # them vao dict contracts_dict {address1: opcode1, address2: opcode2, ...}
    dataset[vuln] = contracts_dict                                      # them vao dict dataset {vuln1: {address1: opcode1, address2: opcode2, ...}, vuln2: {address1: opcode1, address2: opcode2, ...}, ...}


for i in range(1, 11):
    for version_label in range(2):                     # version 0: [0,1,0,0,0] version 1:[1]
        fieldnames = ['Address']                       # ten cot trong file csv
        if version_label:
            fieldnames.append('Label')
        else:
            fieldnames.extend(dataset.keys())          # them ten cac lo hong vao cot de gan nhan
        rows = []                                      # moi hang la 1 contract cung tan suat xuat hien cac bigram cua no
        # bigramPattern = [p for p in itertools.product(all_opcode, repeat=ngram)]
        # fieldnames.extend(bigramPattern)

        # feature extraction bằng n-gram 
        for vuln in dataset.keys():                         # lay ten lo hong
            contracts_addr = dataset[vuln].keys()           # lay cac dia chia chi contract
            for addr in contracts_addr:                     # duyet tung dia chi
                tmp = ''.join(dataset[vuln][addr])          # bien chua opcode dang string
                n_grams = ngrams(tmp.split(), i)            # trich xuat bigram
                count_bigram = collections.Counter(n_grams) # dem tan suat xuat hien

                row = {}                                    # chua tung contract va bigram cua no, la 1 phan tu trong rows, tuong ung 1 hang trong csv
                row.update({"Address": addr[:-8]})          # lay dia chi contract roi them vao cot Address

                for grams in count_bigram:
                    if grams not in fieldnames:
                        fieldnames.append(grams)
                    row.update({grams: count_bigram[grams]})# cap nhat grams, tan suat xuat hien cua bigram 

                if version_label:
                    row.update({'Label': label[vuln]})
                else:
                    row.update({vuln: '1'})                     # gan nhan cho contract
                rows.append(row)

        # ghi vao file csv
        with open(f"csv\contract_labled_{version_label}_{i}-gram.csv" , 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames = fieldnames)
            writer.writeheader()        # ghi fieldnames
            writer.writerows(rows)      # ghi tung phan tu cua list rows vao hang trong csv

