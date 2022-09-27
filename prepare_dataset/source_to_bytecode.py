import os
'''
dataset
    ├──arithmetic
    │    ├──contract1.sol
    │    ├──contract2.sol
    │    ├──contract3.sol
    │    └──...
    ├──reentrancy
    │    ├──contract1.sol
    │    ├──contract2.sol
    │    ├──contract3.sol
    │    └──...
    ├──time_manipulation
    │    ├──contract1.sol
    │    ├──contract2.sol
    │    ├──contract3.sol
    │    └──...
    ├──TOD
    │    ├──contract1.sol
    │    ├──contract2.sol
    │    ├──contract3.sol
    │    └──...
    ├──tx_origin
    │    ├──contract1.sol
    │    ├──contract2.sol
    │    ├──contract3.sol
    │    └──...
'''
dataset_sourcecode = 'dataset/sourcecode'           # folder chua source code cua dataset
dataset_bytecode = 'dataset/bytecode'               # folder chua bytecode

# ham doc version cua contract
def getVersionPragma(filename):                     # param la path cua contract          
    file = open(filename, 'r')
    data = file.readlines()
    for line in data:                               # duyet tung dong trong contract do
        if 'pragma' in line:                        # neu dong do chua chu "pragma"
            temp = line.split()                     # chuyen dong do thanh list ['pragma', 'solidity', '^0.4.19;']
            if len(temp) == 3 and temp[2][0].isnumeric() == True:       # ['pragma', 'solidity', '0.4.19;']
                return temp[2][0:-1]
            elif len(temp) == 3 and temp[2][1].isnumeric() == True:       # ['pragma', 'solidity', '^0.4.19;']
                return temp[2][1:-1]
            elif len(temp) == 3 and temp[2][1].isnumeric() == False:    # ['pragma', 'solidity', '<=0.4.19;']
                return temp[2][2:-1]
            elif len(temp) > 3 and temp[2][1].isnumeric() == True:      # ['pragma', 'solidity', '>0.4.22', '<0.6.0]
                return temp[2][1:]
            elif len(temp) > 3 and temp[2][1].isnumeric() == False:     # ['pragma', 'solidity', '>=0.4.22', '<0.6.0]
                return temp[2][2:]
    return '0.4.22'

vulns = os.listdir(dataset_sourcecode)              # cac folder loi trong source code dataset

if 'dataset/bytecode' not in os.listdir():          # tao folder 'dataset_bytecode' neu chua co
    os.mkdir(dataset_bytecode)

for vuln in vulns:
    vuln_dir_source = os.path.join(dataset_sourcecode, vuln)    # path cua folder loi source code
    vuln_dir_byte = os.path.join(dataset_bytecode, vuln)        # path cua folder loi byte code
    if vuln not in os.listdir(dataset_bytecode):                # tao folder loi byte code neu chua co
        os.mkdir(vuln_dir_byte)
    files = os.listdir(vuln_dir_source)                         # list cac contract trong folder loi source code
    for file in files:
        if '.SOL' in file or '.sol' in file:                    # neu la contract
            contract_source_path = os.path.join(vuln_dir_source, file)              # path cua contract do o folder source code
            contract_byte_path = os.path.join(vuln_dir_byte, file + '.txt')         # file byte code chuan bi tao
            contract_version = getVersionPragma(contract_source_path)               # lay version cua contract

            cmd = 'solc-select use ' + contract_version                             # dung lenh "solc-select use" de chuyen version cho dung voi contract
            os.system(cmd)
            cmd1 = 'solc --bin ' + contract_source_path + '>' + contract_byte_path  # xuat ra bytecode cua contract roi ghi vao file
            os.system(cmd1)
            if os.path.getsize(contract_byte_path) == 0:                            # neu co loi luc xuat bytecode thi size cua file ghi ra la 0
                os.remove(contract_byte_path)                                       # xoa file do di