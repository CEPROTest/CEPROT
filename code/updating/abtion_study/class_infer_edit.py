tpc_id = []
fpc_id = []

from unittest import result
import jsonlines
from tqdm import tqdm
from typing import List
import json

def writejson(data,filename):
    d = []
    with open(filename,'w') as f:
        for item in tqdm(data):
            d.append(item)
        json.dump(data,f)
    f.close()

def readjson_all(filename):
    d = []
    with open(filename, mode='r', encoding='utf-8') as f:
        dicts = json.load(f)
    for i in dicts:
        d.append(i)
    return d

def readjs(filename):
    l=[]
    with open(filename, "r+", encoding="utf-8") as f:
        for item in jsonlines.Reader(f):
            l.append(item)
        return l

def writejs(data,filenamewrite): 
    print(str(len(data)) + ' pieces of data will be written in ',filenamewrite)
    with jsonlines.open(filenamewrite, mode='w') as writer:
        for i in tqdm(range(len(data))):
            writer.write(data[i])
    writer.close()
    print('Write Jsonl File Done! Path is '+filenamewrite)

def readtxt(filename):
    t = []
    with open(filename) as f:
        data = f.readlines()
    for i in data:
        t.append(i.strip())
    return t

def infer_TPC():
    tpc_hypo_data = []
    tpc_ref_data = []
    test_data_c = readjson_all('./class/test.json')
    test_data_g = readjson_all('./gen/test.json')
    
    hypo_data = readtxt('output')
    ref_data = readtxt('gold')
    tpc_cnt = 0
    cnt = 0

    tpc_data = []
    for i in tpc_id:
        tpc_data.append(test_data_c[i])
    print(len(tpc_data))
    for i in range(len(tpc_data)):
        flag = 1
        tmp = tpc_data[i]['edit_seq'] 
        for j in range(len(test_data_g)):
            if tmp == test_data_g[j]['edit_seq']:
                flag = 0
                tpc_hypo_data.append(hypo_data[j])
                tpc_ref_data.append(test_data_g[j]['test_tgt'])
                if hypo_data[j].strip() == test_data_g[j]['test_tgt'].strip():
                    tpc_cnt += 1
                break
        if flag ==  0:
            cnt += 1
    print("TPC ",tpc_cnt)
    print("TP ",cnt)


if __name__ == '__main__':
    infer_TPC()
    fpc_hypo_data = []
    fpc_ref_data = []
    test_data_c = readjson_all('./class/test.json')
    infer_data = []
    FPC = 0
    for i in fpc_id:
        infer_data.append(test_data_c[i])
    hypo_data = readtxt('fpc.output')
    assert len(hypo_data) == len(infer_data)
    for i in range(len(infer_data)):
        fpc_hypo_data.append(hypo_data[i])
        fpc_ref_data.append(infer_data[i]['test_src'])
        if infer_data[i]['test_src'].strip() == hypo_data[i].strip():
            FPC += 1
    print("FPC: ",FPC)
