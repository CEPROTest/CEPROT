import json
from tqdm import tqdm
import numpy as np

def readjson_single(filename):
    file = open(filename, 'r', encoding='utf-8')
    for line in file.readlines():
        dic = json.loads(line)
    return dic

def readjson(filename):
    file = open(filename, 'r', encoding='utf-8')
    data = []
    for line in file.readlines():
        dic = json.loads(line)
        data.append(dic)
    return data

def readjson_all(filename):
    d = []
    with open(filename, mode='r', encoding='utf-8') as f:
        dicts = json.load(f)
    for i in dicts:
        d.append(i)
    return d

def writejson(data,filename):
    d = []
    with open(filename,'w') as f:
        for item in tqdm(data):
            d.append(item)
        json.dump(data,f)
    f.close()

def get_data(filename):
    Xtrain = []
    Ytrain = []
    data = readjson_all(filename)
    for i in range(len(data)):
        if data[i]['label'] == 'NEGATIVE':
            Ytrain.append(0)
        else:
            Ytrain.append(1)
        tmp = [data[i]['add_annotation_line'], data[i]['add_call_line'], data[i]['add_classname_line'], data[i]['add_condition_line'], data[i]['add_field_line'],\
                data[i]['add_import_line'], data[i]['add_packageid_line'], data[i]['add_parameter_line'], data[i]['add_return_line'],\
                data[i]['del_annotation_line'], data[i]['del_call_line'], data[i]['del_classname_line'], data[i]['del_condition_line'], data[i]['del_field_line'],\
                data[i]['del_import_line'], data[i]['del_packageid_line'], data[i]['del_parameter_line'], data[i]['del_return_line']
                    ]
        Xtrain.append(tmp)
    x = np.array(Xtrain)
    y = np.array(Ytrain)
    return x, y