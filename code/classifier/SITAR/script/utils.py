import json
from tqdm import tqdm

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