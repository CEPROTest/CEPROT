from utils.edit import DiffTokenizer, empty_token_filter, construct_diff_sequence_with_con
import json
from tqdm import tqdm, trange
import re
import jsonlines


def test_construct_diff_sequence_with_con(code1, code2):
    diff_tokenizer = DiffTokenizer(token_filter=empty_token_filter)
    a_tokens, b_tokens = diff_tokenizer(code1, code2)
    change_seqs = construct_diff_sequence_with_con(a_tokens,b_tokens)
    dist = 0
    for item in change_seqs:
        if item[2] != 'equal':
            dist += 1
    return change_seqs,dist


def test_tokens(code1, code2):
    diff_tokenizer = DiffTokenizer(token_filter=empty_token_filter)
    a_tokens, b_tokens = diff_tokenizer(code1, code2)
    change_seqs = construct_diff_sequence_with_con(a_tokens,b_tokens)
    a = []
    b = []
    for item in change_seqs:
        if item[0] != '':
            a.append(item[0])
        if item[1] != '':
            b.append(item[1])
    return a, b


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

def readjson(filename):
    import json
    file = open(filename, 'r', encoding='utf-8')
    data = []
    for line in file.readlines():
        dic = json.loads(line)
        data.append(dic)
    print(len(data))
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
data = []
"""
Care
Your
File
Path
!
!
!
"""
raw_data = readjs('./class/train.json')
for idx, item in tqdm(enumerate(raw_data)):
    t = {}
    t['sample_id'] = idx
    t['src_method'] = item['focal_src'] 
    t['dst_method'] = item['focal_tgt']
    t['code_change_seq'], t['dist'] = test_construct_diff_sequence_with_con(t['src_method'], t['dst_method'])
    t['src_desc'] = item['test_src']
    t['label'] = item['label']
    # t['dst_desc'] = item['test_tgt']
    t['dst_desc'] = ''
    t['src_desc_tokens'], t['dst_desc_tokens'] = test_tokens(t['src_desc'], t['dst_desc'])
    t['focal_src_tokens'], t['focal_tgt_tokens'] = test_tokens(t['src_method'], t['dst_method'])
    data.append(t)

writejs(data,"train.jsonl")

