from base64 import encode
import json
import jsonlines
import numpy as np
import torch
from unittest import result
from tqdm import tqdm
from load_train_data import read_examples
from transformers import (RobertaConfig, RobertaModel, RobertaTokenizer)
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
encoder = RobertaModel.from_pretrained("roberta-base").to(device)

def readjson_test_all(filename):
    d = []
    with open(filename, mode='r', encoding='utf-8') as f:
        dicts = json.load(f)
    for i in dicts:
        edit_tmp = []
        for j in i['edit_seq']:
            edit_tmp.append(tuple(j))
        d.append({'edit_seq':edit_tmp, 'test_src':i['test_src'], 'test_tgt':i['test_tgt']})
    return d

def readjs(filename):
    l=[]
    with open(filename, "r+", encoding="utf-8") as f:
        for item in jsonlines.Reader(f):
            l.append(item)
        return l

def read_top(filename):
    file = open(filename, 'r', encoding='utf-8')
    data = []
    for line in file.readlines():
        dic = json.loads(line)
        data.append(dic)
    print(len(data[0]))
    return data[0]

def writejs(data,filenamewrite): 
    print(str(len(data)) + ' pieces of data will be written in ',filenamewrite)
    with jsonlines.open(filenamewrite, mode='w') as writer:
        for i in tqdm(range(len(data))):
            writer.write(data[i])
    writer.close()
    print('Write Jsonl File Done! Path is '+filenamewrite)

def get_embeddings(code, tokenizer, encoder, max_length):
    code_tokens = tokenizer.tokenize(code)[:max_length-2]
    code_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
    code_ids = tokenizer.convert_tokens_to_ids(code_tokens)
    code_ids += [tokenizer.pad_token_id]*(max_length-len(code_ids))
    code_ids = torch.tensor([code_ids]).to(device)
    code_embeddings = encoder(code_ids)['pooler_output'].squeeze(dim=0)
    return code_embeddings

def cosine(code1, code2):
    #  
    # 
    vec1 = get_embeddings(code=repr(code1)[1:-1],tokenizer=tokenizer,encoder=encoder,max_length=150)
    vec2 = get_embeddings(code=repr(code2)[1:-1],tokenizer=tokenizer,encoder=encoder,max_length=150)
    mua = torch.cosine_similarity(vec1, vec2, dim=0)
    return float(mua.data)

def matrix_cos_similar(v1, v2):
    v2 = np.array(v2).T
    dot_matrix = np.dot(v1, v2) 
    v1_row_norm = np.linalg.norm(v1, axis=1).reshape(-1,1)
    v2_col_norm = np.linalg.norm(v2,axis=0).reshape(1,-1)
    norm_matrix = np.dot(v1_row_norm, v2_col_norm)
    res = dot_matrix / norm_matrix
    res[np.isneginf(res)] = 0
    return res    
if __name__ == '__main__':

    top_data = read_top("test_top.jsonl")

    train_data = readjson_test_all('train.json')
    test_data = readjson_test_all('test.json')

    result = []
    print(len(test_data))
    print(len(top_data))
    assert len(test_data) == len(top_data)
    for idx in tqdm(range(len(top_data))):
        candidate_len = len(top_data[str(idx)])
        tmp_res = ''
        best_score = -1
        for i in  range(candidate_len):
            item_idx = top_data[str(idx)][i][1]
            candidate_tmp = train_data[item_idx]
            score = cosine(test_data[idx]['test_src'], candidate_tmp['test_src'])
            if score > best_score:
                best_score = score
                tmp_res = candidate_tmp['test_tgt']
        result.append({'idx':idx,'candidate':repr(tmp_res)[1:-1]})

    writejs(result,'result.jsonl')

