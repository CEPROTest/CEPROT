from base64 import encode
import json
import jsonlines
import numpy as np
import torch
from unittest import result
from tqdm import tqdm
from multiprocessing.pool import Pool
import vthread
import os
import json

from gen_candidate import read_examples
from transformers import (RobertaConfig, RobertaModel, RobertaTokenizer)

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
encoder = RobertaModel.from_pretrained('roberta-base').to(device)

def readjson_all(filename):
    d = []
    with open(filename, mode='r', encoding='utf-8') as f:
        dicts = json.load(f)
    for i in dicts:
        d.append(i)
    return d

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
    code_tokens = tokenizer.tokenize(repr(code)[1:-1])[:max_length-2]
    code_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
    code_ids = tokenizer.convert_tokens_to_ids(code_tokens)
    code_ids += [tokenizer.pad_token_id]*(max_length-len(code_ids))
    code_ids = torch.tensor([code_ids]).to(device)
    code_embeddings = encoder(code_ids)['pooler_output'].squeeze(dim=0)
    return code_embeddings

def cosine(code1, code2):
    vec1 = get_embeddings(code=code1,tokenizer=tokenizer,encoder=encoder,max_length=150)
    vec2 = get_embeddings(code=code2,tokenizer=tokenizer,encoder=encoder,max_length=150)
    mua = torch.cosine_similarity(vec1, vec2, dim=0)
    return float(mua.data)

results = []
def classifiar(item):
    idx = item
    top_k = top_data[idx]
    tmp_label = ''
    best_score = -1
    for i in  top_k:
        candidate_tmp = train_data[i[1]]
        score = cosine(test_data[int(idx)]['test_src'], candidate_tmp['test_src'])
        if score > best_score:
            best_score = score
            tmp_label = candidate_tmp['label']
    results.append({'idx':idx,'candidate_label':tmp_label})

if __name__ == '__main__':
    top_data = read_top("test_top.jsonl")
    train_data = readjson_all('train.json')
    test_data = readjson_all('test.json')
    results = []
    for i in tqdm(top_data):
        classifiar(i)
    writejs(results,'result.jsonl')

