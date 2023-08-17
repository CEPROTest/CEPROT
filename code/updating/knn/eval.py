from unittest import result
from bleu import _bleu, compute_sentence_level_blue
import jsonlines
from tqdm import tqdm
from typing import List
from javalang.tokenizer import tokenize
import json


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
            writer.write(repr(data[i])[1:-1])
    writer.close()
    print('Write Jsonl File Done! Path is '+filenamewrite)

def write_pair():
    results = readjs('result.jsonl')
    test_data = readjson_all('test.json')
    ref = readjson_all('test.json')
    hypo = []
    gold = []
    assert len(results) == len(test_data)
    for i in results:
        hypo.append(i['candidate'])

    for i in test_data:
        gold.append(i['test_tgt'])

    writejs(hypo,'hypo.jsonl')
    writejs(gold,'gold.jsonl')

def eval():
    acc = 0
    hypo = readjs('hypo.jsonl')
    gold = readjs('gold.jsonl')
    assert len(hypo) == len(gold)
    for i in range(len(hypo)):
        if hypo[i] == gold[i]:
            acc += 1
    ACC = round(acc/len(gold),4)
    print('Xmatch:',ACC)
    bleu_4 = _bleu('gold.jsonl', 'hypo.jsonl')
    print('bleu_4:',bleu_4)
    print('ACC+bleu_4:',ACC+bleu_4)

def word_level_edit_distance(a: List[str], b: List[str]) -> int:
    max_dis = max(len(a), len(b))
    distances = [[max_dis for j in range(len(b)+1)] for i in range(len(a)+1)]
    for i in range(len(a)+1):
        distances[i][0] = i
    for j in range(len(b)+1):
        distances[0][j] = j

    for i in range(1, len(a)+1):
        for j in range(1, len(b)+1):
            cost = 0 if a[i-1] == b[j-1] else 1
            distances[i][j] = min(distances[i-1][j] + 1,
                                  distances[i][j-1] + 1,
                                  distances[i-1][j-1] + cost)
    return distances[-1][-1]

if __name__ == '__main__':
    write_pair()

