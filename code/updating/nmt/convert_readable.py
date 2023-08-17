import javalang
import json
from tqdm import tqdm
import collections
import sys
import re

def readtxt(filename):
    t = []
    with open(filename) as f:
        data = f.readlines()
    for i in data:
        t.append(i.strip())
    return t

def write_txt(data, filename):
    with open(filename,'w') as f:
        for i in data:
            f.write(repr(i)[1:-1])
            f.write('\n')
    f.close()

def convert_data(ori_data):
    converted_data = []
    for i in ori_data:
        try:
            after = javalang.tokenizer.reformat_tokens(list(javalang.tokenizer.tokenize(i))).replace('\n',' ').replace('\t',' ').replace('\r',' ')
            after = re.sub(' +', ' ',after)
            converted_data.append(after)
        except:
            print(i)
            converted_data.append(i)
    return converted_data

        

if __name__ == '__main__':
    hypo = readtxt('./re_result_hypo.txt')
    gold = readtxt('./re_result_ref.txt')
    hypo_con = convert_data(hypo)
    gold_con = convert_data(gold)
    write_txt(hypo_con, './re_con_result_hypo.txt')
    write_txt(gold_con, './re_con_result_ref.txt')
