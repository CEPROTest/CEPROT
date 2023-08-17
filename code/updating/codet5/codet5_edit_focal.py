import os

import sys

import argparse
import json
import logging
import os
import pickle
import random
import re
from turtle import clear

import jsonlines
import numpy as np
import torch
from torch.utils.data import (DataLoader, Dataset, RandomSampler,
                              SequentialSampler, TensorDataset)
from tqdm import tqdm, trange
from transformers import T5Model, T5Tokenizer
from tree_sitter import Language, Parser

from model.utils import *


class Config(object):

    def __init__(self): 
        self.model_name = 'codet5_edit_focal'
        self.device = torch.device('cuda',0)
        self.n_gpu = 1
        self.decoder_path = 'codet5-base'
        self.hidden_size = 768
        self.tokenizer = RobertaTokenizer.from_pretrained(self.decoder_path)
        self.batch_size = 2
        self.eval_batch_size = 2
        self.num_epochs = 15 
        self.learning_rate = 1e-5
        self.gradient_accumulation_steps = 1
        self.adam_epsilon = 1e-4
        self.weight_decay = 0.001

        self.do_train = False
        self.do_eval = False
        self.do_test = True
        #test_src
        self.max_edit_seq_length = 150
        self.max_test_src_length = 150
        self.max_focal_src_length = 150
        self.max_focal_tgt_length = 150
        self.max_target_length = 150
        
        self.beam_size=10
        self.max_length=150 #generation sentence length


        self.dev_filename = 'test.json'
        self.train_filename = 'train.json'
        self.test_filename = 'fpc_infer_data.json'
        self.output_dir = 'codet5_edit_focal/'

class Code2Code(nn.Module):
    def __init__(self, decoder_model,t5_tokenizer,config):
        super(Code2Code, self).__init__()
        self.t5_model = decoder_model
        self.config=config
        self.t5_tokenizer = t5_tokenizer

    def forward(self, source_ids=None,source_mask=None,target_ids=None,target_mask=None):
        if target_ids is not None:
            outputs = self.t5_model(input_ids=source_ids, attention_mask=source_mask, labels=target_ids)
            loss, logits = outputs.loss, outputs.logits
            return loss, logits
        else:
            generate_ids = self.t5_model.generate(input_ids=source_ids,attention_mask=source_mask, max_length=self.config.max_target_length)
            pres = self.t5_tokenizer.batch_decode(generate_ids, skip_special_tokens=True)
            return pres

def readjs(filename):
    l=[]
    with open(filename, "r+", encoding="utf-8") as f:
        for item in jsonlines.Reader(f):
            l.append(item)
        return l

def readjson_all(filename):
    d = []
    with open(filename, mode='r', encoding='utf-8') as f:
        dicts = json.load(f)
    for i in dicts:
        d.append(i)
    return d


def readjson(filename):
    d = []
    with open(filename, mode='r', encoding='utf-8') as f:
        dicts = json.load(f)
    for i in dicts:
        d.append(i)
    return d

def filter_code(codes):
  codes = codes.replace('\r',' ').replace('\n',' ').replace('\t',' ')
  codes = re.sub(' +', ' ', codes)
  return codes

class Example(object):
    """A single training/test example."""
    def __init__ (self,
                 idx,
                 edit_seq,
                 focal_src,
                 focal_tgt,
                 test_src,
                 test_tgt
                 ):
        self.idx = idx
        self.edit_seq = edit_seq
        self.focal_src = focal_src
        self.focal_tgt = focal_tgt
        self.test_src = test_src
        # self.test_tgt = test_tgt
        self.test_tgt = 'None'

def read_examples(filename):
    """Read examples from filename."""
    examples=[]
    idx = 0
    data = readjson_all(filename)
    for line in data:
        examples.append(
            Example(
                idx=idx,
                edit_seq = line['edit_seq'],
                focal_src = repr(line['focal_src'])[1:-1],
                focal_tgt = repr(line['focal_tgt'])[1:-1],
                test_src = repr(line['test_src'])[1:-1],
                test_tgt = 'None'
            )
        )
        idx += 1
    return examples

class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 example_id,
                 source_ids,                 
                 target_ids,
                 source_mask,
                 target_mask, 

    ):
        self.example_id = example_id
        self.source_ids = source_ids
        self.target_ids = target_ids
        self.source_mask = source_mask
        self.target_mask = target_mask 


def convert_examples_to_features(examples, args):
    features = []
    for example_index, example in enumerate(tqdm(examples,total=len(examples))):
        edit_seq= example.edit_seq
        focal_src = example.focal_src
        focal_tgt = example.focal_tgt
        test_src = example.test_src
        test_tgt = example.test_tgt

        edit_input_ids = []
        edit_attention_mask = []

        for item in edit_seq:
            encoded_tmp = args.tokenizer(item, padding=True, truncation=True,max_length=args.max_edit_seq_length, return_tensors='pt')
            edit_input_ids.extend(np.array(encoded_tmp['input_ids']).flatten())
            edit_attention_mask.extend(np.array(encoded_tmp['attention_mask']).flatten())

        edit_input_ids = edit_input_ids[:args.max_edit_seq_length]
        edit_attention_mask = edit_attention_mask[:args.max_edit_seq_length]
        padding_length1 = args.max_edit_seq_length - len(edit_input_ids)
        padding_length2 = args.max_edit_seq_length - len(edit_attention_mask)
        edit_input_ids += [args.tokenizer.pad_token_id]*padding_length1
        edit_attention_mask += [0]*padding_length2

        encoding_focal_src = args.tokenizer(focal_src, max_length=args.max_focal_src_length,truncation=True,return_tensors="pt")
        focal_src_input_ids= encoding_focal_src['input_ids'][0].tolist()
        focal_src_attention_mask = encoding_focal_src['attention_mask'][0].tolist()
        padding_length = args.max_focal_src_length - len(focal_src_input_ids)
        focal_src_input_ids += [args.tokenizer.pad_token_id]*padding_length
        focal_src_attention_mask += [0]*padding_length

        encoding_focal_tgt = args.tokenizer(focal_tgt, max_length=args.max_focal_tgt_length,truncation=True,return_tensors="pt")
        focal_tgt_input_ids= encoding_focal_tgt['input_ids'][0].tolist()
        focal_tgt_attention_mask = encoding_focal_tgt['attention_mask'][0].tolist()
        padding_length = args.max_focal_tgt_length - len(focal_tgt_input_ids)
        focal_tgt_input_ids += [args.tokenizer.pad_token_id]*padding_length
        focal_tgt_attention_mask += [0]*padding_length


        encoding_test_src = args.tokenizer(test_src, max_length=args.max_test_src_length,truncation=True,return_tensors="pt")
        test_src_input_ids= encoding_test_src['input_ids'][0].tolist()
        test_src_attention_mask = encoding_test_src['attention_mask'][0].tolist()
        padding_length = args.max_test_src_length - len(test_src_input_ids)
        test_src_input_ids += [args.tokenizer.pad_token_id]*padding_length
        test_src_attention_mask += [0]*padding_length

        source_ids = edit_input_ids + focal_src_input_ids + focal_tgt_input_ids + test_src_input_ids
        source_mask = edit_attention_mask + focal_src_attention_mask + focal_tgt_attention_mask + test_src_attention_mask


        tgt_seq = filter_code(test_tgt)
        encoding_tgt = args.tokenizer(tgt_seq, max_length=args.max_target_length,truncation=True,return_tensors="pt")
        target_ids = encoding_tgt.input_ids[0].tolist()
        target_mask = encoding_tgt.attention_mask[0].tolist()
        padding_length = args.max_target_length - len(target_ids)
        target_ids += [args.tokenizer.pad_token_id]*padding_length
        target_mask += [0]*padding_length
        target_ids = torch.tensor(target_ids)
        target_ids[target_ids == args.tokenizer.pad_token_id] = -100

        features.append(
            InputFeatures(
                 example_index,
                 source_ids,
                 target_ids,
                 source_mask,
                 target_mask
            )
        )
    return features


class GenDataset(Dataset):
    def __init__(self, cofig, pos_data): 
        self.args=cofig
        self.pos_data = pos_data
        self.src_examples = convert_examples_to_features(self.pos_data, self.args)
        
    def __len__(self):
        return len(self.pos_data)

    def __getitem__(self, item):
        return (
                torch.tensor(self.src_examples[item].source_ids),
                torch.tensor(self.src_examples[item].source_mask),
                torch.tensor(self.src_examples[item].target_ids),
                torch.tensor(self.src_examples[item].target_mask))
