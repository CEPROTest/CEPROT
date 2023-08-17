import torch
import os
import numpy as np
import pandas as pd
import pickle
import time
import datetime 
import random
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import torch
import transformers as ppb
import warnings
from transformers import BertTokenizer
from transformers import BertModel
from transformers import RobertaTokenizer, RobertaConfig, RobertaModel
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import get_linear_schedule_with_warmup
from transformers import  AdamW, BertConfig
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import matplotlib

def readjson(filename):
    import json
    file = open(filename, 'r', encoding='utf-8')
    data = []
    for line in file.readlines():
        dic = json.loads(line)
        data.append(dic)
    file.close()
    return data

def loadpkl(filepath):
    return pickle.load(open(filepath, 'rb'))