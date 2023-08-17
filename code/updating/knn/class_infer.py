import json
from queue import PriorityQueue
from tqdm import tqdm
import jsonlines
from cosine_sim import cosine

def readjson_all_knn(filename):
    d = []
    with open(filename, mode='r', encoding='utf-8') as f:
        dicts = json.load(f)
    for i in dicts:
        edit_tmp = []
        for j in i['edit_seq']:
            edit_tmp.append(tuple(j))
        d.append({'edit_seq':edit_tmp, 'test_src':i['test_src'], 'test_tgt':i['test_tgt']})
    return d

def readjson_all_knn_FP(filename):
    d = []
    with open(filename, mode='r', encoding='utf-8') as f:
        dicts = json.load(f)
    for i in dicts:
        edit_tmp = []
        for j in i['edit_seq']:
            edit_tmp.append(tuple(j))
        d.append({'edit_seq':edit_tmp, 'test_src':i['test_src']})
    return d


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

def read_examples(filename):
    """Read examples from filename."""
    examples = {}
    idx = 0
    file = open(filename, 'r', encoding='utf-8')
    for line in file.readlines():
        dic = json.loads(line)
        edit_tmp = []
        for i in dic['edit_seq']:
            edit_tmp.append(tuple(i))
        examples[idx] = {'edit_seq':edit_tmp, 'test_src':dic['test_src'], 'test_tgt':dic['test_tgt']}
        idx += 1
    return examples
    
def read_examples_FP(filename):
    """Read examples from filename."""
    examples = []
    idx = 0
    file = open(filename, 'r', encoding='utf-8')
    for line in file.readlines():
        dic = json.loads(line)
        edit_tmp = []
        for i in dic['edit_seq']:
            edit_tmp.append(tuple(i))
        examples.append({'edit_seq':edit_tmp, 'test_src':dic['test_src']})
        idx += 1
    return examples

def readtxt(filename):
    with open(filename) as f:
        d = f.readlines()
    return d


def infer_TP():
    # TP 
    # TPC 
    tpc_data_hypo = []
    tpc_data_ref = []
    test_data = readjson_all('test.json')
    hypo_data = readtxt('hypo.jsonl')
    ref_data = readtxt('gold.jsonl')
    TP_data = readjson_all("TP_data.json")
    TPC = 0
    cnt = 0
    c = 0
    ori_and_token = []
    tp = []
    print("TP:",len(TP_data))
    for i in tqdm(range(len(TP_data))):
        tp_data = TP_data[i]
        flag = 1
        for j in range(len(test_data)):
            if tp_data['edit_seq'] == test_data[j]['edit_seq']:
                tp.append(j)
                tgt_ground_truth = test_data[j]['test_tgt']
                tgt_gen = hypo_data[j].strip().strip('"')
                tpc_data_hypo.append(tgt_gen)
                tpc_data_ref.append({'tgt':tgt_ground_truth,'src':test_data[j]['test_src']})
                t = test_data[j]['test_src']
                flag = 0
                break
        if flag == 1:
            print(i,'-=-=-='*30)
        if flag == 0:
            cnt += 1
      
        if hypo_data[j] == ref_data[j]:
            TPC += 1
            
    print('[][]'*20)
    print("TPC: ",TPC)
    print(cnt)
    print(tp)

def infer_TP_ori():
    tpc_data_hypo = []
    tpc_data_ref = []
    test_data = readjs('test.jsonl')
    hypo_data = readtxt('hypo.jsonl')
    TP_data = readjs("TP_data.jsonl")
    TPC = 0
    cnt = 0
    c = 0
    ori_and_token = []
    for i in tqdm(range(len(TP_data))):
        tp_data = TP_data[i]
        flag = 1
        for j in range(len(test_data)):
            if tp_data['edit_seq'] == test_data[j]['edit_seq']:
                tgt_ground_truth = test_data[j]['test_tgt']
                tgt_gen = hypo_data[j] #.strip()
                tpc_data_hypo.append(tgt_gen)
                tpc_data_ref.append({'tgt':tgt_ground_truth,'src':test_data[j]['test_src']})
                t = test_data[j]
                flag = 0
                break
        if flag == 1:
            print(i,'-=-=-='*30)
        if flag == 0:
            cnt += 1
      
        if tgt_ground_truth.strip() == tgt_gen.strip() and t['test_src'].strip() != t['test_tgt'].strip():
            TPC += 1
            
    print('[][]'*20)
    print(TPC)
    print(cnt) 
    print(ori_and_token)

def infer_FP():
    fpc_data_hypo = []
    fpc_data_ref = []
    FP_data = readjson_all_knn_FP('FP_data.json')
    test_data = readjson_all_knn('test.json')
    train_data = readjson_all_knn('train.json')
    top_k = 10
    queue = []
    FPC = 0
    cnt = 0
    all_FP_top = {}
    for test_idx in tqdm(range(len(FP_data))):
        edit_seq = FP_data[test_idx]['edit_seq']
        pq = PriorityQueue()
        for idx in range(len(train_data)):
            now_cnt = len(set(edit_seq) & set(train_data[idx]['edit_seq']))
            if pq.qsize() < top_k:
                pq.put((now_cnt,idx))
            elif pq.qsize() == top_k:
                min_tmp = pq.get()
                if now_cnt > min_tmp[0]:
                    pq.put((now_cnt,idx))
                else:
                    pq.put(min_tmp)
        tmp_queue = sorted(pq.queue,reverse=True)
        all_FP_top[test_idx] = tmp_queue
    writejs(all_FP_top,'FP_top.jsonl')
    result = []
    for idx in tqdm(range(len(all_FP_top))):
        candidate_len = len(all_FP_top[idx])
        tmp_res = ''
        best_score = -1
        for i in  range(candidate_len):
            item_idx = all_FP_top[idx][i][1]
            candidate_tmp = train_data[item_idx]
            score = cosine(FP_data[idx]['test_src'], candidate_tmp['test_src'])
            if score > best_score:
                best_score = score
                tmp_res = candidate_tmp['test_tgt']
        result.append({'idx':idx,'candidate':tmp_res})
    
    assert len(result) == len(FP_data)
    for i in range(len(FP_data)):
        fpc_data_hypo.append(result[i]['candidate'])
        fpc_data_ref.append({'tgt':result[i]['candidate'],'src':FP_data[i]['test_src']})
        if FP_data[i]['test_src'] == result[i]['candidate']:
            FPC += 1
        else:
            cnt += 1
    print("FP:",len(FP_data))
    print("FPC:",FPC)
    print(cnt)
    writejs(fpc_data_ref,'fpc_ref.jsonl')
    writejs(fpc_data_hypo,'fpc_hypo.jsonl')

if __name__ == '__main__':
    infer_TP()
    infer_FP()
    fpc = 0
    file = open('fpc_ref.jsonl', 'r', encoding='utf-8')
    for line in file.readlines():
        dic = json.loads(line)
        if dic['tgt'] == dic['src']:
            fpc += 1
    print('fpc: ',fpc)
    print('fp:')   