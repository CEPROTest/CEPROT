import json
from queue import PriorityQueue
from tqdm import tqdm
import jsonlines


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
        examples[idx] = {'edit_seq':edit_tmp, 'test_src':dic['test_src']}
        idx += 1
    return examples

def read_examples_train(filename):
    """Read examples from filename."""
    examples = {}
    idx = 0
    file = open(filename, 'r', encoding='utf-8')
    for line in file.readlines():
        dic = json.loads(line)
        edit_tmp = []
        for i in dic['edit_seq']:
            edit_tmp.append(tuple(i))
        examples[idx] = {'edit_seq':edit_tmp, 'test_src':dic['test_src']}
        idx += 1
    return examples

def read_examples_test(filename):
    """Read examples from filename."""
    examples = {}
    idx = 0
    file = open(filename, 'r', encoding='utf-8')
    for line in file.readlines():
        dic = json.loads(line)
        edit_tmp = []
        for i in dic['edit_seq']:
            edit_tmp.append(tuple(i))
        examples[idx] = {'edit_seq':edit_tmp, 'test_src':dic['test_src_ori']}
        idx += 1
    return examples

def readjson_all(filename):
    d = []
    with open(filename, mode='r', encoding='utf-8') as f:
        dicts = json.load(f)
    for i in dicts:
        edit_tmp = []
        for j in i['edit_seq']:
            edit_tmp.append(tuple(j))
        d.append({'edit_seq':edit_tmp, 'test_tgt':i['test_tgt']})
    return d

def write_top(data, filename):
    with open(filename,'w') as f:
        json.dump(data,f)
    f.close()



train_data = readjson_all("train.json")
test_data = readjson_all("test.json")


if __name__ == '__main__':
    
    top_k = 5
    queue = []
    all_test_top = {}
    for test_idx in tqdm(range(len(test_data))):
        edit_seq = test_data[test_idx]['edit_seq']
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
        all_test_top[test_idx] = tmp_queue

    write_top(all_test_top,'test_top.jsonl')


