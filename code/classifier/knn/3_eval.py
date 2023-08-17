import jsonlines
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, auc
import logging
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
            writer.write(data[i])
    writer.close()
    print('Write Jsonl File Done! Path is '+filenamewrite)
    
def writejson(data,filename):
    d = []
    with open(filename,'w') as f:
        for item in tqdm(data):
            d.append(item)
        json.dump(data,f)
    f.close()

def get_pair():
    results = readjs('result.jsonl')
    ref = readjson_all('test.json')
    predication = []
    trues = []
    for i in results:
        predication.append(i['candidate_label'])

    for i in ref:
        trues.append(i['label'])
    return trues, predication

def eval():
    test_actual, test_pred = get_pair()
    ref = readjson_all('test.json')
    TP_data = []
    TP_data_ori = []
    FP_data = []
    assert len(test_actual) == len(test_pred)
    for i in range(len(test_actual)):
        if test_actual[i] == 1 and test_pred[i] == 1:
            TP_data.append(ref[i])
        elif test_actual[i] == 0 and test_pred[i] == 1:
            FP_data.append(ref[i])
    writejson(TP_data,'TP_data.json')
    writejson(FP_data,'FP_data.json')
    print(len(TP_data))
    print(len(FP_data))
    cm = confusion_matrix(test_actual, test_pred)
    tn,fp,fn,tp = cm[0][0],cm[0][1],cm[1][0],cm[1][1]
    print('TP:',tp)
    print('FP:',fp)
    print('TN:',tn)
    print('FN:',fn)
    acc = round((tp+tn)/(tp+tn+fp+fn),4)
    precision = round(tp/(tp+fp),4)
    recall = round(tp/(tp+fn),4)
    f1_score = round(2*precision*recall/(precision+recall),4)
    fpr = round(fp/(fp+tn),4)
    tpr = round(tp/(tp+fn),4)
    x = [0,fpr,1]
    y = [0,tpr,1]
    auc_score = round(auc(x,y),4)
    return acc,precision,recall,f1_score,fpr,tpr,auc_score
if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
    logger = logging.getLogger(__name__)
    acc,precision,recall,f1,fpr,tpr,auc_score = eval()
    result = {
        'acc':acc,
        "eval_recall": recall,
        "eval_precision": precision,
        "eval_f1": f1,
        "fpr":fpr,
        'tpr':tpr,
        "auc":auc_score,
    }

    logger.info("***** Eval results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(result[key]))
    logger.info("  " + "*" * 20)
