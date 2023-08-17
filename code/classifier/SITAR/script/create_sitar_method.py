from utils import readjson_all, writejson
from tqdm import tqdm
import sqlite3
import time
import os

"""
    reposity: the project's name
    Path: The test file path
    Pro_Path: production file path
    File_Date: test code commit date
    Pro_Date: production code commit date
    Pro_Commit: production code commit ID
    Commit: Test code commit ID
"""
def get_dirs(root_dir):
    for root,dirs,files in os.walk(root_dir):
        return dirs

def write_json(filename, data):
    json_str = json.dumps(data)
    with open(filenames, 'w') as json_file:
        json_file.write(json_str)
    json_file.close()

class Example():
    def __init__(self,reposity,test_path,pro_path,pro_date,pro_commit, label):
        self.reposity = reposity
        self.testPath = test_path
        self.proPath = pro_path
        self.proDate = pro_date
        self.proCommit = pro_commit
        self.label = label

class SITAR():
    def __init__(self, data_path, project_root):
        self.data_path = data_path
        self.project_dir = project_root
        self.repo_dict = dict()
        self.samples = []

    def merge_class_data(self):
        data1 = readjson_all('class/test.json')
        print(len(data1))
        data2 = readjson_all('class/train.json')
        print(len(data2))
        data2.extend(data1)
        print(len(data2))
        writejson(data2, "class/train&test.json")
    
    def get_a_ori_sample(self):
        data = readjson_all(self.data_path)
        return data[0]

    def get_repo_dict(self):
        dirs = get_dirs(self.project_dir)
        for item in dirs:
            tmp = item.split('___')[1]
            repo_dict[tmp] = item
        
    def get_samples(self):
        data = readjson_all(self.data_path)
        for i in tqdm(range(len(data))):
            item = data[i]
            if item['label'] == 1:
                self.samples.append({'reposity':item['focal_db'][1], 'testPath':item['test_db'][5], 'proPath':item['focal_db'][5], 'proCommit':item['focal_db'][3], 'label':1})
            else:
                self.samples.append({'reposity':item['focal_db'][1], 'testPath':item['test_db'][4], 'proPath':item['focal_db'][5], 'proCommit':item['focal_db'][3], 'label':0})

    def write_samples(self):
        tmp_dict = dict()
        for i in tqdm(range(len(self.samples))):
            item = self.samples[i]
            if item['reposity'] not in tmp_dict:
                tmp_dict[item['reposity']] = 1
            else:
                tmp_dict[item['reposity']] += 1
            repo_tmp = item['reposity'].replace('/', '_')
            filename = './Method/' + self.repo_dict[repo_tmp] + '/' + str(tmp_dict[item['reposity']]) + '.json'
            write_json(filename, item)

if __name__ == '__main__':
    
    sitar = SITAR('class/train&test.json')
    

