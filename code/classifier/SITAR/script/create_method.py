from utils import readjson_all
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
class SQL():

    def __init__(self, db_name):
        self.conn = sqlite3.connect(db_name)
        self.cursor = self.conn.cursor()
    
    def search_test(self):
        sql_statement = "SELECT * from mod_methods limit 1;"
        res = self.cursor.execute(sql_statement)
        res = res.fetchall()
        print(res)

    def search(self, sql_statement, sample):
        print(sql_statement)
        res = self.cursor.execute(sql_statement, [sample])
        print(res)
        for row in res:
            print(row)

def get_dirs(root_dir):
    for root,dirs,files in os.walk(root_dir):
        return dirs
    

if __name__ == '__main__':
    root_dir = "project"
    dirs = get_dirs(root_dir)
    local_dict = dict()
    for item in dirs:
        tmp = item.split('___')[1]
        local_dict[tmp] = os.path.join(root_dir,item)
    print(local_dict)

    filename = "class/test.json"
    data = readjson_all(filename)
    i = 0
    for item in data:
        if item['label'] == 1:
            continue
            focal_src_filePath = item['focal_db'][5]
            full_name = item['focal_db'][1]
            test_filePath = item['test_db'][5]
            focal_src_commitId = item['focal_db'][3]
            test_commitId = item['test_db'][3]
            local_path = local_dict[full_name.replace('/','_')]
            print(local_path)
            print(full_name)
            print(focal_src_filePath)
            print(test_filePath)
            if os.path.exists(local_path): 
                git_cmd = 'cd {0} && git log -p ./{1}'.format(local_path,focal_src_filePath)
                print(git_cmd)
                res = os.popen(git_cmd).read()
                print(res)
            else:
                print(local_path)
            git_commit = 'cd {0} && git log --pretty=format:"%ct" {1} -1'.format(local_path, focal_src_commitId)
            focal_time = os.popen(git_commit).read()
            time_local = time.localtime(int(focal_time))
            dt = time.strftime("%Y-%m-%d %H:%M:%S",time_local)
            print(dt)
            
        else:

            focal_src_filePath = item['focal_db'][5]
            full_name = item['focal_db'][1]
            test_filePath = item['test_db'][4]
            focal_src_commitId = item['focal_db'][3]
            test_method_name = item['test_db'][5]
            local_path = local_dict[full_name.replace('/','_')]
            print(local_path)
            print(full_name)
            print(focal_src_filePath)
            test_filePath = '/'.join(test_filePath.split('/')[4:])
            print(test_filePath)
            print(test_method_name)
            print(item['test_db'][6])
            
            if os.path.exists(local_path):
                if os.path.exists(os.path.join(local_path,focal_src_filePath)): 
                    git_cmd = 'cd {0} && git log -p ./{1}'.format(local_path,focal_src_filePath)
                    print(git_cmd)
                    res = os.popen(git_cmd).read()
                    print(res)
                if os.path.exists(os.path.join(local_path,test_filePath)): 
                    git_cmd = 'cd {0} && git log -p ./{1}'.format(local_path,test_filePath)
                    print(git_cmd)
                    res = os.popen(git_cmd).read()
                    print(res)
            else:
                print(local_path)
            print("+++++++++++++++"*12)
            i+=1
            break
        

