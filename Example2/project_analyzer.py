import numpy as np

def classify_project(dataset):
    dic = {}
    for sample in dataset:
        project_name = sample['commit_url'].split('/')[-3]
        if project_name not in dic:
            dic[project_name] = 1
        else:
            dic[project_name] += 1
    ave = np.mean(list(dic.values()))
    return ave