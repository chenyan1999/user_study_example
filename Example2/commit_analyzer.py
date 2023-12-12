import numpy as np

def classify_commit(dataset):
    dic = {}
    for sample in dataset:
        commit_id = sample['commit_url'].split('/')[-1]
        if commit_id not in dic:
            dic[commit_id] = 1
        else:
            dic[commit_id] += 1
    ave = np.mean(list(dic.values()))
    return ave