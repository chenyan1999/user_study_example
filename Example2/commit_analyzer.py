import numpy as np

def classify(dataset, idx):
    dic = {}
    for sample in dataset:
        commit_id = sample['commit_url'].split('/')[idx]
        if commit_id not in dic:
            dic[commit_id] = 1
        else:
            dic[commit_id] += 1
    ave = np.mean(list(dic.values()))
    return ave

def classify_commit(dataset):
    return classify(dataset, -1)