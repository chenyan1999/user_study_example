import numpy as np

def classify_user(dataset):
    dic = {}
    for sample in dataset:
        user_name = sample['commit_url'].split('/')[-4]
        if user_name not in dic:
            dic[user_name] = 1
        else:
            dic[user_name] += 1
    ave = np.mean(list(dic.values()))
    return ave