import re
import os
import sys
import copy
import json
import time
import random
import jsonlines
import subprocess
import numpy as np
from tqdm import tqdm
import concurrent.futures
from rank_bm25 import BM25Okapi
# from retriv import SearchEngine

# each proj sample first 20000
# 1. delete edit type "remove", merge it with label "replace"
# 2. edit type "add" now means adding a new line to the corresponding line
# 3. the previous and post context length for replace type are fixed to 5 lines
# 2. for every hunk data, add key: "del_line_idx", to indicate the line number of deleted lines
MAX_THREADS = 1
lang = 'toy'
ROOT_PATH = '.'

APPLY_FILTER = False
REPOS_PATH = os.path.join(ROOT_PATH, 'repos/')
TMP_PATH = os.path.join(ROOT_PATH, 'tmp/')
TMP1_PATH = os.path.join(ROOT_PATH, 'tmp1/')
repo_files = {}
parent_sha_dict = {}

if not os.path.exists(TMP_PATH):
    os.mkdir(TMP_PATH)

if not os.path.exists(TMP1_PATH):
    os.mkdir(TMP1_PATH)

def list_files_in_directory(path):
    file_list = []
    for root, dirs, files in os.walk(path):
        for file in files:
            file_list.append(os.path.relpath(os.path.join(root, file), path))
    return file_list

if not os.path.exists(os.path.join(ROOT_PATH,f'dataset/repo_file_dict.json')): # collect all the file names in each repo
    for repo in tqdm(os.listdir(REPOS_PATH), desc='Collecting file names'):
       repo_files[repo] = list_files_in_directory(os.path.join(REPOS_PATH, repo))
    with open(os.path.join(ROOT_PATH,f'dataset/repo_file_dict.json'), 'w', encoding='utf-8') as file:
        json.dump(repo_files, file, indent=4, ensure_ascii=False)
else:
    with open(os.path.join(ROOT_PATH,f'dataset/repo_file_dict.json'), 'r', encoding='utf-8') as file:
        repo_files = json.load(file)
    if len(repo_files) != len(os.listdir(REPOS_PATH)):
        for repo in tqdm(os.listdir(REPOS_PATH), desc='Collecting file names'):
            if repo not in repo_files:
                repo_files[repo] = list_files_in_directory(os.path.join(REPOS_PATH, repo))
        with open(os.path.join(ROOT_PATH,f'dataset/repo_file_dict.json'), 'w', encoding='utf-8') as file:
            json.dump(repo_files, file, indent=4, ensure_ascii=False)

if not os.path.exists(os.path.join(ROOT_PATH,f'dataset/{lang}_parent_sha_dict.json')):
    with open(os.path.join(ROOT_PATH,f'dataset/{lang}_dataset.jsonl'), 'r', encoding='utf-8') as file:
        for i in file.readlines():
            data = json.loads(i)
            parent_sha_dict[data['new_sha']] = data['old_sha']
    with open(os.path.join(ROOT_PATH,f'dataset/{lang}_parent_sha_dict.json'), 'w', encoding='utf-8') as file:
        json.dump(parent_sha_dict, file, indent=4, ensure_ascii=False)
else:
    with open(os.path.join(ROOT_PATH,f'dataset/{lang}_parent_sha_dict.json'), 'r', encoding='utf-8') as file:
        parent_sha_dict = json.load(file)

def retrieve_file(repo_path: str, file_path: str, commit_sha: str, splitlines: bool = True):
    '''
    Func: Retrieve file of a given version
    Args:
        repo_path: str, the repository directory
        file_path: str, the relative path for the file in 
                        terms of the repository
        commit_sha: the commit version
    '''
    # Check out the specific commit
    try:
        if ' ' in repo_path:
            git_repo_path = repo_path.replace(' ', r'\ ')
        else:
            git_repo_path = repo_path
        checkout_command = f'git -C {git_repo_path} checkout -q --force {commit_sha} -- {file_path}'
        result = subprocess.run(checkout_command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except subprocess.CalledProcessError as e:
        # 如果命令执行失败，可以处理 stderr 中的错误消息
        error_message = e.stderr
        raise KeyError(f"At repo: {repo_path}\n Git checkout error:", error_message)
    except Exception as ex:
        # 处理其他可能发生的异常
        raise KeyError(f"At repo: {repo_path}\n An error occurred:", ex)
    # Copy the file to the desired location
    source_file = os.path.normpath(os.path.join(repo_path, file_path))
    with open(source_file, 'r', encoding='utf-8') as f:
        if splitlines:
            content = f.readlines()
        else:
            content = f.read()

    # Always return to the original working directory
    return content
    
def extract_info(old_file_path, new_file_path):
    file_path = '/'.join(old_file_path.split('/')[3:])
    elements = old_file_path.split('/')[2].split('_')
    user_name = elements[0]
    old_sha = elements[-1]
    proj_name = '_'.join(elements[1:-1])
    new_sha = new_file_path.split('/')[2].split('_')[-1]

    return user_name, proj_name, old_sha, new_sha, file_path

def contains_non_english_content(content):
    if type(content) == list:
        content = ''.join(content)
    try:
        content.encode('ascii')
    except UnicodeEncodeError:
        # 字符串中包含非ASCII字符
        return True
    # 字符串只包含ASCII字符
    return False
    
def apply_filter(data_sample):
    def extract_extension(file_path):
        # 使用os.path.basename获取文件名
        filename = os.path.basename(file_path)
        # 使用splitext分割文件名和后缀
        file_name_elements = filename.split('.')
        if len(file_name_elements) == 2:
            return '.'+file_name_elements[-1]
        else:
            return '.'+'.'.join(file_name_elements[-2:])
    
    # Hunk level filter
    # 1. if code window is empty, return False
    if len(data_sample['code_window']) <= 3:
        return False
    # 2. if code_window, add_line and commit msg contains more than english words, return False
    if contains_non_english_content(data_sample['add_line']) or \
        contains_non_english_content(data_sample['code_window']) or \
        contains_non_english_content(data_sample['commit_msg']):
        return False
    # 3. if hunk type is 'add', but add_line is empty, return False
    if data_sample['hunk_type'] == 'add':
        # 3.1 if there is no add line, return False
        if len(data_sample['add_line'].splitlines()) == 0 or data_sample['add_line'] == '':
            return False
        # 3.2 if the whole window is add:
        if 'keep' not in data_sample['label_window']:
            return False
        # 3.3 if add more than 15 lines of code:
        if len(data_sample['add_line'].splitlines()) > 15:
            return False
    # 4. if commit msg contains less than 5 words, return False
    if len(data_sample['commit_msg'].split()) < 5:
        return False
        
    # 4. if hunk type is 'replace', but the label window do not contain 'remove'
    if data_sample['hunk_type'] == 'remove':
        # 4.1 if there is no replace line, return False
        if 'replace' not in data_sample['label_window']:
            return False
        # 4.2 if delete more than 15 lines of code:
        if len(data_sample['del_line_idx']) > 15:
            return False
        # 4.3 if add more than 15 lines of code:
        if len(data_sample['add_line'].splitlines()) > 15:
            return False

    # File level filter
    # 1. if the file is automatically generated, return False
    # for each line in auto_gen_file_type, represent auto gen file type in Javascript, Java, Typescript, Python, Go project respectively
    # 1. if the file is auto-generated, remove it
    auto_gen_file_type = ['.map', '.lock', '.build', '.sample', '.min.js', '.bundle.js', '.less'  # Javascript
     '.class', '.jar', '.war', '.ear', '.bak', '.log', '.tmp',  # Java
     '.tsbuildinfo',  # Typescript
     '.pyc', '.pyo', '.pyd', '.so', '.dll',  # Python
     '.a', '.o', '.test', '.cover', '.prof', '.exe', '.pb.go', '.gen.go', '.swagger.go', '.generated.go'# Go
     '.css', '.html', '.md', '.sh', '.pem']
    extension = extract_extension(data_sample['file_path'])
    if extension in auto_gen_file_type:
        return False
    return True

def filter(dataset):
    # apply hunk & file level_filter
    print('apply hunk & file level_filter')
    new_dataset = []
    for datasample in tqdm(dataset):
        if apply_filter(datasample):
            new_dataset.append(datasample)
    
    # apply commit level filter
    # Category each hunk datasample by commit_id's html_url
    print('apply commit level filter')
    commit_id_dict = {}
    for idx, datasample in enumerate(tqdm(new_dataset)):
        if datasample['html_url'] in commit_id_dict:
            commit_id_dict[datasample['html_url']].append(idx)
        else:
            commit_id_dict[datasample['html_url']] = [idx]
    
    failed_datasample_idx = []
    for commit_url in tqdm(commit_id_dict):
        # if the number of changed hunks is less than 3, remove the commit
        if len(commit_id_dict[commit_url]) < 3:
            failed_datasample_idx.extend(commit_id_dict[commit_url])
    # pop out hunk datasamples whose index is in failed_datasample_idx
    failed_datasample_idx.sort(reverse=True)
    for idx in failed_datasample_idx:
        new_dataset.pop(idx)
    return new_dataset

def convert2hunks(dataset, sha):
    # dataset is on project level, the next level is file level(i), then hunk level(j)
    # The dataset is on commit level, the next level is file level(i), then hunk level(j)
    global REPOS_PATH, TMP_PATH, lang
    hunks = []
    prev_line = [3,4,5]
    
    if os.path.exists(os.path.join(TMP_PATH, f'{sha}_hunks.json')):
        return sha, -1

    total = 0
    for i in dataset:
        for j in i['changes']:
            total += 1
            try:
                # a small filter
                if j['del_line'].rstrip() == j['add_line'].rstrip():
                    continue
                # add
                if j['del_line_idx'] == [] and j['add_line_idx']:
                    # with open('../'+i['new_file_path'], encoding='utf-8') as file:
                    proj_name = i['proj_name']
                    commit_sha = i['new_sha']
                    file_path = i['file_path']
                    repo_path = os.path.join(REPOS_PATH, proj_name)
                    # Get the content of this file of the given commit
                    try:
                        lines = retrieve_file(repo_path, file_path, commit_sha)
                    except KeyError:
                        continue
                    except:
                        raise KeyError('Unexpected error:', sys.exc_info()[0])

                    label = ['keep']*len(lines)
                    
                    add_line = j['add_line_idx'][0]-1
                    start_line = j['add_line_idx'][0]-random.choice(prev_line)
                    mid_line = j['add_line_idx'][-1]
                    end_line = mid_line+random.choice(prev_line)+1
                    
                    if start_line < 0:
                        start_line = 0
                    if end_line > len(lines):
                        end_line = None
                        
                    code_window = lines[start_line:add_line] + lines[mid_line:end_line]
                    label_window = label[start_line:add_line-1] + ['add'] + label[mid_line:end_line]
                    assert len(code_window) == len(label_window)
                    assert len(code_window) != 0
                    assert 'add' in label_window
                    data_sample = {
                                'code_window': code_window, 
                                'label_window': label_window, 
                                'commit_msg': i['commit_msg'], 
                                'html_url': i['html_url'], 
                                "del_line_idx": j['del_line_idx'],
                                'add_line': j['add_line'], 
                                'method_name': j['func_name'],
                                'file_path': file_path, 
                                'idx': idx,
                                'hunk_type': 'add'
                            }
                    hunks.append(data_sample)
                        
                # replace
                else:
                    # with open('../'+i['old_file_path'], encoding='utf-8') as file:
                    proj_name = i['proj_name']
                    commit_sha = i['old_sha']
                    file_path = i['file_path']
                    repo_path = os.path.join(REPOS_PATH, proj_name)
                    try:
                        lines = retrieve_file(repo_path, file_path, commit_sha)
                    except KeyError:
                        continue
                    except:
                        raise KeyError('Unexpected error:', sys.exc_info()[0])

                    label = ['keep']*len(lines)
                    for del_line in j['del_line_idx']:
                        label[del_line-1] = 'replace'
                    
                    start_line = j['del_line_idx'][0]-5
                    end_line = j['del_line_idx'][-1]+5
                    if start_line < 0:
                        start_line = 0
                    if end_line > len(lines):
                        end_line = None
                    code_window = lines[start_line:end_line]
                    label_window = label[start_line:end_line]
                    assert len(code_window) == len(label_window)
                    assert len(code_window) != 0
                    assert 'replace' in label_window
                    data_sample = {
                                'code_window': code_window, 
                                'label_window': label_window, 
                                'commit_msg': i['commit_msg'], 
                                'html_url': i['html_url'], 
                                'del_line_idx': j['del_line_idx'],
                                'add_line': j['add_line'], 
                                'method_name': j['func_name'],
                                'file_path': file_path, 
                                'idx': idx,
                                'hunk_type': 'replace'
                            }
                    hunks.append(data_sample)
                
            except:
                continue

    # save to json file
    converted_no = len(hunks)
    if total != 0:
        conversion_rate = converted_no / total
    else:
        conversion_rate = -1
    with open(os.path.join(TMP_PATH, f'{sha}_hunks.json'), 'w', encoding='utf-8') as file:
        json.dump({'hunks': hunks}, file)
    return  sha, conversion_rate

def find_indices_and_retrieve(source_list, element, target_list):
    indices = [i for i, x in enumerate(source_list) if x == element]
    retrieved_content = [target_list[i] for i in indices]
    return retrieved_content

def convert2generator_dataset(commit, sha):
    global TMP1_PATH
    if os.path.exists(os.path.join(TMP1_PATH, f'{sha}_generator_dataset.jsonl')):
        return sha

    output = []
    tokenized_corpus = [''.join(i['code_window']+[i['add_line']]).split() for i in commit]
    bm25 = BM25Okapi(tokenized_corpus) # build a BM25 object with other hunks
    for co_change_idx, co_change in enumerate(commit):
        code_window = ''.join(co_change['code_window'])
        label_window = ' '.join(co_change['label_window'])
        commit_message = co_change['commit_msg']
        context = []
        if len(code_window) == 0:
            continue
        
        # BM25 search for related context
        try:
            tokenized_query = code_window.split()
            retrieval_code = bm25.get_top_n(tokenized_query, tokenized_corpus, n=6)
            context_index = [tokenized_corpus.index(i) for i in retrieval_code] # get the index of the top 5 similar hunks

            # form context, which are the deleted and added lines in the top 5 similar hunkss
            context_index.remove(co_change_idx)
            for idx in context_index:             
                if commit[idx]['hunk_type'] == 'replace': 
                    remove = ''.join(find_indices_and_retrieve(commit[idx]['label_window'], 'replace', commit[idx]['code_window']))
                    context.append('remove '+ remove)
                    context.append('add '+ commit[idx]['add_line'])
                elif commit[idx]['label_window'] == 'add':
                    context.append('add '+ commit[idx]['add_line'])
        except:
            pass
        
        code_window = ''.join([f' <mask> {i}' for i in co_change['code_window']]) # each line of code will have a label mask in the beginning
        input_ = ' </s> '.join([code_window, commit_message] + context)
        output_ =   co_change['add_line']
        html_url =  co_change['html_url']
        file_name = co_change['file_path']
        label_window = co_change['label_window']
        output.append({"docstring_tokens":output_, "code_tokens":input_, "label_window": label_window, "html_url":html_url, "file_name":file_name})
    with open(os.path.join(TMP1_PATH, f'{sha}_generator_dataset.jsonl'), 'w', encoding='utf-8') as file:
        for i in output:
            file.write(json.dumps(i, ensure_ascii=False)+'\n')
    return sha

def convert2locator_dataset(commit, sha):
    global TMP1_PATH
    if os.path.exists(os.path.join(TMP1_PATH, f'{sha}_locator_dataset.jsonl')):
        return sha
    output = []
    tokenized_corpus = [''.join(i['code_window']+[i['add_line']]).split() for i in commit]
    bm25 = BM25Okapi(tokenized_corpus)
    skip_idx_lst = [] # stores index of hunks that have been merged to other hunks
    for co_change_idx, co_change in enumerate(commit):
        if co_change_idx in skip_idx_lst:
            continue
        code_window = co_change['code_window']
        label_window = co_change['label_window']
        commit_message = co_change['commit_msg']
        merged = False
        # try:
        if co_change_idx + 1 != len(commit) and \
        commit[co_change_idx]['file_path'] == commit[co_change_idx+1]['file_path'] and \
        commit[co_change_idx]['hunk_type'] != 'add' and \
        commit[co_change_idx+1]['hunk_type'] != 'add' and \
        len(commit[co_change_idx]['del_line_idx']) != 0 and \
        len(commit[co_change_idx+1]['del_line_idx']) != 0 and \
        commit[co_change_idx+1]['del_line_idx'][0] - co_change['del_line_idx'][-1] <= 5:
            next_code_window = commit[co_change_idx+1]['code_window']
            next_label_window = commit[co_change_idx+1]['label_window']
            # merge code_window
            for i in range(len(code_window)): # find overlap
                if next_code_window[0] == code_window[len(code_window)-1-i]:
                    break
            try:
                assert code_window[len(code_window)-1-i:] == next_code_window[0:i+1]
                curr_window_len = len(code_window)
                code_window = code_window[:curr_window_len-1-i] + next_code_window
                merge_label_window = []
                for idx in range(len(code_window)):
                    if idx < curr_window_len-1-i:
                        merge_label_window.append(label_window[idx])
                    elif idx >= curr_window_len-1-i and idx < curr_window_len:
                        if label_window[idx] == 'replace' or next_label_window[idx-curr_window_len+i+1] == 'replace':
                            merge_label_window.append('replace')
                        else:
                            merge_label_window.append('keep')
                    else:
                        merge_label_window.append(next_label_window[idx-curr_window_len+i+1])
                label_window = merge_label_window
                assert len(code_window) == len(label_window)
                skip_idx_lst.append(co_change_idx+1)
                merged = True
                # trim the first keeps and last keeps in the label window and code window
                trim_start = random.choice([i for i in range(0, label_window.index('replace'))])
                last_replace_idx = len(label_window) - 1 - label_window[::-1].index('replace') + 1
                trim_end = random.choice([i for i in range(last_replace_idx, len(label_window))])
                
                code_window = code_window[trim_start:trim_end]
                label_window = label_window[trim_start:trim_end]
            except:
                pass
        
        masked_code_window = ''.join([f' <mask> {code_window[i]}' for i in range(len(code_window))])
        code_window = ''.join(code_window)
        label_window = ' '.join(label_window)
        context = []
        if len(code_window) == 0:
            continue
        
        # BM25 search for related context
        try:
            tokenized_query = code_window.split()
            retrieval_code = bm25.get_top_n(tokenized_query, tokenized_corpus, n=7)
            context_index = [tokenized_corpus.index(i) for i in retrieval_code]

            context_index.remove(co_change_idx)
            if merged:
                context_index.remove(co_change_idx+1)
            for idx in context_index:
                if commit[idx]['hunk_type'] == 'replace':
                    remove = ''.join(find_indices_and_retrieve(commit[idx]['label_window'], 'replace', commit[idx]['code_window']))
                    context.append('remove '+ remove)
                    context.append('add '+ commit[idx]['add_line'])
                elif commit[idx]['hunk_type'] == 'add':
                    context.append('add '+ commit[idx]['add_line'])
        except:
            pass
        
        input_ = ' </s> '.join([masked_code_window, commit_message] + context)
        html_url =  co_change['html_url']
        file_name = co_change['file_path']
        output.append({"docstring_tokens":label_window, "code_tokens":input_, "html_url":html_url, "file_name":file_name})

    with open(os.path.join(TMP1_PATH, f'{sha}_locator_dataset.jsonl'), 'w', encoding='utf-8') as file:
        for i in output:
            file.write(json.dumps(i, ensure_ascii=False)+'\n')
    return sha

def form_database(files, repo_path, old_sha):
    collection = []
    for file in files:
        try:
            content = retrieve_file(repo_path, file, old_sha, splitlines=False)
        except:
            continue
        d = {"id": file, "text": content}
        collection.append(d)
    
    if collection == []:
        return None
    try:
        se = SearchEngine("new_index").index(collection)
    except:
        return None
    return se

def detect_clone(edits, file_path, se):
    queries = []
    for edit in edits:
        queries.append({"id": len(queries), "text": edit})
    try:
        output = se.msearch(queries=queries, cutoff=100)
        similar_files = []
        for q, r in output.items():
            similar_files = list(set(similar_files).union(set(r.keys())))
        if file_path in similar_files:
            return 1
        else:
            return 0
    except:
        return 0
    
def convert2discriminator_dataset(commit, url, sha):
    global TMP1_PATH, repo_files, REPOS_PATH, parent_sha_dict
    if os.path.exists(os.path.join(TMP1_PATH, f'{sha}_discriminator_dataset.jsonl')):
        return sha

    # prepare common variables
    output = []
    proj_name = commit[0]['html_url'].split('/')[-3]
    repo_path = os.path.join(REPOS_PATH, proj_name)
    new_sha = commit[0]['html_url'].split('/')[-1]
    old_sha = parent_sha_dict[new_sha]
    msg = commit[0]['commit_msg']
    
    # prepare a database
    co_change_file_dict = {} # key is file path, values are edits in this file
    for hunk in commit:
        if hunk['file_path'] in co_change_file_dict:
            co_change_file_dict[hunk['file_path']].append('\n'.join(hunk['code_window']))
            co_change_file_dict[hunk['file_path']].append(hunk['add_line'])
        else:
            co_change_file_dict[hunk['file_path']] = ['\n'.join(hunk['code_window']), hunk['add_line']]

    unchanged_files = list(set(repo_files[proj_name]) - set(co_change_file_dict.keys()))
    picked_unchanged_files = random.sample(unchanged_files, min(len(unchanged_files),len(commit)))
    
    search_engine = form_database(list(co_change_file_dict.keys())+picked_unchanged_files, repo_path, old_sha)
    
    if search_engine == None:
        with open(os.path.join(TMP1_PATH, f'{sha}_discriminator_dataset.jsonl'), 'w', encoding='utf-8') as file:
            for i in output:
                file.write(json.dumps(i, ensure_ascii=False)+'\n')
        return sha

    pair_dict = {} # used to avoid duplication
    # prepare positive samples
    co_change_file = list(co_change_file_dict.keys())
    for i in range(len(co_change_file)-1):
        for j in range(i+1, len(co_change_file)):
            file_pathA = co_change_file[i]
            file_pathB = co_change_file[j]
            input = ' </s> '.join([file_pathA, file_pathB, msg])
    
            if input not in pair_dict:
                pair_dict[input] = 1
                exist_clone = detect_clone(co_change_file_dict[file_pathA], file_pathB, search_engine)
                output.append({"code_tokens":input, "html_url": url, "exist_clone":exist_clone, 'docstring_tokens':1})

    # prepare negative samples
    neg_sample_num = 2 * len(output)
    for i in range(neg_sample_num):
        file_pathA = random.choice(co_change_file)
        file_pathB = random.choice(picked_unchanged_files)
        input = ' </s> '.join([file_pathA, file_pathB, msg])
        if input not in pair_dict:
            pair_dict[input] = 1
            exist_clone = detect_clone(co_change_file_dict[file_pathA], file_pathB, search_engine)
            output.append({"code_tokens":input, "html_url": url, "exist_clone":exist_clone, 'docstring_tokens':0})

    with open(os.path.join(TMP1_PATH, f'{sha}_discriminator_dataset.jsonl'), 'w', encoding='utf-8') as file:
        for i in output:
            file.write(json.dumps(i, ensure_ascii=False)+'\n')
    return sha
        
if __name__ == '__main__':
    time_start = time.time()

    if not os.path.exists(os.path.join(ROOT_PATH,f'dataset/{lang}_hunk_dataset.json')):
        # load dataset
        dataset = []
        with open(os.path.join(ROOT_PATH,f'dataset/{lang}_dataset.jsonl'), 'r', encoding='utf-8') as file:
            for i in file.readlines():
                data = json.loads(i)
                dataset.append(data)
        print(f'{lang} dataset No. of changed files: {len(dataset)}')

        # category data samples by project
        commit_dict = {}
        proj_dict = {}
        for idx, datasample in enumerate(dataset):
            # if datasample['proj_name'] not in ['kratos', 'micro', 'faas', 'gorm', 'hub', 'nsq', 'frp', 'lazydocker', 'caddy', 'fzf', 'echo']:
            #     continue
            if datasample['proj_name'] not in proj_dict:
                proj_dict[datasample['proj_name']] = len(datasample['changes'])
            elif proj_dict[datasample['proj_name']] > 30000:
                continue
            else:
                proj_dict[datasample['proj_name']] += len(datasample['changes'])
            if datasample['new_sha'] in commit_dict:
                commit_dict[datasample['new_sha']].append(datasample)
            else:
                commit_dict[datasample['new_sha']] = [datasample]
        print(f'{lang} dataset No. of commits: {len(commit_dict)}')

        # convert to hunks
        print('Convert datasample to hunks')
        hunks = []
        # executor = concurrent.futures.ThreadPoolExecutor(MAX_THREADS)
        # with tqdm(total=len(commit_dict)) as pbar:
        #     features = [executor.submit(convert2hunks, commit_dict[sha], sha) for sha in sorted(commit_dict.keys())]

        #     for feature in concurrent.futures.as_completed(features):
        #         sha, conversion_rate = feature.result()
        #         pbar.set_description(f"Conversion rate: {conversion_rate:.4f}")
        #         pbar.update(1)
                
        for sha in sorted(commit_dict.keys()):
            convert2hunks(commit_dict[sha], sha)

        # after all datasamples have been converted to hunks, combine them
        for sha in tqdm(commit_dict.keys()):
            with open(os.path.join(TMP_PATH, f'{sha}_hunks.json'), 'r', encoding='utf-8') as file:
                try:
                    data = json.load(file)
                except:
                    continue
                hunks.extend(data['hunks'])

        print('#hunks:', len(hunks))
        
        # filter hunks and save them
        result_dict = {} 
        if APPLY_FILTER:
            all_hunks = filter(hunks)
        else:
            all_hunks = hunks
        for i in all_hunks:
            html_url = i['html_url']
            if html_url in result_dict:
                result_dict[html_url].append(i)
            else:
                result_dict[html_url] = [i]

        # save the dict into json file
        with open(os.path.join(ROOT_PATH,f'dataset/{lang}_hunk_dataset.json'), 'w', encoding='utf-8') as file:
            json.dump(result_dict, file, indent=4, ensure_ascii=False)
        # delete the tmp files
        # for file in os.listdir(TMP_PATH):
        #     os.remove(os.path.join(TMP_PATH, file))
    else:
        with open(os.path.join(ROOT_PATH,f'dataset/{lang}_hunk_dataset.json'), 'r', encoding='utf-8') as file:
            result_dict = json.load(file)

    # convert to edit generation dataset
    print('Make generator dataset')
    gen_data_path = os.path.join(ROOT_PATH, f'generator_data/{lang}')
    if not os.path.exists(os.path.join(gen_data_path, f'test.jsonl')):
        # executor = concurrent.futures.ThreadPoolExecutor(MAX_THREADS)
        # with tqdm(total=len(result_dict)) as pbar:
        #     features = [executor.submit(convert2generator_dataset, result_dict[sha], sha.split('/')[-1]) for sha in sorted(result_dict.keys())]

        #     for feature in concurrent.futures.as_completed(features):
        #         pbar.update(1)
        #         sha = feature.result()
        
        for sha in sorted(result_dict.keys()):
            convert2generator_dataset(result_dict[sha], sha.split('/')[-1])
                
        outputs = []
        for sha in sorted(result_dict.keys()):
            sha = sha.split('/')[-1]
            with open(os.path.join(TMP1_PATH, f'{sha}_generator_dataset.jsonl'), 'r', encoding='utf-8') as file:
                for i in file.readlines():
                    outputs.append(json.loads(i))

        print('The size of generator dataset is:', len(outputs))
        
        if not os.path.exists(gen_data_path):
            os.makedirs(gen_data_path)
        with jsonlines.open(os.path.join(gen_data_path, f'train.jsonl'), 'w') as f:
            for item in outputs:
                f.write(item)

        for file in os.listdir(TMP1_PATH):
            os.remove(os.path.join(TMP1_PATH, file))
    else:
        print('Generator dataset already exists!')

    # convert to edit locator dataset
    print('Make locator dataset')
    loc_data_path = os.path.join(ROOT_PATH, f'locator_data/{lang}')
    if not os.path.exists(os.path.join(loc_data_path, 'test.jsonl')):
        # executor = concurrent.futures.ThreadPoolExecutor(MAX_THREADS)

        # with tqdm(total=len(result_dict)) as pbar:
        #     features = [executor.submit(convert2locator_dataset, result_dict[sha], sha.split('/')[-1]) for sha in sorted(result_dict.keys())]

        #     for feature in concurrent.futures.as_completed(features):
        #         pbar.update(1)
        #         sha = feature.result()

        for sha in tqdm(sorted(result_dict.keys())):
            convert2locator_dataset(result_dict[sha], sha.split('/')[-1])

        outputs = []
        for sha in sorted(result_dict.keys()):
            sha = sha.split('/')[-1]
            with open(os.path.join(TMP1_PATH, f'{sha}_locator_dataset.jsonl'), 'r', encoding='utf-8') as file:
                for i in file.readlines():
                    outputs.append(json.loads(i))

        print('The size of locator dataset is:', len(outputs))
        if not os.path.exists(loc_data_path):
            os.makedirs(loc_data_path)
        with jsonlines.open(os.path.join(loc_data_path, 'train.jsonl'), 'w') as f:
            for item in outputs:
                f.write(item)

        for file in os.listdir(TMP1_PATH):
            os.remove(os.path.join(TMP1_PATH, file))
   
    print('Make discriminator dataset')
    random.seed(42)
    dis_data_path = os.path.join(ROOT_PATH, f'discriminator_data/{lang}')
    if not os.path.exists(os.path.join(dis_data_path, 'test.jsonl')):
        # sample commit as discriminator do not need this big dataset
        sampled_result_list = {}
        for commit_url in result_dict.keys():
            sampled_result_list[commit_url] = result_dict[commit_url]

        # executor = concurrent.futures.ThreadPoolExecutor(MAX_THREADS)

        # with tqdm(total=len(sampled_result_list)) as pbar:
        #     features = [executor.submit(convert2discriminator_dataset, sampled_result_list[sha], sha.split('/')[-1]) for sha in sorted(sampled_result_list.keys())]

        #     for feature in concurrent.futures.as_completed(features):
        #         pbar.update(1)
        #         sha = feature.result()

        for sha in tqdm(sorted(sampled_result_list.keys())):
            convert2discriminator_dataset(sampled_result_list[sha], sha, sha.split('/')[-1])

        outputs = []
        for sha in sorted(sampled_result_list.keys()):
            sha = sha.split('/')[-1]
            with open(os.path.join(TMP1_PATH, f'{sha}_discriminator_dataset.jsonl'), 'r', encoding='utf-8') as file:
                for i in file.readlines():
                    outputs.append(json.loads(i))

        print('The size of discriminator dataset is:', len(outputs))
        if not os.path.exists(dis_data_path):
            os.makedirs(dis_data_path)
        with jsonlines.open(os.path.join(dis_data_path, 'train.jsonl'), 'w') as f:
            for item in outputs[:int(0.7*len(outputs))]:
                f.write(item)

        for file in os.listdir(TMP1_PATH):
            os.remove(os.path.join(TMP1_PATH, file))

    
    print('Done!')
    time_end = time.time()
    print('Time cost:', time_end-time_start)

