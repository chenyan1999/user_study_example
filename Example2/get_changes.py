import os
import re
import json
import time
import random
import requests
import jsonlines

from proxies_pool import proxy_list
from user_agent_pool import user_agents    

GITHUB_TOKENS = ['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']
CURR_TOKEN_IDX = 0
GITHUB_TOKENS_RST_TIME = [time.time()-3600 for _ in range(len(GITHUB_TOKENS))]
ROOT_PATH = './'

def get_response(request_url, params=None):
    global CURR_TOKEN_IDX
    MAX_RETRIES = 10
    headers = {
        'User-Agent': random.choice(user_agents),
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Authorization':'token '+ GITHUB_TOKENS[CURR_TOKEN_IDX],
        'Accept-Encoding': 'gzip,deflate,sdch',
        'Accept-Language': 'zh-CN,zh;q=0.8'
    }
    for i in range(MAX_RETRIES):
        proxy = random.choice(proxy_list)
        try:
            r = requests.get(request_url, params, headers=headers,
                            proxies={"http": proxy}, timeout=40)
        except requests.exceptions.RequestException as e:
            if i < MAX_RETRIES - 1:
                continue
            raise Exception(e)

        if r.status_code == 200:
            break # if successfully get response, break the loop and return content
        else:
            if r.status_code == 403: # if the request budget has been used up, sleep for 1 hour
                print("==> 403 Forbidden, the request budget has been used up")
                print("==> Switch to another token")
                GITHUB_TOKENS_RST_TIME[CURR_TOKEN_IDX] = time.time()
                CURR_TOKEN_IDX = (CURR_TOKEN_IDX + 1) % len(GITHUB_TOKENS)
                headers = { # update the headers
                    'User-Agent': random.choice(user_agents),
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                    'Authorization':'token '+ GITHUB_TOKENS[CURR_TOKEN_IDX],
                    'Accept-Encoding': 'gzip,deflate,sdch',
                    'Accept-Language': 'zh-CN,zh;q=0.8'
                }
                if GITHUB_TOKENS_RST_TIME[CURR_TOKEN_IDX] + 3600 > time.time():
                    print("==> All tokens have been used up, sleep until next token is available")
                    time.sleep(3600-(time.time()-GITHUB_TOKENS_RST_TIME[CURR_TOKEN_IDX])+10)
            else: # other errors, sleep for 1 second
                time.sleep(1)
            if i == MAX_RETRIES - 1: # if all retries failed, raise error
                raise ConnectionError(f"Cannot connect to website: {request_url}, status code: {r.status_code}")
    return r.content

def get_all_response(request_url, params=None):
    '''
    Regardless of the number of pages and the request page index, get all the response
    '''
    if params == None:
        params = {
            "per_page": '100', 
            "page": "1"
        }
    all_d = []
    per_page = int(params['per_page'])
    params['page'] = 1
    while True:
        content = get_response(request_url, params)
        d = json.loads(content)
        all_d.extend(d)
        if len(d) < per_page:
            break
        else:
            params['page'] += 1
            time.sleep(1)
    return all_d

def get_small_response(request_url, params=None):
    '''
    only get 1 page with 5 items, for test, remove it later
    '''
    if params == None:
        params = {
            "per_page": '5', 
            "page": "1"
        }
    content = get_response(request_url, params)
    d = json.loads(content)
    return d

def get_repos(lang, repo_num):
    # get the top star repos' information of this language
    repos = []
    for page_idx in range(1, repo_num // 100 + 2):
        request_url = "https://api.github.com/search/repositories"
        params = {
            "q": "language:{} stars:>500".format(lang),
            "page": "{}".format(str(page_idx)),
            "per_page": "100",
            "sort": "stars",
            "order": "desc"
        }
        content = get_response(request_url, params)
        d = json.loads(content)
        items = d["items"]

        for item in items:
            title = item["full_name"]
            url = item["html_url"]
            date_time = item["updated_at"]
            description = item["description"]
            stars = item["stargazers_count"]
            line = u"* [{title}]({url})|{stars}|{date_time}|:\n {description}\n". \
                format(title=title, date_time=date_time, url=url, description=description, stars=stars)
            print(line)
            repos.append(item)
            if len(repos) >= repo_num:
                break
    if len(repos) == 0:
        raise Exception("No repos found")
    return repos

def extract_patch(patch):
    change_rec = []
    '''
    first split pharagraph that start with 
    "@@ -start_line_idx,line_num +start_line_idx,line_num @@"
    '''
    for window in patch.split("@@ -"):
        if len(window) == 0:
            continue
        window = "@@ -" + window
        try:
            old_window_start_line = int(re.findall(r"@@ \-(.+?)[,|\s]", window)[0]) # get the start line index of this window's old version
            new_window_start_line = int(re.findall(r"\+(.+?)[,|\s]", window)[0]) # get the start line index of this window's new version
        except:
            raise Exception("Cannot find start line index")
        lines = window.split("\n")
        func_name = lines[0].split("@@")[-1].strip(' ')
        change_item = {
            'func_name':func_name,
            'del_line_idx': [],
            'add_line_idx': [],
            'del_line': '',
            'add_line': ''}
        for i in range(1, len(lines)):
            if not lines[i].startswith("-") and not lines[i].startswith("+"):
                old_window_start_line += 1
                new_window_start_line += 1
                if not (len(change_item['del_line_idx']) == 0 and len(change_item['add_line_idx']) == 0): # if it is a non-empty change item
                    # print(change_item)
                    change_rec.append(change_item) # record this change
                    change_item = { # reset change item
                        'func_name':func_name,
                        'del_line_idx': [],
                        'add_line_idx': [],
                        'del_line': '',
                        'add_line': ''
                    } 
            else:
                if lines[i].startswith("-"):
                    change_item['del_line_idx'].append(old_window_start_line)
                    change_item['del_line'] += ('' if change_item['del_line'] == '' else "\n") + lines[i][1:]
                    old_window_start_line += 1
                if lines[i].startswith("+"):
                    change_item['add_line_idx'].append(new_window_start_line)
                    change_item['add_line'] += ('' if change_item['add_line'] == '' else "\n") + lines[i][1:]
                    new_window_start_line += 1
    if len(change_item['del_line_idx']) == len(lines)-1 or len(change_item['add_line_idx']) == len(lines)-1:
        return []  # when this patch deleted or added the entire file, we exclude this change
    # if the change include the last line of the file
    if not (len(change_item['del_line_idx']) == 0 and len(change_item['add_line_idx']) == 0): 
        change_rec.append(change_item) # record this change
    return change_rec

def get_changes(user_name, proj_name, sha, old_sha):
    global ROOT_PATH

    print(f'==> Record {proj_name} changes in {sha}')
    request_url = "https://api.github.com/repos/{}/{}/commits/{}".format(user_name, proj_name,sha)
    params = {
        "per_page": '100',
        "page": 1
    }
    files = []
    while True: # loop to retrieve every file changed under this commit version 
        content = get_response(request_url, params)
        changes_d = json.loads(content)
        files.extend(changes_d["files"])
        if len(changes_d["files"]) < 100:
            break
        else:
            time.sleep(1)
            params['page'] += 1
            
    change_records = []
    for file in files: # for every file changed in this commit version
        if file['status'] not in ['modified', 'added', 'removed']: # if the file is not modified, added or removed, we ignore it
            continue
        try:
            patch = file["patch"]  # exist situations where the commit contains no patch (e.g. rename a file) 
        except:
            continue
        file_name_w_path = file["filename"]
        # identify delted and added lines
        try:   # if the patch is not valid, we ignore it
            for item in extract_patch(patch):
                item['file_path'] = file_name_w_path # the file name with path
                change_records.append(item)
        except:
            print(f'==> Patch extraction failed in {sha}, {file_name_w_path}, ignored')
            continue
    with jsonlines.open(ROOT_PATH+f"/changes/{user_name}_{proj_name}_{sha}_{old_sha}.jsonl", 'w') as writer:
        writer.write_all(change_records)
 


            
