import os
import sys
import json
import zipfile
import jsonlines
import subprocess
import urllib.request
from tqdm import tqdm
from proxies_pool import proxy_list
from user_agent_pool import user_agents

from get_changes import get_response, ROOT_PATH, GITHUB_TOKENS, CURR_TOKEN_IDX
def git_clone(user_name, proj_name):
    # Check if this repo has been downloaded
    global ROOT_PATH
    global GITHUB_TOKENS, CURR_TOKEN_IDX
    if not os.path.exists(ROOT_PATH+'/repos'):
        os.mkdir(ROOT_PATH+'/repos')
    if os.path.exists(ROOT_PATH+f'/repos/{proj_name}/'):
        return
    # if not, download the whole repo of the latest version
    curr_dir = os.getcwd()
    clone_url = f"https://{GITHUB_TOKENS[CURR_TOKEN_IDX]}@github.com/{user_name}/{proj_name}.git"
    try:
        os.chdir(os.path.normpath(ROOT_PATH+'/repos'))
        git_clone_command = ["git", "clone", clone_url]
        # Run the Git clone command
        subprocess.run(git_clone_command, check=True)
    except:
        os.chdir(curr_dir)
        raise Exception(f"==> Downloading {user_name}/{proj_name} from {clone_url} failed")
    
    os.chdir(curr_dir)
    
def get_datasample():
    global ROOT_PATH
    if not os.path.exists(ROOT_PATH+'/dataset'):
        os.mkdir(ROOT_PATH+'/dataset')
    pbar = tqdm(sorted(os.listdir(ROOT_PATH+f"/changes/")))
    for file_name in pbar:   # for every change recorded in jsonl
        samples = []
        if file_name.startswith('.') or file_name[-6:] != '.jsonl':  # ignore any hidden file or files not jsonl
            continue
        l = file_name[:-6].split('_')
        if len(l) < 4: # if this file don't have the correct format, ignore it
            continue
        else:
            user_name = l[0]
            old_sha = l[-1]
            sha = l[-2]
            proj_name = '_'.join(l[1:-2]) # in case that the project name contain _
        
        pbar.set_description(proj_name)
        # 从已经下载的 commit history jsonl 中找到对应的 commit message 和 html_url
        commit_msg = ""
        html_url = f'https://github.com/{user_name}/{proj_name}/commit/{sha}'

        # print(f'==> Converting {user_name}/{proj_name}\'s commit {sha} into data samples')     
        # open jsonl file and convert to list
        with jsonlines.open(ROOT_PATH+f'/changes/{file_name}') as reader:
            changes = list(reader)

        # aggregate changes that happens on the same file
        file_changes = {}
        for change in changes:
            file_path = change['file_path']
            if file_path not in file_changes:
                file_changes[file_path] = []
            change.pop('file_path')
            file_changes[file_path].append(change)

        # write a data sample for each file
        for file in file_changes:
            dic = {
                'user_name': user_name,
                'proj_name': proj_name,
                'old_sha': old_sha,
                'new_sha': sha,
                'file_path': file,
                'changes': file_changes[file],
                'commit_msg': commit_msg,
                'html_url': html_url
            }
            samples.append(dic)      

        with jsonlines.open(ROOT_PATH+f"/dataset/toy_dataset.jsonl", 'a') as writer:
            writer.write_all(samples)
        # delete the changes file when done
        os.remove(ROOT_PATH+f'/changes/{file_name}')


        git_clone(user_name, proj_name)
