# README

# Project description
In this project, we analyze the `toy_dataset.json`. In this json file, we have 4 samples, each have a key: `commit_url`, it value have specific format: https://github.com/{user_name}/{project_name}/commit/{commit_id}. In `commit_analyzer.py`, we defined function `classify_commit`, where calculates the average number of samples per commit. In `project_analyzer.py`, we defined function `classify_project` and in `user_analyzer.py`, we defined function `classify_user`. In `main.py`, we invocate this 3 function to analyze the `toy_dataset.json`.

# Edit description
We found that function `classify_commit`, `classify_project` and `classify_user` share great similarity, hence we want to extract them into a new function `classify` into file `commit_analyzer.py`, with arguments `dataset` and `idx`. After this extraction, we can invocate this function in function `classify_commit`, `classify_project` and `classify_user`.

After edit, you may run `python main.py` to validate your edit.

# Edit prompt
The edit description you may feed into the extension is:

extract similar code into new function: def classify(dataset, idx)