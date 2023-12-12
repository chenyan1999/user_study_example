# README
# Project description
`string.py` provide different chars. In `char.py` defines a function `get_chars()` that return 3 types of chars. In `password_generator.py` defines a function `password_generator`, that receive all chars and randomly sample 1 for several times to generate a password.

# Edit description
A bug has occured in `password_generator.py` when it try to shuffle the chars. The bug is caused since `shuffle()` can not shuffle a string. We should correct it by making varable `letters`, `digits`, `symbols` and `chars` in `char.py` into list. You may run `python password_generator.py` to validate your edit.