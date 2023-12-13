# README

# Project description
This project contains 2 files: `MultiClassClassifier.py` and `BinaryClassClassifier.py`, where `MultiClassClassifier.py` defined a class named `MultiClassClassifier`, and `BinaryClassClassifier.py` defined a subclass named `BinaryClassClassifier`

# Edit description
We wish to add an boolean attribute named `scale` to both `MultiClassClassifier` and `BinaryClassClassifier`. If set to True, a normalization should be applied to $X$ when method `train` and `predict` is called. The normalization function is:

$X = (X-\bar{X})/(std(X)+1e^{-8})$, 

where $\bar{X}$ is the mean of $X$, and $std(X)$ is the standard deviation of $X$. You may add new instance variable like `self.mean` and `self.std`.

You may run `python validation.py` to validate your edit.

# Edit prompt
The edit description you may feed into the extension is:

add boolean attribute scale, X = (X - X.mean) / (X.std + 1e-8)