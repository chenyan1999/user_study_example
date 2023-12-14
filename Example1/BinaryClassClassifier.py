import numpy as np
from MultiClassClassifier import MultiClassClassifier

class BinaryClassClassifier(MultiClassClassifier):
    def __init__(self):
        MultiClassClassifier.__init__(self, 2)