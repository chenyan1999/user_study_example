import numpy as np
from BinaryClassClassifier import BinaryClassClassifier

# Example data
X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
y = np.array([0, 1, 0, 1])

# Create an instance of the classifier with scaling enabled
classifier_with_scaling = BinaryClassClassifier(scale=True)
classifier_with_scaling.train(X, y, learning_rate=0.01, epochs=100)

# Make predictions with scaled data
predictions_with_scaling = classifier_with_scaling.predict(X)
assert (predictions_with_scaling == np.array([0, 0, 1, 1])).all()

# Create another instance of the classifier without scaling
classifier_without_scaling = BinaryClassClassifier(scale=False)
classifier_without_scaling.train(X, y, learning_rate=0.01, epochs=100)

# Make predictions without scaling
predictions_without_scaling = classifier_without_scaling.predict(X)
assert (predictions_without_scaling == np.array([1, 1, 1, 1])).all()
print('Congrats, test passed')
