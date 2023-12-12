import numpy as np
from BinaryClassClassifier import BinaryClassClassifier

# Generate a simple binary classification dataset with varying scales
X_binary_unscaled = np.array([[1000, 1], [800, 1.2], [1200, 1.5], [1500, 1.8], [2000, 2]])
y_binary_unscaled = np.array([0, 1, 0, 1, 0])
# Create an instance of MultiClassClassifier without scaling for binary classification
binary_classifier_no_scale = BinaryClassClassifier(scale=False)
# Train the model without scaling
binary_classifier_no_scale.train(X_binary_unscaled, y_binary_unscaled)
# Make predictions on a new set of binary data without scaling
X_test_binary_unscaled = np.array([[1100, 1.3], [1800, 1.6]])
predictions_binary_no_scale = binary_classifier_no_scale.predict(X_test_binary_unscaled)
assert (predictions_binary_no_scale == np.array([0, 0])).all()

binary_classifier_scale = BinaryClassClassifier(scale=True)
# Train the model without scaling
binary_classifier_scale.train(X_binary_unscaled, y_binary_unscaled)
# Make predictions on a new set of binary data without scaling
X_test_binary_unscaled = np.array([[1100, 1.3], [1800, 1.6]])
predictions_binary_scale = binary_classifier_scale.predict(X_test_binary_unscaled)
assert (predictions_binary_scale == np.array([1, 0])).all()
print('Congrats, test passed')