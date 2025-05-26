from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import numpy as np
from sklearn import datasets

# Load the Iris dataset
iris = datasets.load_iris()
iris_data = iris.data
iris_labels = iris.target

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(iris_data, iris_labels, test_size=0.20, random_state=50)

# Initialize the KNN classifier with n_neighbors=10
classifier = KNeighborsClassifier(n_neighbors=10)

# Train the classifier
classifier.fit(x_train, y_train)

# Make predictions
y_pred = classifier.predict(x_test)

# Print the classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Print the confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
