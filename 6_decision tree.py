import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import accuracy_score, confusion_matrix
from matplotlib.colors import ListedColormap
import pydotplus
from six import StringIO
from IPython.display import Image

# Load and prepare data
data = pd.read_csv('Social_Network_Ads.csv')
x = data.iloc[:, [2, 3]].values  # Age and EstimatedSalary
y = data.iloc[:, 4].values       # Purchased

# Split the dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

# Feature scaling
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# Train the Decision Tree Classifier
clf = DecisionTreeClassifier(criterion="gini", max_depth=3)
clf.fit(x_train, y_train)

# Predict the test set results
y_pred = clf.predict(x_test)

# Accuracy and confusion matrix
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Visualize decision boundary
x1, x2 = np.meshgrid(
    np.arange(x_test[:, 0].min() - 1, x_test[:, 0].max() + 1, 0.01),
    np.arange(x_test[:, 1].min() - 1, x_test[:, 1].max() + 1, 0.01)
)
plt.contourf(x1, x2, clf.predict(np.c_[x1.ravel(), x2.ravel()]).reshape(x1.shape),
             alpha=0.75, cmap=ListedColormap(["red", "green"]))

for i, j in enumerate(np.unique(y_test)):
    plt.scatter(x_test[y_test == j, 0], x_test[y_test == j, 1],
                c=ListedColormap(["red", "green"])(i), label=j)

plt.title("Decision Tree (Test Set)")
plt.xlabel("Age")
plt.ylabel("Estimated Salary")
plt.legend()
plt.show()

# Export the tree as a .png image
dot = StringIO()
export_graphviz(clf, out_file=dot, filled=True, rounded=True, special_characters=True,
                feature_names=['Age', 'EstimatedSalary'], class_names=['0', '1'])
graph = pydotplus.graph_from_dot_data(dot.getvalue())
graph.write_png('opt_decisiontree_gini.png')
Image(graph.create_png())
