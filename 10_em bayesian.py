from sklearn.mixture import GaussianMixture
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load and prepare the Iris dataset
data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Standardize the data (important for EM)
X_scaled = StandardScaler().fit_transform(X)

# Apply EM algorithm using Gaussian Mixture Model
gmm = GaussianMixture(n_components=3)
gmm.fit(X_scaled)
clusters = gmm.predict(X_scaled)

# Visualize clustering results
colors = np.array(['red', 'green', 'blue'])
plt.scatter(X.iloc[:, 2], X.iloc[:, 3], c=colors[clusters], s=40)
plt.title("EM Clustering (GMM)")
plt.xlabel("Petal length")
plt.ylabel("Petal width")
plt.show()
