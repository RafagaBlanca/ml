from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


iris = load_iris()
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names) # type: ignore
iris_df['target'] = iris.target #type: ignore
iris_df.head()

kmeans = KMeans(n_clusters=3, init = 'k-means++', max_iter=300, random_state=0)
kmeans.fit(iris.data)
print(kmeans.labels_)

iris_df['cluster'] = kmeans.labels_
iris_df.groupby(['target', 'cluster']).agg({'sepal length (cm)':'count'})





