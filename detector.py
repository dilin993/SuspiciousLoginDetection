import pca
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.cluster import KMeans
import seaborn as sns

df = pd.read_csv('features.csv')

x = df[df.columns[1:len(df.columns)]].values
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df = pd.DataFrame(x_scaled)
pca = pca.PCA(x)

pca.fit()

y = pca.reduce(keep_info=0.95)

kmeans = KMeans(n_clusters=2, random_state=0).fit(x)

print float(sum(kmeans.labels_==0)) / float(len(kmeans.labels_))

print df[kmeans.labels_==1]

color_mapping = {0: sns.xkcd_rgb['bright purple'], 1: sns.xkcd_rgb['lime']}
colors = list(map(lambda x: color_mapping[x], kmeans.labels_))

plt.scatter(y[:, 0], y[:, 1], c=colors)
plt.show()
