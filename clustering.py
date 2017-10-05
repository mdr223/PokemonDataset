from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import fcluster
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# goal: run PCA and then KMeans

def importTypeData(filename):
	df = pd.read_csv(filename)
	#k = len(df["Type 1"])
	k = df["Type 1"].nunique()
	types = df["Type 1"].unique()
	df_types = df[["Type 1", "Type 2"]]
	df_types = pd.get_dummies(df_types, columns=["Type 1", "Type 2"], prefix=["type_1", "type_2"])
	#df_kmeans = df.drop(["#", "Name", "Generation"], axis=1)
	return (k, types, df, df_types)


# script starts here
(k, types, df, df_types) = importTypeData('pokemon.csv')
arr_original = df.values
arr_kmeans = df_types.values

kmeans = KMeans(n_clusters=k).fit(arr_kmeans)

type_id = 1
type_to_int = {}
for type in types:
	type_to_int[type] = type_id
	type_id += 1

clusters = {}
for i in range(len(kmeans.labels_)):
	label = kmeans.labels_[i]
	pokemon = df.iloc[i]
	x = type_to_int[pokemon["Type 1"]]
	y = 0
	if not (pd.isnull(pokemon["Type 2"])):
		y = type_to_int[pokemon["Type 2"]]
	if label not in clusters:
		clusters[label] = ([],[])
	clusters[label][0].append(x)
	clusters[label][1].append(y)

cm = plt.get_cmap('gist_rainbow')

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_color_cycle([cm(1.*i/k) for i in range(k)])
for label in clusters:
	(x,y) = clusters[label]
	ax.scatter(x,y)
ax.set_xticks([i for i in range(k+1)])
ax.set_yticks([i for i in range(k+1)])
ax.set_title("Pokemon Clustered by Types")
ax.set_xlabel("Type 1")
ax.set_ylabel("Type 2")

print(type_to_int)
plt.show()