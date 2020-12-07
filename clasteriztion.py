import pandas
from sklearn.cluster import AgglomerativeClustering
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import seaborn as sns
import numpy as np
from scipy.special import comb

plot_kwds = {'alpha' : 0.25, 's' : 80, 'linewidths':0}

def plot_clusters(data, labels, args, kwds):
    palette = sns.color_palette('deep', np.unique(labels).max() + 1)
    colors = [palette[x] if x >= 0 else (0.0, 0.0, 0.0) for x in np.unique(labels)]
    plt.scatter(data.T[0], data.T[1], c=colors, **plot_kwds)
    frame = plt.gca()
    frame.axes.get_xaxis().set_visible(False)
    frame.axes.get_yaxis().set_visible(False)

def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)

df = pandas.read_csv("hotel_bokings_2016.csv", sep=',')

labels = ['days_in_hotel', 'all_guests']
months = ["January", "February", "March", "April", "May", "June", "July"]
#data = df.loc[:, labels[:-1]].values
data_list = []
for i in range(len(df)):
    if df["arrival_date_month"][i] in months:
        data_list.append([df["days_in_hotel"][i], df["all_guests"][i]])

print(len(data_list))
data = np.asarray(data_list, dtype=np.float64)
agg_clustering  = AgglomerativeClustering(distance_threshold=3, n_clusters=None)
cluster = agg_clustering.fit(data)
# plt.title('Hierarchical Clustering Dendrogram')
#
# plot_dendrogram(cluster, truncate_mode='level')
# plt.show()

# print(cluster.n_clusters_)
# plot_clusters(data, cluster.n_clusters_, (), {'n_clusters':cluster.n_clusters_, 'linkage':'ward'})
# plt.show()

# plt.scatter(data.T[0], data.T[1], c='b', **plot_kwds)
# frame = plt.gca()
# frame.axes.get_xaxis().set_visible(False)
# frame.axes.get_yaxis().set_visible(False)
# plt.show()




# pca = PCA(n_components=2)
# pca_data = pca.fit_transform(data)
kmeans = KMeans(n_clusters=8)
kmeans.fit(data)
y_means = kmeans.predict(data)

plt.scatter(data[:, 0], data[:, 1], c=y_means, s=50, cmap='viridis')

plt.show()