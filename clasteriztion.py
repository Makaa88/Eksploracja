import pandas
from sklearn.cluster import AgglomerativeClustering
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import seaborn as sns
import numpy as np
from scipy.special import comb
from sklearn import metrics

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

stays_mean = df['days_in_hotel'].mean()
stays_std = df['days_in_hotel'].std()
limit  = 3*stays_std
labels = ['days_in_hotel', 'all_guests']
months = ["January", "February", "March", "April", "May", "June", "July"]
#data = df.loc[:, labels[:-1]].values
data_list = []
for i in range(len(df)):
    if df["arrival_date_month"][i] in months:
        if np.fabs(df['days_in_hotel'][i] - stays_mean) > limit:
            continue
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

# for label in labels:
#     label_mean = df[label].mean()
#     print("LABEL: " + label + "  mean: " + str(label_mean))
#     data_dict = {}
#     count_dict = {}
#     for i in range(len(df)):
#         key = cluster.labels_[i]
#         if key not in data_dict:
#             data_dict[key] = 0
#             count_dict[key] = 0
#         data_dict[cluster.labels_[i]] += df[label][i]
#         count_dict[key] += 1
#
#     for i in data_dict.keys():
#         print("cluster: " + str(i) + " mean: " + str(data_dict[i]/count_dict[i]))
#     print("\n\n")


max = 0
clusters_number = 0
for no_of_clusters in range(2, 100):
    mean = 0.0
    times = 10
    for _ in range(times):
        kmeans = KMeans(n_clusters=no_of_clusters).fit(data_list)
        labels = kmeans.labels_
        mean += metrics.calinski_harabasz_score(data_list, labels)

    mean = mean/times
    if mean > max:
        max = mean
        clusters_number = no_of_clusters
    print("For the number of clusters: " + str(no_of_clusters) + " score is: " + str(mean))

print("MAX SCORE: " + str(max) + " for number of clusters: " + str(clusters_number))


# pca = PCA(n_components=2)
# pca_data = pca.fit_transform(data)
kmeans = KMeans(n_clusters=8)
result_k_means = kmeans.fit(data)
y_means = kmeans.predict(data)

plt.scatter(data[:, 0], data[:, 1], c=y_means, s=50, cmap='viridis')

plt.show()

print("Global mean days stayed: " + str(df["days_in_hotel"].mean()))
print("Global mean guests: " + str(df["all_guests"].mean()))

cluster_labels = result_k_means.labels_

cluster_means = {}

for i in range(len(data)):
    if cluster_labels[i] not in cluster_means:
        cluster_means[cluster_labels[i]] = []
    cluster_means[cluster_labels[i]].append([data[i][0], data[i][1]])

for key, values in cluster_means.items():
        days = sum([j[0]  for j in values])/len(values)
        guests = sum([j[1]  for j in values])/len(values)

        print("For cluster " + str(key) + "means are: " + str(days) + "  " + str(guests))
