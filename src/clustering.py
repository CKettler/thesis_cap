import sklearn.cluster as cluster
import numpy as np
import random
import matplotlib.pylab as plt

class clustering:
    def __init__(self, df_all, used_columns):
        self.df_all = df_all
        self.used_columns = used_columns

    def dtw_clustering(self):
        """
        Dynamic Time Warping
        :return:
        """
        # used_columns = [col for col in self.df_all.columns if 'stroke' in col and '-' not in col and '2000' not in col]
        cluster_df = self.df_all.ix[:, self.used_columns]
        cluster_data = cluster_df.as_matrix(columns=None)
        # train = cluster_data[:2000,:]
        # test = cluster_data[2000:,:]
        #
        # data = np.vstack((train[:, :-1], test[:, :-1]))

        centroids = k_means_clust(cluster_data, 200, 10)
        for i in centroids:
            plt.plot(i)
        plt.show()
        assignments = {}
        for ind, i in enumerate(cluster_data):
            min_dist = float('inf')
            closest_clust = None
            for c_ind, j in enumerate(centroids):
                if lb_keogh(i, j, 2) < min_dist:
                    cur_dist = dtw_distance(i, j)
                    if cur_dist < min_dist:
                        min_dist = cur_dist
                        closest_clust = c_ind
            if closest_clust in assignments:
                assignments[closest_clust].append(ind)
            else:
                assignments[closest_clust] = [ind]

        for clust_nr, clust_indexes in assignments.iteritems():
            for index in clust_indexes:
                self.df_all.loc[index, 'label'] = clust_nr
        labels = self.df_all.loc[:, 'label']
        return centroids, self.df_all


def dtw_distance(s1, s2):

    dtw = {}

    for i in range(len(s1)):
        dtw[(i, -1)] = float('inf')
    for i in range(len(s2)):
        dtw[(-1, i)] = float('inf')
    dtw[(-1, -1)] = 0

    for i in range(len(s1)):
        for j in range(len(s2)):
            dist = np.power((s1[i] - s2[j]),2)
            dtw[(i, j)] = dist + min(dtw[(i - 1, j)], dtw[(i, j - 1)], dtw[(i - 1, j - 1)])

    return np.sqrt(dtw[len(s1) - 1, len(s2) - 1])

def lb_keogh(s1, s2, r):
    """
    Credit to:  Dr. Eamonn Keogh's group at UC Riverside
    :param s1: sequence 1
    :param s2: sequence 2
    :param r: reach, number of points of the sequence which belong to one part of the lb_keogh representation
    :return:
    """
    lb_sum = 0
    for ind, i in enumerate(s1):

        lower_bound = min(s2[(ind - r if ind - r >= 0 else 0):(ind + r)])
        upper_bound = max(s2[(ind - r if ind - r >= 0 else 0):(ind + r)])

        if i > upper_bound:
            lb_sum = lb_sum + (i - upper_bound) ** 2
        elif i < lower_bound:
            lb_sum = lb_sum + (i - lower_bound) ** 2

    return np.sqrt(lb_sum)


def k_means_clust(data, num_clust, num_iter):
    centroids = random.sample(data, num_clust)
    counter = 0
    for n in range(num_iter):
        counter += 1
        print('iteration: %s' % n)
        assignments = {}
        # assign data points to clusters
        for ind, i in enumerate(data):
            min_dist = float('inf')
            closest_clust = None
            for c_ind, j in enumerate(centroids):
                if lb_keogh(i, j, 2) < min_dist:
                    cur_dist = dtw_distance(i, j)
                    if cur_dist < min_dist:
                        min_dist = cur_dist
                        closest_clust = c_ind
            if closest_clust in assignments:
                assignments[closest_clust].append(ind)
            else:
                assignments[closest_clust] = []

        # recalculate centroids of clusters
        for key in assignments:
            clust_sum = 0
            for k in assignments[key]:
                # print('key: %s' % key)
                # print('nr assignment: %s' % k)
                clust_sum = clust_sum + data[k]
                # print('clust sum: %s' % clust_sum)
            try:
                centroids[key] = [m / len(assignments[key]) for m in clust_sum]
            except:
                print('clust sum is an index instead of list')

    return centroids

