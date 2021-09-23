import numpy as np
import cryopicls


def parse_thresh_args(thresh, **kwargs):
    assert len(thresh) > 0
    thresh_list = []
    for thresh_val in thresh:
        z_dim, z_min, z_max = int(thresh_val[0]), float(thresh_val[1]), float(thresh_val[2])
        thresh_list.append([z_dim, z_min, z_max])
    return thresh_list


class ManualSelector:
    def __init__(self, thresh_list):
        self.thresh_list = thresh_list

    def fit(self, X):
        self.idxs_select_ = np.ones(X.shape[0], dtype=bool)
        for thresh in self.thresh_list:
            z_dim, z_min, z_max = thresh
            idxs = np.logical_and(z_min < X[:, z_dim], X[:, z_dim] < z_max)
            self.idxs_select_ = np.logical_and(idxs, self.idxs_select_)
        self.cluster_centers_ = []
        if np.sum(self.idxs_select_) == 0:
            print('Warning: No data is selected')
            self.cluster_centers_.append(np.mean(X, axis=0))
        else:
            self.cluster_centers_.append(np.mean(X[~self.idxs_select_], axis=0))
            self.cluster_centers_.append(np.mean(X[self.idxs_select_], axis=0))
        self.cluster_labels_ = self.idxs_select_.astype(int)

        return self, self.cluster_labels_, self.cluster_centers_

    def print_result_summary(self):
        print(f'Number of clusters: {len(self.cluster_centers_)}')
        cryopicls.clustering.utils.print_num_samples_each_cluster(self.cluster_labels_)
