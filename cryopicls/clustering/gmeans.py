from pyclustering.cluster.gmeans import gmeans

import cryopicls


class GMeansClustering:
    """G-Means clustering

    Wrapper of G-Means implementation of PYClustering library (pyclustering.cluster.gmeans.gmeans).

    Parameters
    ---------
    k_min : int, default 1.
        Minimum number of clusters.

    k_max : int, default 20.
        Maximum number of clusters.

    no_ccore : bool, default True.
        Use Python implementation of PyClustering library instead of C++(ccore).

    tolerance : float, default 1e-3.
        Stop condition for each K-Means iteration: if maximum value of change of centers of clusters is less than tolerance than algorithm will stop processing.

    repeat : int, default 3.
        How many times K-Means should be run to improve parameters. With larger 'repeat' values suggesting higher probability of finding global optimum.

    random_state : int, default None.
        Random seed value.
    """

    def __init__(self, k_min=1, no_ccore=False, tolerance=1e-3, repeat=3,
                 k_max=20, random_state=None, **kwargs):
        self.k_min = k_min
        self.ccore = not no_ccore
        self.tolerance = tolerance
        self.repeat = repeat
        self.k_max = k_max
        self.random_state = random_state

    def fit(self, X):
        """G-Means fitting

        Parameters
        ----------
        X : array-like of shape (n_samples, n_latent_dims)
            Input data for clustering.

        Returns
        -------
        model
            G-Means model instance after fit (pyclustering.cluster.gmeans.gmeans)

        cluster_labels
            Cluster label vector of shape (n_samples, )

        cluster_centers
            Cluster center coordinates (n_clusters, n_latent_dims)
        """

        self.model_ = gmeans(
            X,
            k_init=self.k_min,
            ccore=self.ccore,
            tolerance=self.tolerance,
            repeat=self.repeat,
            k_max=self.k_max,
            random_state=self.random_state
        )

        self.model_.process()

        self.sse_ = self.model_.get_total_wce()
        self.cluster_centers_ = self.model_.get_centers()
        self.cluster_labels_ = cryopicls.clustering.utils.get_clusters_in_index_labeling(
            self.model_, X
        )

        self.print_result_summary()

        return self.model_, self.cluster_labels_, self.cluster_centers_

    def print_result_summary(self):
        """Print result summary of the fitted model.
        """

        print(f'Number of clusters: {len(self.cluster_centers_)}')
        cryopicls.clustering.utils.print_num_samples_each_cluster(self.cluster_labels_)
        print(f'Sum of squared errors: {self.sse_}')
