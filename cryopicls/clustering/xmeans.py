import sys

from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from pyclustering.cluster.xmeans import xmeans, splitting_type

import cryopicls

class XMeansClustering:
    """X-Means clustering

    Wrapper of X-Means implementation of PyClustering library (pyclustering.cluster.xmeans.xmeans).

    Parameters
    ----------
    k_min : int, default 1.
        Minimum number of clusters.

    k_max : int, default 20.
        Maximum number of clusters.

    tolerance : float, default 1e-3.
        Stop condition for each iteration: if maximum value of change of centers of clusters is less than tolerance than algorithm will stop processing.

    criterion : {'bic', 'mndl'}, default 'bic'.
        Type of cluster splitting criterion.
            'bic' : Bayesian information criterion.
            'mndl' : Minimum noiseless description length.

    random_state : int, default None.
        Random seed value.

    no_ccore : bool, default True.
        Use Python implementation of PyClustering library instead of C++(ccore).

    repeat : int, default 10.
        How many times K-Means should be run to improve parameters. With larger repeat values suggesting higher probability of finding global optimum.

    alpha : float, default 0.9.
        Parameter distributed [0.0, 1.0] for alpha probabilistic bound. The parameter is used only in case of MNDL splitting criterion, in all other cases this value is ignored.

    beta : float, default 0.9.
        Parameter distributed [0.0, 1.0] for beta probabilistic bound. The parameter is used only in case of MNDL splitting criterion, in all other cases this value is ignored.
    """

    def __init__(self, k_min=1, k_max=20, tolerance=1e-3, criterion='bic',
                 random_state=None, no_ccore=False,
                 repeat=10, alpha=0.9, beta=0.9, **kwargs):
        assert k_min <= k_max
        self.k_min = k_min
        self.k_max = k_max

        if criterion == 'bic':
            self.criterion = splitting_type.BAYESIAN_INFORMATION_CRITERION
        elif criterion == 'mndl':
            self.criterion = splitting_type.MINIMUM_NOISELESS_DESCRIPTION_LENGTH
        else:
            sys.exit(f'Not supported criterion: {criterion}')

        self.tolerance=tolerance
        self.random_state = random_state
        self.ccore = not no_ccore
        self.repeat = repeat
        self.alpha = alpha
        self.beta = beta

    def fit(self, X):
        """X-Means fitting

        Parameters
        ----------
        X : array-like of shape (n_samples, n_latent_dims)
            Input data for clustering.

        Returns
        -------
        xmeans
            X-Means model instance after fit (pyclustering xmeans)

        cluster_labels
            Cluster label vector of shape (n_samples, )

        cluster_centers
            Cluster center coordinates (n_clusters, n_latent_dims)
        """

        self.X_ = X
        # self.initial_centers_ = kmeans_plusplus_initializer(
        #     self.X_,
        #     self.k_min,
        #     random_state=self.random_state).initialize()

        self.xmeans_ = xmeans(
            data=self.X_,
            # initial_centers=self.initial_centers_,
            initial_centers=None,
            tolerance=self.tolerance,
            criterion=self.criterion,
            kmax=self.k_max,
            ccore=self.ccore,
            repeat=self.repeat,
            alpha=self.alpha,
            beta=self.beta)

        self.xmeans_.process()

        self.sse_ = self.xmeans_.get_total_wce()
        self.cluster_centers_ = self.xmeans_.get_centers()
        self.cluster_labels_ = cryopicls.clustering.utils.get_clusters_in_index_labeling(
            self.xmeans_, self.X_
        )

        self.print_result_summary()

        return self.xmeans_, self.cluster_labels_, self.cluster_centers_

    def print_result_summary(self):
        """Print result summary of the fitted model.
        """

        print(f'Number of clusters: {len(self.cluster_centers_)}')
        cryopicls.clustering.utils.print_num_samples_each_cluster(self.cluster_labels_)
        print(f'Sum of squared errors: {self.sse_}')
