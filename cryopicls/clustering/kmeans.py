from sklearn.cluster import KMeans

import cryopicls


class KMeansClustering:
    """Perform K-Means clustering.

    Wrapper of K-Means implementation of scikit-learn library (sklearn.cluster.KMeans).

    Parameters
    ----------
    n_clusters : int, default=8.
        The number of clusters to form as well as the number of centroids to generate.

    init : {'k-means++', 'random'}, default='k-means++'.
        Method for initialization:
            'k-means++' : selects initial cluster centers for k-mean clustering in a smart way to speed up convergence. See section Notes in k_init for more details.
            'random' : choose n_clusters observations (rows) at random from data for the initial centroids.

    n_init : int, default=10.
        Number of time the k-means algorithm will be run with different centroid seeds. The final results will be the best output of n_init consecutive runs in terms of inertia.

    max_iter : int, default=300.
        Maximum number of iterations of the k-means algorithm for a single run.

    tol : float, default=1e-4.
        Relative tolerance with regards to Frobenius norm of the difference in the cluster centers of two consecutive iterations to declare convergence.

    random_state : int, default=None.
        Determines random number generation for centroid initialization. Use an int to make the randomness deterministic
    """

    def __init__(self, n_clusters=8, init='k-means++', n_init=10, max_iter=300, tol=1e-4,
                 random_state=None, **kwargs):
        self.n_clusters = n_clusters
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def fit(self, X):
        """Compute k-means clustering.

        Parameters
        ----------
        X : array-kile of shape (n_samples, n_latent_dims)
            Input data for clustering.

        Returns
        -------
        model
            The fitted model. (sklearn.cluster.KMeans)

        cluster_labels
            Cluster label vector of shape (n_samples, )

        cluster_centers
            Cluster center coordinates (n_clusters, n_latent_dims)
        """

        self.model_ = KMeans(
            n_clusters=self.n_clusters,
            init=self.init,
            n_init=self.n_init,
            max_iter=self.max_iter,
            tol=self.tol,
            random_state=self.random_state
        )

        self.model_.fit(X)

        self.cluster_labels_ = self.model_.labels_
        self.cluster_centers_ = self.model_.cluster_centers_

        self.print_result_summary()

        return self.model_, self.cluster_labels_, self.cluster_centers_

    def print_result_summary(self):
        """Print result summary of the fitted model.
        """

        cryopicls.clustering.utils.print_num_samples_each_cluster(self.cluster_labels_)
        print(f'Sum of squared distances: {self.model_.inertia_}')
