import numpy as np

from sklearn.mixture import GaussianMixture

import cryopicls


class AutoGMMClustering:
    """Gaussian mixture model with automatic cluster number selection based on information criterion values.

    Build upon the scikit-learn implementation of GMM (https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html)

    Parameters
    ----------
    k_min : int
        Minimum number of clusters.

    k_max : int
        Maximum number of clusters.

    criterion : {'aic', 'bic'}
        Information criterion for model selection.

            'bic' : Bayesian information criterion.
            'aic' : Akaike information criterion.

    n_init : int
        The number of initializations to perform for each k. The best result are kept for each k.

    random_state : int
        Random seed value.

    covariance_type : {'full', 'tied', 'diag', 'spherical'}
        Type of covariance parameters to use.

            'full' : Each component has its own general covariance matrix.
            'tied' : All components share the same general covariance matrix.
            'diag' : Each component has its own diagonal covariance matrix.
            'spherical' : Each component has its own single variance.

    tol : float
        The convergence threshold. EM iterations will stop when the lower bound average gain is below this threshold.

    reg_covar : float
        Non-negative regularization added to the diagonal of covariance. Allow to assure that the covariance matrices are all positive.

    max_iter : int
        The number of EM iterations to perform.

    init_params : {'kmeans', 'random'}
        The method used to initialize the weights, the means and the precisions (variances) of the components.

    Attributes
    ----------
    aic_list_ : list of float
        AIC values of each k.

    bic_list_ : ist of float
        BIC values of each k.

    k_list_ : list of float
        List of k values.

    gm_list_ : list of Gaussian Mixture instances
        List of each fitted model

    k_fit_ : int
        k (the number of clusters) of the best model

    ic_fit_ : float
        Information criterion value of the best model

    elbo_fit_ : float
        Evidence lower bound value of the best model

    gm_fit_ : GaussianMixture instance
        The best GMM model

    cluster_labels_ : ndarray of shape (n_samples, )
        Array of cluster labels.

    cluster_centers_ : ndarray of shape (k_fit_, n_latent_dims)
        Array of cluster center coordinates.
    """

    def __init__(self, k_min=1, k_max=20, n_init=10, criterion='bic', random_state=None,
                 covariance_type='full', tol=1e-3, reg_covar=1e-6, max_iter=100,
                 init_params='kmeans', **kwargs):
        assert k_min <= k_max
        self.k_min = k_min
        self.k_max = k_max
        self.n_init = n_init
        assert criterion in ['bic', 'aic'], f'Not supported criterion: {criterion}'
        self.criterion = criterion
        self.random_state = random_state
        assert covariance_type in ['full', 'tied', 'diag', 'spherical'], f'Not supported covariance_type: {covariance_type}'
        self.covariance_type = covariance_type
        self.tol = tol
        self.reg_covar = reg_covar
        self.max_iter = max_iter
        self.init_params = init_params

    def fit(self, X):
        """Auto GMM fitting

        Parameters
        ----------
        X : array-like of shape (n_samples, n_latent_dims)
            Input data for clustering

        Returns
        -------
        gm_fit
            The best GMM model instance (sklearn GaussianMixture)

        cluster_labels
            Cluster label vector of shape (n_samples, )

        cluster_centers
            Cluster center coordinates (n_clusters, n_latent_dims)
        """

        self.aic_list_ = []
        self.bic_list_ = []
        self.k_list_ = np.arange(self.k_min, self.k_max + 1)
        self.gm_list_ = []
        for k in self.k_list_:
            gm = GaussianMixture(
                n_components=k, n_init=self.n_init, random_state=self.random_state,
                covariance_type=self.covariance_type, tol=self.tol, reg_covar=self.reg_covar,
                max_iter=self.max_iter, init_params=self.init_params)
            print(f'Fitting GMM K={k}...')
            gm.fit(X)
            self.gm_list_.append(gm)
            self.aic_list_.append(gm.aic(X))
            self.bic_list_.append(gm.bic(X))

        # Model selection
        if self.criterion == 'bic':
            best_idx = np.argmin(self.bic_list_)
            self.ic_fit_ = self.bic_list_[best_idx]
        elif self.criterion == 'aic':
            best_idx = np.argmin(self.aic_list_)
            self.ic_fit_ = self.aic_list_[best_idx]
        self.k_fit_ = self.k_list_[best_idx]
        self.gm_fit_ = self.gm_list_[best_idx]
        self.elbo_fit_ = self.gm_fit_.lower_bound_

        self.cluster_labels_ = self.gm_fit_.predict(X)
        self.cluster_centers_ = self.gm_fit_.means_

        self.print_result_summary()

        return self.gm_fit_, self.cluster_labels_, self.cluster_centers_

    def print_result_summary(self):
        """Print result summary of the best fitted model.
        """

        print(f'Number of clusters: {len(self.cluster_centers_)}')
        cryopicls.clustering.utils.print_num_samples_each_cluster(self.cluster_labels_)
        print(f'Information criterion ({self.criterion}): {self.ic_fit_}')
        print(f'Evidence lower bound: {self.elbo_fit_}')
