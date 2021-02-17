import numpy as np


def nearest_in_array(arr, query):
    """Return the nearest sample to query in arr

    Parameters
    ----------
    arr : array-like of shape (n_samples, n_latent_dims)
        Array

    query : array-like of shape (n_latent_dims, )
        Query array

    Returns
    -------
    idx
        Index of the nearest sample

    val
        The nearest sample
    """
    # Assuming arr size of NxD, query size of D (vector)
    idx = np.argmin(np.sum(np.square(arr - query), axis=1))
    val = arr[idx]
    return idx, val
