import numpy as np

from pyclustering.cluster.encoder import cluster_encoder, type_encoding


def get_clusters_in_index_labeling(model, data):
    """Get clusters in index labeling style (same as sklearn) from pyclustering model

    Parameters
    ----------
    model : pyclustering model
        Fitted model

    data : array-like
        Data used to fit the model

    Returns
    -------
    clusters
        Labeling result in index labeling style (sklearn style)
    """

    clusters_org = model.get_clusters()
    if model.get_cluster_encoding() != type_encoding.CLUSTER_INDEX_LABELING:
        enc = cluster_encoder(model.get_cluster_encoding(), clusters_org, data)
        enc.set_encoding(type_encoding.CLUSTER_INDEX_LABELING)
        clusters = enc.get_clusters()
    else:
        clusters = clusters_org
    return clusters


def print_num_samples_each_cluster(cluster_labels):
    """Print the number of samples in each cluster

    Parameters
    ----------
    cluster_labels : array-like of shape (n_samples, )
        Cluster labels vector in index labeling style (Each element is cluster label)
    """

    print('Number of samples in each cluster:')
    for label in np.unique(cluster_labels):
        num = np.sum(cluster_labels == label)
        print(f'    cluster {label:03d} : {num:6d}')
