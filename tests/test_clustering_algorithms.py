"""Tests the clustering algorithms. Use a toy 2D dataset of 5 gaussian blobs"""

import cryopicls
import sys
sys.path.append('../')
import pickle
import os

import matplotlib.pyplot as plt
import numpy as np
import pytest

z_file = 'tests/z_dummy_5class.pkl'


@pytest.fixture
def input():
    with open(z_file, 'rb') as f:
        Z = pickle.load(f)
    return Z


def run(model, Z, name):
    _, labels, _ = model.fit(Z)

    outdir = os.path.join('test_results', f'test_{name}')
    os.makedirs(outdir, exist_ok=True)

    plt.figure(figsize=(10, 10))
    label_list = np.unique(labels)
    for label in label_list:
        plt.scatter(Z[labels==label, 0], Z[labels==label, 1], label=label)
    plt.legend()
    plt.savefig(os.path.join(outdir, 'plot.png'), facecolor='w')

    with open(os.path.join(outdir, 'labels.pkl'), 'wb') as f:
        pickle.dump(labels, f)


def test_kmeans(input):
    model = cryopicls.clustering.kmeans.KMeansClustering(
        n_clusters=5,
        # random_state=0
    )
    run(model, input, 'k-means')


def test_xmeans(input):
    model = cryopicls.clustering.xmeans.XMeansClustering(
        # random_state=0,
        criterion='bic'
    )
    run(model, input, 'x-means-bic')

    model = cryopicls.clustering.xmeans.XMeansClustering(
        # random_state=0,
        criterion='mndl'
    )
    run(model, input, 'x-means-mndl')


def test_autogmm(input):
    model = cryopicls.clustering.autogmm.AutoGMMClustering(
        # random_state=0
    )
    run(model, input, 'auto-gmm')


def test_gmeans(input):
    model = cryopicls.clustering.gmeans.GMeansClustering(
        # random_state=0
    )
    run(model, input, 'g-means')
