import argparse
import sys
import os


def add_general_arguments(parser):
    group = parser.add_argument_group('General arguments')
    group.add_argument(
        '--cryodrgn', action='store_true', help='Use latent representations learned by cryoDRGN.'
    )
    group.add_argument(
        '--cryosparc', action='store_true', help='Use latent representations calculated by cryoSPARC 3D variability analysis.'
    )
    group.add_argument(
        '--z-file', type=str, help='Required for --cryodrgn. The pickled file containing the learned latent representation data (z.pkl).'
    )
    group.add_argument(
        '--metadata', type=str, help='Required for --cryodrgn. If a RELION refinement was the input for cryoDRGN, specify the star file here. Else if a cryoSPARC refinement was the input for cryoDRGN, specify the job directory here.'
    )
    group.add_argument(
        '--threedvar-dir', help='Required for --cryosparc. The 3D variability job directory.'
    )
    group.add_argument(
        '--threedvar-num-components', default=-1, type=int, help='Option for --cryosparc. How many variability components to use. For example, "--3dvar-num-components 3" uses the component 0, 1 and 2 for cluster analysis. By default use all the components.'
    )
    group.add_argument(
        '--random-state', type=int, help='Random state (random seed value).'
    )
    group.add_argument(
        '--output-dir', type=str, help='Output directory. By default the current directory.'
    )
    group.add_argument(
        '--output-file-rootname', default='cryopicls', type=str, help='Output file root name.'
    )
    return parser


def add_autogmm_parser(subparsers):
    parser_autogmm = add_general_arguments(
        subparsers.add_parser('auto-gmm', formatter_class=argparse.ArgumentDefaultsHelpFormatter, help='Gaussian mixture model with automatic cluster number selection based on information criterion values. Build upon the scikit-learn implementation of GMM (https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html)')
    )
    group_autogmm = parser_autogmm.add_argument_group('Auto-GMM parameters')
    group_autogmm.add_argument(
        '--k-min', type=int, default=1, help='Minimum number of clusters.'
    )
    group_autogmm.add_argument(
        '--k-max', type=int, default=20, help='Maximum number of clusters.'
    )
    group_autogmm.add_argument(
        '--criterion', type=str, default='bic', choices=['bic', 'aic'], help='Information criterion for model selection. bic: Bayesian information criterion. aic: Akaike information criterion.'
    )
    group_autogmm.add_argument(
        '--n-init', type=int, default=10, help='The number of initializations to perform for each k. The best result are kept for each k. '
    )
    group_autogmm.add_argument(
        '--covariance-type', type=str, default='full', choices=['full', 'tied', 'diag', 'spherical'], help='Type of covariance parameters to use. "full": each component has its own general covariance matrix. "tied":     all components share the same general covariance matrix. "diag": each component has its own diagonal covariance matrix. "spherical": each component has its own single variance.'
    )
    group_autogmm.add_argument(
        '--tol', type=float, default=1e-3, help='The convergence threshold. EM iterations will stop when the lower bound average gain is below this threshold.'
    )
    group_autogmm.add_argument(
        '--reg-covar', type=float, default=1e-6, help='Non-negative regularization added to the diagonal of covariance. Allows to assure that the covariance matrices are all positive.'
    )
    group_autogmm.add_argument(
        '--max-iter', type=int, default=100, help='The number of EM iterations to perform.'
    )
    group_autogmm.add_argument(
        '--init-params', type=str, default='kmeans', choices=['kmeans', 'random'], help='The method used to initialize the weights, the means and the precisions(variances) of the components.'
    )


def add_xmeans_parser(subparsers):
    parser_xm = add_general_arguments(
        subparsers.add_parser('x-means', formatter_class=argparse.ArgumentDefaultsHelpFormatter, help='X-Means clustering. Using PyClustering implementation (class pyclustering.cluster.xmeans.xmeans)')
    )
    group_xm = parser_xm.add_argument_group('X-Means parameters')
    group_xm.add_argument(
        '--k-min', type=int, default=1, help='Minimum number of clusters.'
    )
    group_xm.add_argument(
        '--k-max', type=int, default=20, help='Maximum number of clusters.'
    )
    group_xm.add_argument(
        '--criterion', type=str, default='bic', choices=['bic', 'mndl'], help='Splitting criterion. bic: Bayesian information criterion. mndl: minimum noiseless description length.'
    )
    group_xm.add_argument(
        '--no-ccore', action='store_true', help='Use Python implementation of PyClustering library instead of C++(ccore)'
    )
    group_xm.add_argument(
        '--tolerance', type=float, default=0.025, help='Stop condition for each iteration. If maximum value of change of clusters is less than this value, algorithm will stop processing.'
    )
    group_xm.add_argument(
        '--repeat', type=int, default=10, help='How many times K-Means should be run to improve parameters. With larger repeat values suggesting higher probability of finding global optimum.'
    )
    group_xm.add_argument(
        '--alpha', type=float, default=0.9, help='Parameter distributed [0.0, 1.0] for alpha probabilistic bound. The parameter is used only in case of MNDL splitting criterion, in all other cases this value is ignored.'
    )
    group_xm.add_argument(
        '--beta', type=float, default=0.9, help='Parameter distributed [0.0, 1.0] for beta probabilistic bound. The parameter is used only in case of MNDL splitting criterion, in all other cases this value is ignored.'
    )


def add_kmeans_parser(subparsers):
    parser_km = add_general_arguments(
        subparsers.add_parser('k-means', formatter_class=argparse.ArgumentDefaultsHelpFormatter, help='K-Means clustering. Using scikit-learn implementation (class sklearn.cluster.KMeans)')
    )
    group_km = parser_km.add_argument_group('K-Means parameters')
    group_km.add_argument(
        '--n-clusters', type=int, default=8, help='The number of clusters to form as well as the number of centroids to generate.'
    )
    group_km.add_argument(
        '--init', type=str, default='k-means++', choices=['k-means++', 'random'], help='Method for initialization.'
    )
    group_km.add_argument(
        '--n-init', type=int, default=10, help='Number of time the k-means algorithm will be run with different centroid seeds. The final results will be the best output of n_init consecutive runs in terms of inertia.'
    )
    group_km.add_argument(
        '--max-iter', type=int, default=300, help='Maximum number of iterations of the k-means algorithm for a single run.'
    )
    group_km.add_argument(
        '--tol', type=float, default=1e-4, help='Relative tolerance with regards to Frobenius norm of the difference in the cluster centers of two consecutive iterations to declare convergence.'
    )


def add_gmeans_parser(subparsers):
    parser_gm = add_general_arguments(
        subparsers.add_parser('g-means', formatter_class=argparse.ArgumentDefaultsHelpFormatter, help='G-Means clustering. Using PyClustering implementation (class pyclustering.cluster.gmeans.gmeans)')
    )
    group_gm = parser_gm.add_argument_group('G-Means parameters')
    group_gm.add_argument(
        '--k-min', type=int, default=1, help='Minimum number of clusters.'
    )
    group_gm.add_argument(
        '--k-max', type=int, default=20, help='Maximum number of clusters.'
    )
    group_gm.add_argument(
        '--no-ccore', action='store_true', help='Use Python implementation of PyClustering library instead of C++(ccore)'
    )
    group_gm.add_argument(
        '--tolerance', type=float, default=1e-3, help='Stop condition for each K-Means iteration: if maximum value of change of centers of clusters is less than tolerance than algorithm will stop processing.'
    )
    group_gm.add_argument(
        '--repeat', type=int, default=3, help='Stop condition for each iteration. If maximum value of change of clusters is less than this value, algorithm will stop processing.'
    )


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__
    )
    subparsers = parser.add_subparsers(title='Clustering algorithms', dest='algorithm')

    add_autogmm_parser(subparsers)
    add_xmeans_parser(subparsers)
    add_kmeans_parser(subparsers)
    add_gmeans_parser(subparsers)

    args = parser.parse_args()
    print('##### Command #####\n\t' + ' '.join(sys.argv))
    args_print_str = '##### Input parameters #####\n'
    for opt, val in vars(args).items():
        args_print_str += '\t{} : {}\n'.format(opt, val)
    print(args_print_str)

    assert args.cryodrgn or args.cryosparc, 'Must specify either --cryodrgn or --cryosparc.'
    assert not (args.cryodrgn and args.cryosparc), '--cryodrgn and --cryosparc cannot be specified at the same time.'

    if args.cryodrgn:
        assert args.z_file is not None, 'Must specify --z-file'
        assert os.path.exists(args.z_file), f'--z-file {args.z_file} not found.'
        assert args.metadata is not None, 'Must specify --metadata'
        assert os.path.exists(args.metadata), f'--metadata {args.metadata} not found.'

    elif args.cryosparc:
        assert args.threedvar_dir is not None, 'Must specify --threedvar_dir'
        assert os.path.isdir(args.threedvar_dir), f'--threedvar-dir {args.threedvar_dir} is not a directory or does not exist.'

    if args.output_dir is None:
        # Defaults to the current directory
        args.output_dir = os.getcwd()

    return args
