import argparse
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


def add_umap_parser(subparsers):
    parser_umap = add_general_arguments(
        subparsers.add_parser('umap', formatter_class=argparse.ArgumentDefaultsHelpFormatter, help='Perform UMAP projection. See the UMAP documentation for details: https://umap-learn.readthedocs.io/en/latest/parameters.html')
    )
    group_umap = parser_umap.add_argument_group('UMAP parameters')
    group_umap.add_argument(
        '--n-neighbors', type=int, default=15, help='The size of local neighborhood (in terms of number of neighboring sample points) used for manifold approximation.'
    )
    group_umap.add_argument(
        '--n-components', type=int, default=2, help='The dimension of the space to embed into.'
    )
    group_umap.add_argument(
        '--metric', type=str, default='euclidean', help='The metric to use to compute distances in high dimensional space.'
    )
    group_umap.add_argument(
        '--min-dist', type=float, default=0.1, help='The effective minimum distance between embedded points.'
    )


def add_pca_parser(subparsers):
    parser_pca = add_general_arguments(
        subparsers.add_parser('pca', formatter_class=argparse.ArgumentDefaultsHelpFormatter, help='Perform PCA projection. See the scikit-learn documentation for details: https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html')
    )
    group_pca = parser_pca.add_argument_group('PCA parameters')
    group_pca.add_argument(
        '--n-components', type=int, default=None, help='Number of components to keep. Defaults (None) keeps all components.'
    )


def parser_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__
    )
    subparsers = parser.add_subparsers(title='Projection algorithms', dest='algorithm')

    add_umap_parser(subparsers)
    add_pca_parser(subparsers)

    args = parser.parse_args()
    args_print_str = '##### Input parameters #####\n'
    for opt, val in vars(args).items():
        args_print_str += '\t{} : {}\n'.format(opt, val)
    print(args_print_str)

    assert args.cryodrgn or args.cryosparc, 'Must specify either --cryodrgn or --cryosparc.'
    assert not (args.cryodrgn and args.cryosparc), '--cryodrgn and --cryosparc cannot be specified at the same time.'

    if args.cryodrgn:
        assert args.z_file is not None, 'Must specify --z-file'
        assert os.path.exists(args.z_file), f'--z-file {args.z_file} not found.'

    elif args.cryosparc:
        assert args.threedvar_dir is not None, 'Must specify --threedvar_dir'
        assert os.path.isdir(args.threedvar_dir), f'--threedvar-dir {args.threedvar_dir} is not a directory or does not exist.'

    if args.output_dir is None:
        # Defaults to the current directory
        args.output_dir = os.getcwd()

    return args
