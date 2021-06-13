import argparse
import sys
import pathlib
import os


def get_absolute_path(path_str):
    return str(pathlib.Path(path_str).resolve())


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Auto-refine of cluster particles with cryoSPARC.'
    )
    parser.add_argument(
        '--cryopicls-result-dir', required=True, type=get_absolute_path,
        help='Path to the cryoPICLS result direcotry.'
    )
    parser.add_argument(
        '--cryopicls-result-basename', required=True, type=str,
        help='File basename of cryoPICLS results. For example, if you have cryopicls_cluster000_particles.csg, then this parameter should be "cryopicls".'
    )
    parser.add_argument(
        '--cache-dir', required=True, type=get_absolute_path,
        help='Path to the cache directory in cryoSPARC project directory. Must be writable.'
    )
    parser.add_argument(
        '--ssh-user', required=True, type=str,
        help='User name for ssh login into cryoSPARC master node.'
    )
    parser.add_argument(
        '--ssh-host', required=True, type=str,
        help='cryoSPARC master node hostname.'
    )
    parser.add_argument(
        '--ssh-port', default=22, type=int,
        help='Port number for ssh login into cryoSPARC master node.'
    )
    parser.add_argument(
        '--csparc-lane', default='default', type=str,
        help='cryoSPARC lane to use.'
    )
    parser.add_argument(
        '--csparc-user-email', required=True, type=str,
        help='E-mail address of cryoSPARC user.'
    )
    parser.add_argument(
        '--csparc-project-uid', required=True, type=str,
        help='cryoSPARC project uid. (such as P1, P2, ...)'
    )
    parser.add_argument(
        '--csparc-workspace-uid', default='', type=str,
        help='cryoSPARC workspace uid. (such as W1, W2, ...)  If not specified, a new workspace will be created.'
    )
    parser.add_argument(
        '--csparc-workspace-title', default='', type=str,
        help='Title for new workspace.'
    )
    parser.add_argument(
        '--csparc-refine-symmetry', default='C1', type=str,
        help='Point group symmetry of reconstruction and refinement.'
    )
    parser.add_argument(
        '--csparc-abinitio', action='store_true',
        help='Do ab-initio reconstruction instead of reconstruction only.'
    )
    parser.add_argument(
        '--csparc-abinitio-symmetry', default='C1', type=str,
        help='Point group symmetry of ab-initio reconstruction.'
    )
    parser.add_argument(
        '--csparc-consensus-job-uid', required=True, type=str,
        help='cryoSPARC job uid (such as J1, J2, ...) of the consensus reconstruction job.'
    )

    args = parser.parse_args()
    print('##### Command #####\n\t' + ' '.join(sys.argv))
    args_print_str = '##### Input parameters #####\n'
    for opt, val in vars(args).items():
        args_print_str += '\t{} : {}\n'.format(opt, val)
    print(args_print_str)

    assert os.path.isdir(args.cryopicls_result_dir)
    assert os.path.isdir(args.cache_dir) and os.access(args.cache_dir, os.W_OK)

    return args
