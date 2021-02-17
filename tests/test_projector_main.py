import sys
sys.path.append('../')
import os
from cryopicls.cryopicls_projector import main

z_file = 'tests/cryodrgn_z_p2_w1_j744_vae128_zdim3_seed1.pkl'
cryosparc_threedvar = 'tests/cryosparc_threedvar'
output_dir_root = 'test_results'


def test_cryodrgn():
    com = f"cryopicls_projector.py pca --cryodrgn --z-file {z_file} --random-state 1 --output-dir {output_dir_root}/test_projector_cryodrgn"
    sys.argv = com.split()
    main()


def test_cryosparc_threedvar():
    com = f"cryopicls_projector.py pca --cryosparc --threedvar-dir {cryosparc_threedvar} --random-state 1 --output-dir {output_dir_root}/test_projector_cryosparc"
    sys.argv = com.split()
    main()


def test_cryodrgn_umap():
    com = f"cryopicls_projector.py umap --cryodrgn --z-file {z_file} --random-state 1 --output-dir {output_dir_root}/test_projector_cryodrgn_umap --n-neighbors 15 --n-components 2 --metric euclidean --min-dist 0.1"
    sys.argv = com.split()
    main()
