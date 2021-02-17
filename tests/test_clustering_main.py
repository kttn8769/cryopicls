"""Tests that focus on reading and writing data. Only simple k-means clustering is used."""

import sys
sys.path.append('../')
import os
from cryopicls.cryopicls_clustering import main

z_file = 'tests/cryodrgn_z_p2_w1_j744_vae128_zdim3_seed1.pkl'
cryosparc_consensus = 'tests/cryosparc_consensus'
cryosparc_threedvar = 'tests/cryosparc_threedvar'
relion_consensus = 'tests/cryosparc_P2_J744_005_particles_pyem.star'
output_dir_root = 'test_results'


def test_cryodrgn_meta_cryosparc():
    """Test cryodrgn clustering using cryosparc consensus metadata"""

    com = f"cryopicls_clustering.py k-means --cryodrgn --z-file {z_file} --metadata {cryosparc_consensus} --random-state 1 --output-dir {output_dir_root}/test_cryodrgn_meta_cryosparc"
    sys.argv = com.split()
    main()


def test_cryodrgn_meta_relion():
    """Test cryodrgn clustering using relion consensus metadata"""

    com = f"cryopicls_clustering.py k-means --cryodrgn --z-file {z_file} --metadata {relion_consensus} --random-state 1 --output-dir {output_dir_root}/test_cryodrgn_meta_relion"
    sys.argv = com.split()
    main()


def test_cryosparc_threedvar():
    """Test cryosparc 3d variability analysis result clustering"""

    print('Without component selection')
    com = f"cryopicls_clustering.py k-means --cryosparc --threedvar-dir {cryosparc_threedvar} --random-state 1 --output-dir {output_dir_root}/test_cryosparc_threedvar"
    sys.argv = com.split()
    main()

    print('\n\nWith component selection --threedvar-num-components 3')
    com = f"cryopicls_clustering.py k-means --cryosparc --threedvar-dir {cryosparc_threedvar} --threedvar-num-components 3 --random-state 1 --output-dir {output_dir_root}/test_cryosparc_threedvar_components3"
    sys.argv = com.split()
    main()
