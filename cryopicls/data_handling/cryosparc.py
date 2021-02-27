import glob
import sys
import os
import copy
import datetime

import yaml
import numpy as np


def load_cs(cs_file):
    return np.load(cs_file)


def save_cs(cs_file, cs):
    np.save(cs_file, cs)
    # np.save automatically add .npy extension, thus remove it
    os.rename(cs_file + '.npy', cs_file)


def load_csg(csg_file):
    with open(csg_file, 'r') as f:
        csg = yaml.load(f, Loader=yaml.FullLoader)
    return csg


def save_csg(csg_file, csg):
    with open(csg_file, 'w') as f:
        yaml.dump(csg, stream=f)


def get_metafiles_from_csg(csg_file):
    # Assumes the same directory as csg file
    dirpath = os.path.dirname(csg_file)

    csg = load_csg(csg_file)

    metafiles = []
    for key in csg['results'].keys():
        metafiles.append(csg['results'][key]['metafile'].replace('>', ''))
    metafiles = np.unique(metafiles)

    cs_file = None
    passthrough_file = None
    for metafile in metafiles:
        if 'passthrough_particles.cs' in metafile:
            if passthrough_file is not None and passthrough_file != metafile:
                sys.exit('More than two king of passthrough_particles.cs files found.')
            passthrough_file = os.path.join(dirpath, metafile)
        elif 'particles.cs' in metafile:
            if cs_file is not None and cs_file != metafile:
                sys.exit('More than two kind of particles.cs files found.')
            cs_file = os.path.join(dirpath, metafile)
        else:
            sys.exit('Unknown metafile found.')

    assert cs_file is not None
    # Some times there is no passthrough file.

    return cs_file, passthrough_file


def find_cryosparc_files(dir):
    """Find required cryoSPARC files from a job directory

    Parameters
    ----------
    dir : string
        Job directory

    Returns
    -------
    cs_file : string
        Particle .cs file

    csg_file : string
        Particle group .csg file

    passthrough_file : string
        Particle passthrough .cs file
    """

    assert os.path.isdir(dir), f'{dir} is not a directory.'

    cs_list = sorted(glob.glob(os.path.join(dir, 'cryosparc*_particles.cs')))
    assert len(cs_list) > 0, f'Particle cs file not found in {dir}'
    # The last cs file found in the directory.
    cs_file = cs_list[-1]

    csg_list = sorted(glob.glob(os.path.join(dir, '*_particles.csg')))
    assert len(csg_list) > 0, f'cs group file (*_particles.csg) was not found in {dir}'
    assert len(csg_list) == 1, f'*_particles.csg matched more than 1 file: {csg_list}'
    csg_file = csg_list[0]

    passthrough_list = sorted(glob.glob(os.path.join(dir, '*_passthrough_particles.cs')))
    assert len(passthrough_list) > 0, f'Particle passthrough file was not found in {dir}'
    assert len(passthrough_list) == 1, f'*_passthrough_particles.cs matched more than 1 file: {passthrough_list}'
    passthrough_file = passthrough_list[0]

    return cs_file, csg_file, passthrough_file


def load_latent_variables(infile, num_components=-1):
    """Loat latent variables from cryoSPARC 3D variability job result.

    Parameters
    ----------
    infile : string
        A particle .cs file containing the latent variables (variability components). Typically like 'cryosparc_<project id>_<job id>_particles.cs'

    num_components : int, optional
        Number of components to use. By default (-1) use all the components.

    Returns
    -------
    ndarray
        Array containing the latent variables. shape=(num_samples, num_variables)
    """

    assert os.path.exists(infile)
    cs = np.load(infile)
    Z = []
    components_mode = 0
    while True:
        if f'components_mode_{components_mode}/value' in cs.dtype.names:
            Z.append(cs[f'components_mode_{components_mode}/value'])
            components_mode += 1
        else:
            break
    assert num_components <= components_mode
    Z = np.vstack(Z).T
    Z = Z[:, :num_components]
    return Z


class CryoSPARCMetaData:
    """cryoSPARC metadata handling class.

    Parameters
    ----------
    cs : ndarray
        Array containing cryoSPARC particles .cs file contents (loaded by np.load)

    csg_template : dict
        Dictionary containing cryoSPARC particles .csg file contents (loaded by yaml.load)

    passthrough : ndarray
        Array containing cryoSPARC passthrough_particles .cs file contents (loaded by np.load)
    """

    def __init__(self, csg, cs, passthrough=None):
        self.cs = cs
        self.csg = csg
        self.passthrough = passthrough

        if self.passthrough is not None:
            assert self.cs.shape[0] == self.passthrough.shape[0]

    @classmethod
    def load(cls, csg_file):
        """Load cryoSPARC metadata from .csg file.

        Parameters
        ----------
        csgfile : string
            particles .csg file.

        Returns
        -------
        CryoSparcMetaData
            CryoSparcMetaData class instance.
        """

        csg = load_csg(csg_file)

        cs_file, passthrough_file = get_metafiles_from_csg(csg_file)

        cs = load_cs(cs_file)
        if passthrough_file:
            passthrough = load_cs(passthrough_file)
        else:
            passthrough = None

        return cls(csg, cs, passthrough)

    def write(self, outdir, outfile_rootname):
        """Save metadata in files.

        Parameters
        ----------
        outdir : string
            Output directory.

        outfile_rootname : string
            Output file rootname.
        """

        os.makedirs(outdir, exist_ok=True)

        cs_file = os.path.join(outdir, outfile_rootname + '_particles.cs')
        save_cs(cs_file, self.cs)

        if self.passthrough is not None:
            passthrough_file = os.path.join(outdir, outfile_rootname + '_passthrough_particles.cs')
            save_cs(passthrough_file, self.passthrough)
        else:
            passthrough_file = None

        csg_file = os.path.join(outdir, outfile_rootname + '_particles.csg')
        self._update_csg(cs_file, passthrough_file)
        save_csg(csg_file, self.csg)

    def _update_csg(self, cs_file, passthrough_file=None):
        """Update cs group file content.

        Parameters
        ----------
        cs_file : string
            Filename of new particles .cs file. (Directory path not required.)

        passthrough_file : string
            Filename of new passthrough_particles .cs file. (Directory path not required.)
        """

        self.csg['created'] = datetime.datetime.now()
        self.csg['group']['description'] = 'Created by cryoPICLS. cryopcls.data_handling.cryosparc.CryoSPARCMetaData._update_csg()'

        num_items = self.cs.shape[0]
        cs_basename = os.path.basename(cs_file)
        if passthrough_file:
            passthrough_basename = os.path.basename(passthrough_file)

        for key in self.csg['results'].keys():
            if 'passthrough_particles.cs' in self.csg['results'][key]['metafile']:
                assert passthrough_file is not None
                self.csg['results'][key]['metafile'] = '>' + passthrough_basename
            elif 'particles.cs' in self.csg['results'][key]['metafile']:
                self.csg['results'][key]['metafile'] = '>' + cs_basename
            else:
                sys.exit(f'Unknown metafile name in {key}: {self.csg["results"][key]["metafile"]}')
            if 'num_items' in self.csg['results'][key].keys():
                self.csg['results'][key]['num_items'] = num_items

    def iloc(self, idxs):
        """Fancy indexing.

        Parameters
        ----------
        idxs : array-like
            Indices to select.

        Returns
        -------
        CryoSPARCMetaData
            New metadata object with the selected rows.
        """

        cs = self.cs[idxs]
        if self.passthrough is not None:
            passthrough = self.passthrough[idxs]
        else:
            passthrough = None
        return self.__class__(self.csg, cs, passthrough)
