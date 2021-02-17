import glob
import sys
import os
import copy
import datetime

import yaml
import numpy as np


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

    csfile : string
        Particles .cs file name

    csgfile : string
        Particles group .csg file name

    passthroughfile : string
        Particles passthrough .cs file name
    """

    def __init__(self, cs, csg_template, passthrough, csfile=None, csgfile=None, passthroughfile=None):
        self.cs = cs
        self.csg_template = csg_template
        self.passthrough = passthrough
        self.csfile = csfile
        self.csgfile = csgfile
        self.passthroughfile = passthroughfile

    @classmethod
    def load(cls, csfile, csgfile, passthroughfile):
        """Load cryoSPARC metadata from files.

        Parameters
        ----------
        csfile : string
            particles .cs file.

        csgfile : string
            particles .csg file.

        passthroughfile : string
            passthrough_particles .cs file.

        Returns
        -------
        CryoSparcMetaData
            CryoSparcMetaData class instance.
        """

        cs = np.load(csfile)
        with open(csgfile, 'r') as f:
            csg_template = yaml.load(f, Loader=yaml.FullLoader)
        passthrough = np.load(passthroughfile)
        return cls(cs, csg_template, passthrough, csfile, csgfile, passthroughfile)

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
        csfile = os.path.join(outdir, outfile_rootname + '_particles.cs')
        passthroughfile = os.path.join(outdir, outfile_rootname + '_passthrough_particles.cs')
        csgfile = os.path.join(outdir, outfile_rootname + '_particles.csg')

        np.save(csfile, self.cs)
        # np.save forces .npy extension...
        os.rename(csfile + '.npy', csfile)
        np.save(passthroughfile, self.passthrough)
        os.rename(passthroughfile + '.npy', passthroughfile)
        csg = self._create_csg(csfile, passthroughfile)
        with open(csgfile, 'w') as f:
            yaml.dump(csg, stream=f)

    def _create_csg(self, csfile, passthroughfile):
        """Create cs group file content from csg_template.

        Parameters
        ----------
        csfile : string
            Filename of particles .cs file.

        passthroughfile : string
            Filename of passthrough_particles .cs file.

        Returns
        -------
        dict
            Dictionary containing cs group file contents. Save this with yaml.dump as .csg file, then it will be a valid cs group file.
        """

        csg = copy.deepcopy(self.csg_template)
        csg['created'] = datetime.datetime.now()
        csg['group']['description'] = 'Created by cryoPICLS. cryopcls.data_handling.cryosparc.CryoSPARCMetaData._create_csg()'

        num_items = self.cs.shape[0]
        cs_basename = os.path.basename(csfile)
        passthrough_basename = os.path.basename(passthroughfile)
        for key in csg['results'].keys():
            if 'passthrough_particles.cs' in csg['results'][key]['metafile']:
                csg['results'][key]['metafile'] = '>' + passthrough_basename
            elif 'particles.cs' in csg['results'][key]['metafile']:
                csg['results'][key]['metafile'] = '>' + cs_basename
            else:
                sys.exit(f'Unknown metafile name in {key}: {csg["results"][key]["metafile"]}')
            if 'num_items' in csg['results'][key].keys():
                csg['results'][key]['num_items'] = num_items
        return csg

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
        passthrough = self.passthrough[idxs]
        return self.__class__(cs, self.csg_template, passthrough)
