# **************************************************************************
# *
# * Authors:     you (you@yourinstitution.email)
# *
# * your institution
# *
# * This program is free software; you can redistribute it and/or modify
# * it under the terms of the GNU General Public License as published by
# * the Free Software Foundation; either version 2 of the License, or
# * (at your option) any later version.
# *
# * This program is distributed in the hope that it will be useful,
# * but WITHOUT ANY WARRANTY; without even the implied warranty of
# * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# * GNU General Public License for more details.
# *
# * You should have received a copy of the GNU General Public License
# * along with this program; if not, write to the Free Software
# * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
# * 02111-1307  USA
# *
# *  All comments concerning this program package may be sent to the
# *  e-mail address 'scipion@cnb.csic.es'
# *
# **************************************************************************

import pwem
import os
from pyworkflow.utils import Environ
from .constants import DEEPICT_HOME, VERSION, DEEPICT, DEEPICT_ENV_NAME, \
DEFAULT_ACTIVATION_CMD, DEEPICT_CUDA_LIB, DEEPICT_ENV_ACTIVATION

_logo = "icon.png"
_references = ['deteresa2022']
__version__ = "0"


class Plugin(pwem.Plugin):

    _homeVar = DEEPICT_HOME
    _pathVars = [DEEPICT_HOME]
    #_url = 'https://github.com/scipion-em/scipion-em-deepict'

    @classmethod
    def _defineVariables(cls):
        # DeePiCt does NOT need EmVar because it uses a conda environment.
        cls._defineVar(DEEPICT, DEFAULT_ACTIVATION_CMD)
        cls._defineEmVar(DEEPICT_HOME, 'DeePiCt-' + VERSION)

    @classmethod
    def getDeepictEnvActivation(cls):
        return cls.getVar(DEEPICT_ENV_ACTIVATION)

    @classmethod
    def getEnviron(cls, gpuId='0'):
        """ Setup the environment variables needed to launch Deepict. """
        environ = Environ(os.environ)
        if 'PYTHONPATH' in environ:
            # this is required for python virtual env to work
            del environ['PYTHONPATH']

        #environ.update({'CUDA_VISIBLE_DEVICES': gpuId})

        #cudaLib = environ.get(DEEPICT_CUDA_LIB, pwem.Config.CUDA_LIB)
        #environ.addLibrary(cudaLib)
        return environ

    @classmethod
    def defineBinaries(cls, env):
        DEEPICT_INSTALLED = '%s_%s_installed' % (DEEPICT, VERSION)

        # try to get CONDA activation command
        installationCmd = cls.getCondaActivationCmd()

        # Create the environment
        installationCmd += ' git clone https://github.com/ZauggGroup/DeePiCt.git && '

        installationCmd += 'conda create -y -n %s python=3.7 -c conda-forge -c anaconda && ' % DEEPICT_ENV_NAME
        installationCmd += 'conda activate %s && ' % DEEPICT_ENV_NAME
        installationCmd += 'conda install -y cudatoolkit==11.8 -c conda-forge &&'

        installationCmd += 'pip install torch torchvision torchaudio &&'
        installationCmd += 'pip install pandas && '
        installationCmd += 'pip install mrcfile && '
        installationCmd += 'pip install scipy && '
        installationCmd += 'pip install scikit-image && '
        installationCmd += 'pip install pyyaml && '
        installationCmd += 'pip install tqdm && '
        installationCmd += 'pip install h5py && '
        installationCmd += 'pip install tensorboardX && '


        # Activate new the environment
        installationCmd += 'conda activate %s && ' % DEEPICT_ENV_NAME

        installationCmd += 'touch %s' % DEEPICT_INSTALLED
        
        deeepict_commands = [(installationCmd, DEEPICT_INSTALLED)]

        envPath = os.environ.get('PATH', "")  # keep path since conda likely in there
        installEnvVars = {'PATH': envPath} if envPath else None

        env.addPackage(DEEPICT,
                       version=VERSION,
                       tar='void.tgz',
                       commands=deeepict_commands,
                       neededProgs=cls.getDependencies(),
                       vars=installEnvVars,
                       default=bool(cls.getCondaActivationCmd()))

    @classmethod
    def getDependencies(cls):
        # try to get CONDA activation command
        condaActivationCmd = cls.getCondaActivationCmd()
        neededProgs = []
        if not condaActivationCmd:
            neededProgs.append('conda')

        return neededProgs

    @classmethod
    def activatingDeepict(cls):
        return 'conda activate %s ' % DEEPICT_ENV_NAME

    @classmethod
    def runDeepict(cls, protocol, program, args, cwd=None, gpuId='0'):
        """ Run DeePict command from a given protocol. """
        script = os.path.join(cls.getHome(), args)
        fullProgram = '%s %s && %s' % (cls.getCondaActivationCmd(), cls.activatingDeepict(), program)
        protocol.runJob(fullProgram, script, env=cls.getEnviron(gpuId=gpuId), cwd=cwd, numberOfMpi=1)