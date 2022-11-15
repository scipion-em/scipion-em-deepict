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
    #_url = 'https://github.com/scipion-em/scipion-em-deepict'

    @classmethod
    def _defineVariables(cls):
        # cryoCARE does NOT need EmVar because it uses a conda environment.
        cls._defineVar(DEEPICT, DEFAULT_ACTIVATION_CMD)

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

        installationCmd += 'conda create -y -n %s -c conda-forge -c anaconda python=3.8 && ' % DEEPICT_ENV_NAME
        installationCmd += 'conda install -y pandas && '
        installationCmd += 'pip install mrcfile && '
        installationCmd += 'pip install scipy && '
        #installationCmd += 'conda install -n %s -c conda-forge mamba && ' % DEEPICT_ENV_NAME

        # Activate new the environment
        installationCmd += 'conda activate %s && ' % DEEPICT_ENV_NAME

        
        #installationCmd += 'pip install tensorboardX && '

        # Install non-conda required packages
        #installationCmd += 'mamba create -c conda-forge -c bioconda -n snakemake snakemake==5.13.0 && '
        #installationCmd += 'conda activate snakemake && '
        
        #installationCmd += 'conda install -c -y pytorch pytorch torchvision && '
        #installationCmd += 'conda install -c -y anaconda keras-gpu=2.3.1'
        
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
    def runDeepict(cls, protocol, program, args, cwd=None, gpuId='0'):
        """ Run DeePict command from a given protocol. """
        print("init --> runDeepict") 
        print(cls.getCondaActivationCmd())
        fullProgram = '%s %s' % (cls.getCondaActivationCmd(),
                                       program)
        protocol.runJob(fullProgram, args, env=cls.getEnviron(gpuId=gpuId), cwd=cwd, numberOfMpi=1)