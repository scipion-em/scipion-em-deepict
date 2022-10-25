# -*- coding: utf-8 -*-
# **************************************************************************
# *
# * Authors:     Daniel Prieto (daniel.prietof@estudiante.uam.es)
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


"""
Describe your python module here:
This module will provide the traditional Hello world example
"""
from email.policy import default

#from sqlalchemy import true
from pyworkflow.protocol import Protocol, params, Integer
from pyworkflow.utils import Message
from pyworkflow.protocol import EnumParam, IntParam, FloatParam, BooleanParam, LT, GT
from scipion.constants import PYTHON
import csv
#import yaml

from deepict import Plugin


class DeepictSegmentation(Protocol):
    """
    TODO resumir
    Cryo-electron tomograms capture a wealth of structural information on the molecular constituents
    of cells and tissues. We present DeePiCt (Deep Picker in Context), an open-source deep-learning
    framework for supervised structure segmentation and macromolecular complex localization in
    cellular cryo-electron tomography. To train and benchmark DeePiCt on experimental data, we
    comprehensively annotated 20 tomograms of Schizosaccharomyces pombe for ribosomes, fatty acid
    synthases, membranes, nuclear pore complexes, organelles and cytosol. By comparing our method to
    state-of-the-art approaches on this dataset, we show its unique ability to identify low-abundance
    and low-density complexes. We use DeePiCt to study compositionally-distinct subpopulations of
    cellular ribosomes, with emphasis on their contextual association with mitochondria and the
    endoplasmic reticulum. Finally, by applying pre-trained networks to a HeLa cell dataset, we
    demonstrate that DeePiCt achieves high-quality predictions in unseen datasets from different
    biological species in a matter of minutes. The comprehensively annotated experimental data and
    pre-trained networks are provided for immediate exploitation by the community.
    """
    _label = 'Segmentation'
    RIBOSOME    = 0
    MEMBRANE    = 1
    MICROTUBULE = 2
    FAS         = 3

    INTERSECTION    = 0
    CONTACT         = 1
    COLOCALIZATION  = 2

    AMP_SPECTRUM_FN     = 'amp_spectrum.tsv'
    FILTERED_TOMO_FN    ='filtered_tomo.mrc'

    #EXTRA_PATH = self._getExtraPath() -> parametro dentro concatena a la ruta
    #TODO Icono
    DEEPICT_TEMPORAL_PATH = '/home/kdna/opt/scipion/software/em/DeePiCt-0/DeePiCt/3d_cnn/src'

    # -------------------------- DEFINE param functions ----------------------
    def _defineParams(self, form):
        """ Define the input parameters that will be used.
        Params:
            form: this is the form to be populated with sections and params.
        """
        # You need a params to belong to a section:
        form.addSection(label=Message.LABEL_INPUT)
        form.addParam('inputTomogram', params.PointerParam,
                      pointerClass='SetOfTomograms',
                      label='Even tomograms',
                      important=True,
                      allowsNull=False,
                      help='Set of tomogram reconstructed from the even frames of the tilt'
                           'series movies.')
        
        form.addParam('mask', params.PointerParam,
                      pointerClass='SetOfTomograms',
                      label='Mask',
                      important=True,
                      allowsNull=False,
                      help='Set of tomogram reconstructed from the even frames of the tilt'
                           'series movies.')

        form.addParam('tomogramOption',
                      EnumParam,
                      choices=['ribosome', 'membrane', 'microtubule', 'FAS'],
                      default=self.MEMBRANE,
                      label='Resize option',
                      isplay=EnumParam.DISPLAY_COMBO,
                      help='Choose the model based on what you want to segment. \n '
                           'The available models are prediction for membrane, ribosome, microtubules, and FAS.')
        
        #TODO parametros de POST PROCESSING
        form.addSection(label='Post-processing')
        form.addParam('threshold',
                      FloatParam,
                      label='Threshold',
                      default=0.5,
                      validators=[GT(0), LT(1)],
                      help='TODO')

        form.addParam('minClusterSize',
                      IntParam,
                      label='Min cluster size',
                      default=500,
                      help='TODO')

        form.addParam('maxClusterSize',
                      IntParam,
                      label='Max cluster size',
                      default=0,
                      help='TODO')
        
        form.addParam('clusteringConnectivity',
                      IntParam,
                      label='Clustering connectivity',
                      default=1,
                      help='TODO')
        
        form.addParam('calculateMotl',
                      BooleanParam,
                      label='Calculate motl',
                      help='TODO')

        form.addParam('contactMode',
                      EnumParam,
                      choices=['intersection', 'contact', 'colocalization'],
                      default=self.INTERSECTION,
                      label='Contact mode',
                      isplay=EnumParam.DISPLAY_COMBO,
                      help='TODO Choose the model based on what you want to segment. \n '
                           'The available models are prediction for membrane, ribosome, microtubules, and FAS.')
        
        form.addParam('contactDistance',
                      IntParam,
                      label='Contact distance',
                      default=0,
                      help='TODO')


    # --------------------------- STEPS functions ------------------------------
    def _insertAllSteps(self):
        # Insert processing steps
        self._insertFunctionStep('extractSpectrumStep')
        self._insertFunctionStep('matchSpectrumStep')

    def extractSpectrumStep(self):
        # python extract_spectrum.py --input <input_tomo.mrc> --output <amp_spectrum.tsv>

        Plugin.runDeepict(self, PYTHON, '$(which extract_spectrum.py) --input %s --output %s'
                        % (self.inputTomogram.get().getFileName(),
                           self._getExtraPath(self.AMP_SPECTRUM_FN)))

    def matchSpectrumStep(self):
        # say what the parameter says!!

        Plugin.runDeepict(self, PYTHON, '$(which match_spectrum.py) --input %s --target %s --output %s'
                        % (self.inputTomogram.get().getFileName(),
                           self._getExtraPath(self.AMP_SPECTRUM_FN), self._getExtraPath(self.FILTERED_TOMO_FN)))

    #TODO crear nuevos steps (punto 3 del notebook)
    def splitIntoPatchesStep(self):
        # Create the 64^3 patches
        # TODO 
        # preguntar params
        Plugin.runDeepict(self, PYTHON, '$(which generate_prediction_partition.py) --config_file %s --pythonpath %s --tomo_name %s'
                        % (self.inputTomogram.get().getFileName(),
                           self.DEEPICT_TEMPORAL_PATH, self.inputTomogram.get().getFileName()))

    def segmentStep(self):
        # Create the segmentation of the 64^3 patches
        # TODO 
        # preguntar params
        Plugin.runDeepict(self, PYTHON, '$(which segment.py) --config_file %s --pythonpath %s --tomo_name %s --gpu 0'
                        % (self.inputTomogram.get().getFileName(),
                           self.DEEPICT_TEMPORAL_PATH, self._getExtraPath(self.FILTERED_TOMO_FN)))
        
    def assemblePredictionStep(self):
        # Assemnble the segmentated patches
        # TODO 
        # preguntar params
            Plugin.runDeepict(self, PYTHON, '$(which segment.py) --config_file %s --pythonpath %s --tomo_name %s'
                        % (self.inputTomogram.get().getFileName(),
                           self.DEEPICT_TEMPORAL_PATH, self._getExtraPath(self.FILTERED_TOMO_FN)))
    '''
        def generateConfigFile():
            header = ['tomo_name','raw_tomo','filtered_tomo', 'no_mask']

    # Define the elements of this list:
        data = [tomo_name, '', tomogram_path, mask_path]

        with open(user_data_file, 'w', encoding='UTF8') as f:
            writer = csv.writer(f)

            # write the header
            writer.writerow(header)

            # write the data
            writer.writerow(data)
        
        data_dictionary = dict(zip(header, data))

        def read_yaml(file_path):
            with open(file_path, "r") as stream:
                data = yaml.safe_load(stream)
            return data

        def save_yaml(data, file_path):
            with open(file_path, 'w') as yaml_file:
                yaml.dump(data, yaml_file, default_flow_style=False)

            d = read_yaml(original_config_file)
            d['dataset_table'] = user_data_file
            d['output_dir'] = user_prediction_folder
            d['work_dir'] = user_work_folder
            d['model_path'] = f'{model_path}'
            d['tomos_sets']['training_list'] = []
            d['tomos_sets']['prediction_list'] = [f'{tomo_name}']
            d['cross_validation']['active'] = False
            d['training']['active'] = False
            d['prediction']['active'] = True
            d['evaluation']['particle_picking']['active'] = False
            d['evaluation']['segmentation_evaluation']['active'] = False
            d['training']['processing_tomo'] = 'filtered_tomo'
            d['prediction']['processing_tomo'] = 'filtered_tomo'
            d['postprocessing_clustering']['region_mask'] = 'no_mask'
            save_yaml(d, user_config_file)
    '''
    # --------------------------- INFO functions -----------------------------------
    def _summary(self):
        """ Summarize what the protocol has done"""
        summary = []

        if self.isFinished():
            summary.append("This protocol has printed *%s* %i times." % (self.message, self.times))
        return summary

    def _methods(self):
        methods = []

        if self.isFinished():
            methods.append("%s has been printed in this run %i times." % (self.message, self.times))
            if self.previousCount.hasPointer():
                methods.append("Accumulated count from previous runs were %i."
                               " In total, %s messages has been printed."
                               % (self.previousCount, self.count))
        return methods
