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

from pyworkflow.protocol import Protocol, params, Integer
from pyworkflow.utils import Message
from pyworkflow.protocol import EnumParam, IntParam, FloatParam, BooleanParam, LT, GT
from scipion.constants import PYTHON
import csv
import os
from deepict import Plugin


class DeepictSegmentation(Protocol):
    """
    TODO resumir
    Cryo-electron tomograms capture a wealth of structural information on the molecular constituents
    of cells and tissues. We present DeePiCt (Deep Picker in Context) is a deep-learning
    framework for supervised structure segmentation and macromolecular complex localization in
    cellular cryo-electron tomography. To train and benchmark DeePiCt on experimental data, we
    comprehensively annotated 20 tomograms of Schizosaccharomyces pombe for ribosomes, fatty acid
    synthases, membranes, nuclear pore complexes, organelles and cytosol.
    """
    _label = 'Segmentation'

    tomo_name = None
    tomogram_path = None
    mask_path = None


    RIBOSOME    = 0
    MEMBRANE    = 1
    MICROTUBULE = 2
    FAS         = 3

    INTERSECTION    = 0
    CONTACT         = 1
    COLOCALIZATION  = 2

    AMP_SPECTRUM_FN     = 'amp_spectrum.tsv'
    FILTERED_TOMO_FN    = 'match_spectrum_filt.mrc'

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
                      label='Input tomograms',
                      important=True,
                      allowsNull=False,
                      help='Set of reconstructed tomograms to be segmented by DeePiCt')
        
        form.addParam('inputMask', params.PointerParam,
                      pointerClass='SetOfTomograms',
                      label='Mask',
                      allowsNull=True,
                      help='Set of tomo masks that helps the DeePiCt image processing.')

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
                      default=False,
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


        form.addHidden(params.GPU_LIST,
                       params.StringParam,
                       default='0',
                       expertLevel=params.LEVEL_ADVANCED,
                       label="Choose GPU IDs",
                       help="GPU ID. To pick the best available one set 0. For a specific GPU set its number ID.")


    # --------------------------- STEPS functions ------------------------------
    def _insertAllSteps(self):
        # Insert processing steps
        inTomogram = self.inputTomogram.get()
        inMask = self.inputMask.get()

        for tom in inTomogram:
            tomId = tom.getObjId()
            self._insertFunctionStep('setupFolderStep', inTomogram, tomId)
            self._insertFunctionStep('spectrumStep', inTomogram, tomId)
            self._insertFunctionStep('createConfigFiles', inTomogram, tomId)
            self._insertFunctionStep('splitIntoPatchesStep', inTomogram, tomId)
            self._insertFunctionStep('segmentStep', inTomogram, tomId)
            self._insertFunctionStep('assemblePredictionStep', inTomogram, tomId)
            self._insertFunctionStep('postProcessingStep', inTomogram, tomId)

    def setupFolderStep(self, inputTom, tomId):
        # Obtaining the ts and the tsId
        ts = inputTom[tomId]
        tsId = ts.getTsId()

        # Creating the tomogram folder
        tomoPath = self._getExtraPath(tsId)
        os.mkdir(tomoPath)


    def spectrumStep(self, inputTom, tomId):
        input_tomo = inputTom[tomId].getFileName()
        target_spectrum = os.path.join(self.getTsIdFolder(inputTom, tomId), self.AMP_SPECTRUM_FN)
        filtered_tomo = os.path.join(self.getTsIdFolder(inputTom, tomId), self.FILTERED_TOMO_FN)

        Plugin.runDeepict(self, PYTHON, 'DeePiCt/spectrum_filter/extract_spectrum.py --input %s --output %s'
                        % (input_tomo, target_spectrum))

        Plugin.runDeepict(self, PYTHON, 'DeePiCt/spectrum_filter/match_spectrum.py --input %s --target %s --output %s'
                          % (input_tomo, target_spectrum, filtered_tomo))


    #TODO create new steps (notebook section 3)
    def splitIntoPatchesStep(self, inputTom, tomId):
        # Create the 64^3 patches
        fnConfig = os.path.join(self.getTsIdFolder(inputTom, tomId), 'config.yaml')
        pathPython = os.path.join(Plugin.getHome(), 'DeePiCt/3d_cnn/src')
        tomo_name = inputTom[tomId].getTsId()

        Plugin.runDeepict(self, PYTHON, 'DeePiCt/3d_cnn/scripts/generate_prediction_partition.py --config_file %s --pythonpath %s --tomo_name %s'
                        % (fnConfig, pathPython, tomo_name))


    def segmentStep(self, inputTom, tomId):
        tsid = inputTom[tomId].getTsId()

        Plugin.runDeepict(self, PYTHON, 'DeePiCt/3d_cnn/scripts/segment.py --config_file %s --pythonpath %s --tomo_name %s --gpu %i'
                        % (os.path.join(self.getTsIdFolder(inputTom, tomId), 'config.yaml'),
                           os.path.join(Plugin.getHome(), 'DeePiCt/3d_cnn/src'),
                           tsid, self.getGpuList()[0]))

    def assemblePredictionStep(self, inputTom, tomId):
        tsid = inputTom[tomId].getTsId()
        # Assemnble the segmentated patches
        Plugin.runDeepict(self, PYTHON, 'DeePiCt/3d_cnn/scripts/assemble_prediction.py --config_file %s --pythonpath %s --tomo_name %s'
                          % (os.path.join(self.getTsIdFolder(inputTom, tomId), 'config.yaml'),
                             os.path.join(Plugin.getHome(), 'DeePiCt/3d_cnn/src'), tsid))

    def postProcessingStep(self, inputTom, tomId):
        import yaml
        def read_yaml(file_path):
            with open(file_path, "r") as stream:
                data = yaml.safe_load(stream)
            return data

        def save_yaml(data, file_path):
            with open(file_path, 'w') as yaml_file:
                yaml.dump(data, yaml_file, default_flow_style=False)

        user_config_file = os.path.join(self.getTsIdFolder(inputTom, tomId), 'config.yaml')
        d = read_yaml(user_config_file)

        # @markdown #### If you don't want to use the default parameters, unclick the button for `default_options` and define the parameters. Otherwise, the default options will be used.

        default_options = True  # @param {type:"boolean"}

        if default_options:
            d['postprocessing_clustering']['active'] = True
            d['postprocessing_clustering']['threshold'] = 0.5
            d['postprocessing_clustering']['min_cluster_size'] = 500
            d['postprocessing_clustering']['max_cluster_size'] = None
            d['postprocessing_clustering']['clustering_connectivity'] = 1
            d['postprocessing_clustering']['calculate_motl'] = True
            d['postprocessing_clustering']['ignore_border_thickness'] = 0
            d['postprocessing_clustering']['region_mask'] = 'no_mask'
            d['postprocessing_clustering']['contact_mode'] = 'intersection'
            d['postprocessing_clustering']['contact_distance'] = 0
        else:
            threshold = 0.5  # @param {type:"number"}
            min_cluster_size = 500  # @param {type:"integer"}
            max_cluster_size = 0  # @param {type:"integer"}
            clustering_connectivity = 1  # @param {type:"integer"}
            calculate_motl = True  # @param {type:"boolean"}
            contact_mode = 'intersection'  # @param ["contact", "colocalization", "intersection"]
            contact_distance = 0  # @param {type:"integer"}
            if max_cluster_size == 0:
                max_cluster_size = None
            d['postprocessing_clustering']['active'] = True
            d['postprocessing_clustering']['threshold'] = threshold
            d['postprocessing_clustering']['min_cluster_size'] = min_cluster_size
            d['postprocessing_clustering']['max_cluster_size'] = max_cluster_size
            d['postprocessing_clustering']['clustering_connectivity'] = clustering_connectivity
            d['postprocessing_clustering']['calculate_motl'] = calculate_motl
            d['postprocessing_clustering']['ignore_border_thickness'] = 0
            d['postprocessing_clustering']['region_mask'] = 'no_mask'
            d['postprocessing_clustering']['contact_mode'] = 'intersection'
            d['postprocessing_clustering']['contact_distance'] = contact_distance

        save_yaml(d, user_config_file)

        tsid = inputTom[tomId].getTsId()

        Plugin.runDeepict(self, PYTHON, 'DeePiCt/3d_cnn/scripts/clustering_and_cleaning.py --config_file %s --pythonpath %s --tomo_name %s'
                          % (user_config_file, os.path.join(Plugin.getHome(), 'DeePiCt/3d_cnn/src'), tsid))


    def getTsIdFolder(self, inputTom, tomId):
        ts = inputTom[tomId]
        tsId = ts.getTsId()

        #Defining the output folder
        tomoPath = self._getExtraPath(tsId)
        return tomoPath

    def defineImportanVariables(self):
        pass

    def getModel(self):
        tomoOpt = self.tomogramOption.get()

        if tomoOpt == self.MEMBRANE:
            modelWeights = os.path.join('models', 'membraneModel.pth')
        elif tomoOpt == self.MICROTUBULE:
            modelWeights = os.path.join('models', 'microtubuleModel.pth')
        elif tomoOpt == self.RIBOSOME:
            modelWeights = os.path.join('models', 'ribosomeModel.pth')
        elif tomoOpt == self.FAS:
            modelWeights = os.path.join('models', 'fasModel.pth')

        return modelWeights


    def createConfigFiles(self, inputTom, tomId):
        def read_yaml(file_path):
            with open(file_path, "r") as stream:
                data = yaml.safe_load(stream)
            return data

        def save_yaml(data, file_path):
            with open(file_path, 'w') as yaml_file:
                yaml.dump(data, yaml_file, default_flow_style=False)

        model_path = os.path.join(Plugin.getHome(), self.getModel())

        tomo_name = inputTom[tomId].getTsId()#inputTom[tomId].getFileName()
        tomogram_path = os.path.join(self.getTsIdFolder(inputTom, tomId), self.FILTERED_TOMO_FN)

        mask_path = ''
        #if self.inputMask:
        #    mask_path = self.inputMask[tomId].getFileName()

        user_config_file = os.path.join(self.getTsIdFolder(inputTom, tomId), 'config.yaml')
        user_data_file = os.path.join(self.getTsIdFolder(inputTom, tomId), 'data.csv')
        user_prediction_folder = self.getTsIdFolder(inputTom, tomId)

        #TODO: user_work_folder should be tmp
        user_work_folder = self.getTsIdFolder(inputTom, tomId)

        os.makedirs(os.path.split(user_config_file)[0], exist_ok=True)
        os.makedirs(os.path.split(user_data_file)[0], exist_ok=True)
        os.makedirs(os.path.split(user_prediction_folder)[0], exist_ok=True)
        os.makedirs(os.path.split(user_work_folder)[0], exist_ok=True)

        import yaml

        header = ['tomo_name', 'raw_tomo', 'filtered_tomo', 'no_mask']

        # Define the elements of this list:
        data = [tomo_name, '', tomogram_path, mask_path]

        with open(user_data_file, 'w', encoding='UTF8') as f:
            writer = csv.writer(f)

            # write the header
            writer.writerow(header)

            # write the data
            writer.writerow(data)

        data_dictionary = dict(zip(header, data))

        original_config_file = os.path.join(Plugin.getHome(), 'DeePiCt/3d_cnn/config.yaml')
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
