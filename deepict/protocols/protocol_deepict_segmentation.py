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
from pyworkflow.object import Set
from scipion.constants import PYTHON
from tomo.objects import Tomogram, SetOfTomograms
from tomo.protocols import ProtTomoBase
from pwem.protocols import EMProtocol
import csv
import os
from deepict import Plugin

import yaml

class DeepictSegmentation(EMProtocol, ProtTomoBase):
    """
    Cryo-electron tomograms capture a wealth of structural information on the molecular constituents
    of cells and tissues. DeePiCt (Deep Picker in Context) is a deep-learning
    framework for supervised structure segmentation and macromolecular complex localization in
    cellular cryo-electron tomography (ribosomes, fatty acid
    synthases, membranes, nuclear pore complexes, organelles and cytosol).
    """
    _label = 'Segmentation'

    tomo_name = None
    tomogram_path = None
    mask_path = None
    Tomograms = None

    RIBOSOME    = 0
    MEMBRANE    = 1
    MICROTUBULE = 2
    FAS         = 3

    INTERSECTION    = 0
    CONTACT         = 1
    COLOCALIZATION  = 2

    OUTPUT_TOMOGRAMS_NAME = "Tomograms"

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
            self._insertFunctionStep(self.setupFolderStep, inTomogram, tomId)
            self._insertFunctionStep(self.spectrumStep, inTomogram, tomId)
            self._insertFunctionStep(self.createConfigFiles, inTomogram, tomId)
            self._insertFunctionStep(self.splitIntoPatchesStep, inTomogram, tomId)
            self._insertFunctionStep(self.segmentStep, inTomogram, tomId)
            self._insertFunctionStep(self.assemblePredictionStep, inTomogram, tomId)
            self._insertFunctionStep(self.postProcessingStep, inTomogram, tomId)
            self._insertFunctionStep(self.createOutputStep, tomId)
        self._insertFunctionStep(self.closeOutputSetsStep)

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

    def read_yaml(self, file_path):
        with open(file_path, "r") as stream:
            data = yaml.safe_load(stream)
        return data


    def postProcessingStep(self, inputTom, tomId):

        user_config_file = os.path.join(self.getTsIdFolder(inputTom, tomId), 'config.yaml')
        d = self.read_yaml(user_config_file)

        max_cluster_size = None
        if self.maxClusterSize.get() != 0:
            max_cluster_size = self.maxClusterSize.get()

        ctMOpt = self.contactMode.get()

        if ctMOpt == self.INTERSECTION:
            contact_mode = 'intersection'
        elif ctMOpt == self.CONTACT:
            contact_mode = 'contact'
        elif ctMOpt == self.COLOCALIZATION:
            contact_mode = 'colocalization'

        d['postprocessing_clustering']['active'] = True
        d['postprocessing_clustering']['threshold'] = self.threshold.get()
        d['postprocessing_clustering']['min_cluster_size'] = self.minClusterSize.get()
        d['postprocessing_clustering']['max_cluster_size'] = max_cluster_size
        d['postprocessing_clustering']['clustering_connectivity'] = self.clusteringConnectivity.get()
        d['postprocessing_clustering']['calculate_motl'] = self.calculateMotl.get()
        d['postprocessing_clustering']['ignore_border_thickness'] = 0
        d['postprocessing_clustering']['region_mask'] = 'no_mask'
        d['postprocessing_clustering']['contact_mode'] = contact_mode
        d['postprocessing_clustering']['contact_distance'] = self.contactDistance.get()

        self.save_yaml(d, user_config_file)

        tsid = inputTom[tomId].getTsId()

        Plugin.runDeepict(self, PYTHON, 'DeePiCt/3d_cnn/scripts/clustering_and_cleaning.py --config_file %s --pythonpath %s --tomo_name %s'
                          % (user_config_file, os.path.join(Plugin.getHome(), 'DeePiCt/3d_cnn/src'), tsid))


    def getTsIdFolder(self, inputTom, tomId):
        ts = inputTom[tomId]
        tsId = ts.getTsId()

        #Defining the output folder
        tomoPath = self._getExtraPath(tsId)
        return tomoPath


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

    def save_yaml(self, data, file_path):
        with open(file_path, 'w') as yaml_file:
            yaml.dump(data, yaml_file, default_flow_style=False)


    def createConfigFiles(self, inputTom, tomId):

        model_path = os.path.join(Plugin.getHome(), self.getModel())

        tomo_name = inputTom[tomId].getTsId()
        tomogram_path = os.path.join(self.getTsIdFolder(inputTom, tomId), self.FILTERED_TOMO_FN)

        mask_path = ''
        #if self.inputMask:
        #    mask_path = self.inputMask[tomId].getFileName()

        user_config_file = os.path.join(self.getTsIdFolder(inputTom, tomId), 'config.yaml')
        user_data_file = os.path.join(self.getTsIdFolder(inputTom, tomId), 'data.csv')
        user_prediction_folder = self.getTsIdFolder(inputTom, tomId)

        user_work_folder = self.getTsIdFolder(inputTom, tomId)

        os.makedirs(os.path.split(user_config_file)[0], exist_ok=True)
        os.makedirs(os.path.split(user_data_file)[0], exist_ok=True)
        os.makedirs(os.path.split(user_prediction_folder)[0], exist_ok=True)
        os.makedirs(os.path.split(user_work_folder)[0], exist_ok=True)

        header = ['tomo_name', 'raw_tomo', 'filtered_tomo', 'no_mask']

        # Define the elements of this list:
        data = [tomo_name, '', tomogram_path, mask_path]

        with open(user_data_file, 'w', encoding='UTF8') as f:
            writer = csv.writer(f)

            # write the header
            writer.writerow(header)

            # write the data
            writer.writerow(data)

        original_config_file = os.path.join(Plugin.getHome(), 'DeePiCt/3d_cnn/config.yaml')
        d = self.read_yaml(original_config_file)
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
        self.save_yaml(d, user_config_file)

    def getOutputSetOfTomograms(self, inputSet):

        if self.Tomograms:
            getattr(self, self.OUTPUT_TOMOGRAMS_NAME).enableAppend()
        else:
            outputSetOfTomograms = self._createSetOfTomograms()

            if isinstance(inputSet, SetOfTomograms):
                outputSetOfTomograms.copyInfo(inputSet)

            outputSetOfTomograms.setStreamState(Set.STREAM_OPEN)

            self._defineOutputs(**{self.OUTPUT_TOMOGRAMS_NAME: outputSetOfTomograms})
            self._defineSourceRelation(inputSet, outputSetOfTomograms)

        return self.Tomograms

    def createOutputStep(self, tsObjId):
        ts = self.inputTomogram.get()[tsObjId]
        tsId = ts.getTsId()

        typeOfModel = os.path.split(os.path.splitext(self.getModel())[0])[1]
        print(typeOfModel)
        predfolder = os.path.join('predictions', typeOfModel, tsId, 'memb')
        outputSeg = os.path.join(self._getExtraPath(tsId), predfolder)

        output = self.getOutputSetOfTomograms(self.inputTomogram.get())

        newTomogram = Tomogram()
        newTomogram.setLocation(os.path.join(outputSeg,
                                             'post_processed_prediction.mrc'))
        newTomogram.setTsId(tsId)
        newTomogram.setSamplingRate(ts.getSamplingRate())
        print(os.path.join(outputSeg,'post_processed_prediction.mrc'))
        # Set default tomogram origin
        newTomogram.setOrigin(newOrigin=None)
        newTomogram.setAcquisition(ts.getAcquisition())

        output.append(newTomogram)
        output.update(newTomogram)
        output.write()
        self._store()

    def closeOutputSetsStep(self):
        self.Tomograms.setStreamState(Set.STREAM_CLOSED)
        self.Tomograms.write()
        self._store()


    # --------------------------- INFO functions -----------------------------------
    def _summary(self):
        """ Summarize what the protocol has done"""
        summary = []

        if self.isFinished():
            summary.append("A set of %s tomograms have been segmented with deepict using a %s model" % (self.inputTomogram.get().getSize(), self.times))
        return summary

    def _methods(self):
        methods = []

        if self.Tomograms:
            methods.append("The segmentations has been computed for %d "
                           "tomograms using the deepict segmenter.\n"
                           % (self.inputTomogram.get().getSize()))
        return methods
