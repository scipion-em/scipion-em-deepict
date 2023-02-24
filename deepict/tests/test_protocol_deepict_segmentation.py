# **************************************************************************
# *
# * Authors:    Jose Luis Vilas Prieto (jlvilas@cnb.csic.es)
# *
# * Unidad de  Bioinformatica of Centro Nacional de Biotecnologia , CSIC
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
from os.path import exists
from pyworkflow.tests import BaseTest, DataSet, setupTestProject
from tomo.protocols import ProtImportTomograms
from deepict.protocols import DeepictSegmentation



class TestDeepictBase(BaseTest):
    @classmethod
    def setData(cls, dataProject='monotomo'):
        cls.dataset = DataSet.getDataSet(dataProject)
        cls.even = cls.dataset.getFile('even_tomogram_rx*.mrc')
        cls.odd = cls.dataset.getFile('odd_tomogram_rx*.mrc')

    @classmethod
    def runImportTomograms(cls, pattern, samplingRate):
        """ Run an Import volumes protocol. """
        cls.protImport = cls.newProtocol(ProtImportTomograms,
                                         filesPath=pattern,
                                         samplingRate=samplingRate
                                         )
        cls.launchProtocol(cls.protImport)
        return cls.protImport


class TestDeepict(TestDeepictBase):
    @classmethod
    def setUpClass(cls):
        setupTestProject(cls)
        TestDeepictBase.setData()
        cls.protImportHalf1 = cls.runImportTomograms(cls.odd, 16.14)
        cls.protImportHalf2 = cls.runImportTomograms(cls.even, 16.14)


    def deepictSetProtocol(self, option):
        return self.newProtocol(DeepictSegmentation,
                                    objLabel='deepict segmentation ' + option,
                                    inputTomogram=self.protImportHalf1.outputTomograms,
                                    inputMask=self.protImportHalf2.outputTomograms,
                                    tomogramOption=option,
                                    )
                                    
    def testDeepictMembrane(self):
        Deepict = self.deepictSetProtocol(self, 'membrane')
        self.launchProtocol(Deepict)
        self.assertTrue(exists(Deepict._getExtraPath('tomo_1/filtered_tomo.mrc')),
                        "Deepict has failed")

    def testDeepictRibosome(self):
        Deepict = self.deepictSetProtocol('ribosome')
        self.launchProtocol(Deepict)
        self.assertTrue(exists(Deepict._getExtraPath('tomo_1/filtered_tomo.mrc')),
                        "Deepict has failed")

    def testDeepictMicrotubule(self):
        Deepict = self.deepictSetProtocol('microtubule')
        self.launchProtocol(Deepict)
        self.assertTrue(exists(Deepict._getExtraPath('tomo_1/filtered_tomo.mrc')),
                        "Deepict has failed")
    
    def testDeepictFAS(self):
        Deepict = self.deepictSetProtocol('FAS')
        self.launchProtocol(Deepict)
        self.assertTrue(exists(Deepict._getExtraPath('tomo_1/filtered_tomo.mrc')),
                        "Deepict has failed")
    
        

