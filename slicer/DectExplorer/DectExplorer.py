import os
import unittest
import vtk, qt, ctk, slicer
import subprocess
import re
import numpy
from slicer.ScriptedLoadableModule import *
import logging

#
# slicer-dect
#

class DectExplorer(ScriptedLoadableModule):
  """Uses ScriptedLoadableModule base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self, parent):
    ScriptedLoadableModule.__init__(self, parent)
    self.parent.title = "DECTExplorer" # TODO make this more human readable by adding spaces
    self.parent.categories = ["DECT"]
    self.parent.dependencies = []
    self.parent.contributors = ["John Cronin (KCL)"] # replace with "Firstname Lastname (Organization)"
    self.parent.helpText = """
    Interfaces to the external DECT program for DECT 
    analysis.
    """
    self.parent.acknowledgementText = """
    By John Cronin
""" # replace with organization, grant and thanks.

#
# slicer-dectWidget
#

class DectExplorerWidget(ScriptedLoadableModuleWidget):
  """Uses ScriptedLoadableModuleWidget base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """
   
  def setup(self):
    ScriptedLoadableModuleWidget.setup(self)
    
    # Instantiate and connect widgets ...

    #
    # Parameters Area
    #
    parametersCollapsibleButton = ctk.ctkCollapsibleButton()
    parametersCollapsibleButton.text = "Parameters"
    self.layout.addWidget(parametersCollapsibleButton)

    # Layout within the dummy collapsible button
    parametersFormLayout = qt.QFormLayout(parametersCollapsibleButton)

    #
    # input volume selector
    #
    self.inputa = slicer.qMRMLNodeComboBox()
    self.inputa.nodeTypes = ["vtkMRMLScalarVolumeNode"]
    self.inputa.selectNodeUponCreation = True
    self.inputa.addEnabled = False
    self.inputa.removeEnabled = False
    self.inputa.noneEnabled = False
    self.inputa.showHidden = False
    self.inputa.showChildNodeTypes = False
    self.inputa.setMRMLScene( slicer.mrmlScene )
    self.inputa.setToolTip( "Pick the input to the algorithm." )
    parametersFormLayout.addRow("Input Volume A: ", self.inputa)
       
    self.inputb = slicer.qMRMLNodeComboBox()
    self.inputb.nodeTypes = ["vtkMRMLScalarVolumeNode"]
    self.inputb.selectNodeUponCreation = True
    self.inputb.addEnabled = False
    self.inputb.removeEnabled = False
    self.inputb.noneEnabled = False
    self.inputb.showHidden = False
    self.inputb.showChildNodeTypes = False
    self.inputb.setMRMLScene( slicer.mrmlScene )
    self.inputb.setToolTip( "Pick the input to the algorithm." )
    parametersFormLayout.addRow("Input Volume B: ", self.inputb)
    
    # output volume template
    #
    self.ovol = slicer.qMRMLNodeComboBox()
    self.ovol.nodeTypes = ["vtkMRMLScalarVolumeNode"]
    self.ovol.selectNodeUponCreation = True
    self.ovol.addEnabled = True
    self.ovol.removeEnabled = False
    self.ovol.noneEnabled = False
    self.ovol.renameEnabled = True
    self.ovol.showHidden = False
    self.ovol.showChildNodeTypes = False
    self.ovol.setMRMLScene( slicer.mrmlScene )
    self.ovol.setToolTip( "Pick the output to the algorithm." )
    parametersFormLayout.addRow("Output Volume: ", self.ovol)


    # Add vertical spacer
    parametersFormLayout.addRow(qt.QLabel(" "), qt.QLabel(" "))

    #
    # Apply Button
    #
    self.applyButton = qt.QPushButton("Apply")
    self.applyButton.toolTip = "Run the algorithm."
    self.applyButton.enabled = False
    parametersFormLayout.addRow(self.applyButton)
    
    #
    # Progress Bar
    #
    self.progbar = qt.QProgressBar();
    self.progbar.setValue(0);
    parametersFormLayout.addRow(self.progbar);

    # connections
    self.applyButton.connect('clicked(bool)', self.onApplyButton)
    self.inputa.connect("currentNodeChanged(vtkMRMLNode*)", self.onSelect)
    self.inputb.connect("currentNodeChanged(vtkMRMLNode*)", self.onSelect)
   
    # Add vertical spacer
    self.layout.addStretch(1)
    
    self.onSelect()

  def cleanup(self):
    pass
    
  def onSelect(self):
    self.applyButton.enabled = self.inputa.currentNode() and self.inputb.currentNode() and self.ovol.currentNode()

  def onApplyButton(self):
    logic = DectExplorerLogic()
    logic.run(self.inputa.currentNode(), self.inputb.currentNode(), self.ovol.currentNode(), 200, self.progbar)

#
# slicer-dectLogic
#

class DectExplorerLogic(ScriptedLoadableModuleLogic):
  """This class should implement all the actual
  computation done by your module.  The interface
  should be such that other python code can import
  this class and make use of the functionality without
  requiring an instance of the Widget.
  Uses ScriptedLoadableModuleLogic base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def run(self, inputa, inputb, ovol, bins = 200, pb = None):
    """
    Run the actual algorithm
    """

    logging.info('Processing started')
    if pb is None:
      pass
    else:
      pb.setValue(0)
      slicer.app.processEvents()
    
    # get the input arrays
    import vtk.util.numpy_support
    inputa_im = inputa.GetImageData()
    inputa_shape = list(inputa_im.GetDimensions())
    inputa_shape.reverse()
    a = vtk.util.numpy_support.vtk_to_numpy(inputa_im.GetPointData().GetScalars()).reshape(inputa_shape)

    inputb_im = inputb.GetImageData()
    inputb_shape = list(inputb_im.GetDimensions())
    inputb_shape.reverse()
    b = vtk.util.numpy_support.vtk_to_numpy(inputb_im.GetPointData().GetScalars()).reshape(inputb_shape)
    
    # flatten then coalesce into a 2xn array
    import numpy as np
    af = a.flatten()
    bf = b.flatten()
    c = np.concatenate((af.reshape(af.shape[0], 1), bf.reshape(bf.shape[0], 1)), axis=1)

    # get histogram data
    r = np.histogramdd(c, bins=bins)
    vals = np.log1p(r[0])
    
    vals = np.transpose(vals)
    
    # output
    ospacing = [r[1][0][1] - r[1][0][0], r[1][1][1] - r[1][1][0], 10]
    osize = [bins,bins,1]
    ovtype = vtk.VTK_FLOAT
    
    imageData = ovol.GetImageData()
    if not imageData:
      import vtk
      imageData = vtk.vtkImageData()
      ovol.SetAndObserveImageData(imageData)
    imageData.SetDimensions(bins, bins, 1)
    imageData.AllocateScalars(ovtype, 1)
    
    o = vtk.util.numpy_support.vtk_to_numpy(imageData.GetPointData().GetScalars()).reshape([1,200,200])
    o[:] = vals
    
    ovol.SetSpacing(ospacing)
    ovol.SetOrigin([r[1][0][0], r[1][1][0], 0])
    
    dn = ovol.GetDisplayNode()
    if not dn:
      import vtk
      dn = slicer.vtkMRMLScalarVolumeDisplayNode()
      slicer.mrmlScene.AddNode(dn)
      ovol.SetAndObserveDisplayNodeID(dn.GetID())
    dn.SetAndObserveColorNodeID(slicer.util.getNode("Rainbow").GetID())
    
    ovol.StorableModified()
    ovol.Modified()
    #ovol.InvokeEvent(slicer.vtkMRMLVolumeNode.ImageDataModifiedEvent, ovol)
    
    #setSliceViewerLayers(background=ovol)

  
    logging.info('Processing completed')
    if pb is None:
      pass
    else:
      pb.setValue(100)
      slicer.app.processEvents()


    return True


class DectExplorerTest(ScriptedLoadableModuleTest):
  """
  This is the test case for your scripted module.
  Uses ScriptedLoadableModuleTest base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def setUp(self):
    """ Do whatever is needed to reset the state - typically a scene clear will be enough.
    """
    slicer.mrmlScene.Clear(0)

  def runTest(self):
    """Run as few or as many tests as needed here.
    """
    self.setUp()
    self.test_DectExplorer1()

  def test_DectExplorer1(self):
    """ Ideally you should have several levels of tests.  At the lowest level
    tests should exercise the functionality of the logic with different inputs
    (both valid and invalid).  At higher levels your tests should emulate the
    way the user would interact with your code and confirm that it still works
    the way you intended.
    One of the most important features of the tests is that it should alert other
    developers when their changes will have an impact on the behavior of your
    module.  For example, if a developer removes a feature that you depend on,
    your test should break so they know that the feature is needed.
    """

    self.delayDisplay("Starting the test")
    #
    # first, get some data
    #
    import urllib
    downloads = (
        ('http://slicer.kitware.com/midas3/download?items=5767', 'FA.nrrd', slicer.util.loadVolume),
        )

    for url,name,loader in downloads:
      filePath = slicer.app.temporaryPath + '/' + name
      if not os.path.exists(filePath) or os.stat(filePath).st_size == 0:
        logging.info('Requesting download %s from %s...\n' % (name, url))
        urllib.urlretrieve(url, filePath)
      if loader:
        logging.info('Loading %s...' % (name,))
        loader(filePath)
    self.delayDisplay('Finished with download and loading')

    volumeNode = slicer.util.getNode(pattern="FA")
    logic = DectExplorerLogic()
    self.assertTrue( logic.hasImageData(volumeNode) )
    self.delayDisplay('Test passed!')
