import os
import unittest
import vtk, qt, ctk, slicer
import subprocess
import re
from slicer.ScriptedLoadableModule import *
import logging

#
# slicer-dect
#

class slicerdect(ScriptedLoadableModule):
  """Uses ScriptedLoadableModule base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self, parent):
    ScriptedLoadableModule.__init__(self, parent)
    self.parent.title = "DECT" # TODO make this more human readable by adding spaces
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

class slicerdectWidget(ScriptedLoadableModuleWidget):
  """Uses ScriptedLoadableModuleWidget base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """
   
  def dectChanged(self):
    # respond to changes in the slicer executable name
    p = self.dectapp.text
    
    self.device.clear()
    
    try:    
      startupinfo = None
      if os.name == 'nt':
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
      p2 = subprocess.Popen([p, '-h'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, startupinfo=startupinfo)
      p2.wait()
      p3 = p2.communicate()
      p4 = p3[0].decode("utf-8")
      
      alphaare = 'a in file A \(defaults to (\-*[0-9]+)'
      betaare = 'b in file A \(defaults to (\-*[0-9]+)'
      gammaare = 'c in file A \(defaults to (\-*[0-9]+)'
      alphabre = 'a in file B \(defaults to (\-*[0-9]+)'
      betabre = 'b in file B \(defaults to (\-*[0-9]+)'
      gammabre = 'c in file B \(defaults to (\-*[0-9]+)'
      mratiore = 'merged image \(defaults to (\-*[0-9]+\.[0-9]+)'
      
      # if there are pre-stored persistent settings, use them
      us = slicer.app.userSettings()
      
      aA = us.value("DECT/aA")
      bA = us.value("DECT/bA")
      cA = us.value("DECT/cA")
      aB = us.value("DECT/aB")
      bB = us.value("DECT/bB")
      cB = us.value("DECT/cB")
      m = us.value("DECT/m")
      D = us.value("DECT/D")
      
      if(aA is not None):
        self.alphaa.text = aA
      else:
        self.alphaa.text = re.search(alphaare, p4).group(1)
        
      if(bA is not None):
        self.betaa.text = bA
      else:
        self.betaa.text = re.search(betaare, p4).group(1)
        
      if(cA is not None):
        self.gammaa.text = cA
      else:
        self.gammaa.text = re.search(gammaare, p4).group(1)
        
      if(aB is not None):
        self.alphab.text = aB
      else:
        self.alphab.text = re.search(alphabre, p4).group(1)
        
      if(bB is not None):
        self.betab.text = bB
      else:
        self.betab.text = re.search(betabre, p4).group(1)
        
      if(cB is not None):
        self.gammab.text = cB
      else:
        self.gammab.text = re.search(gammabre, p4).group(1)
        
      if(m is not None):
        self.mratio.text = m
      else:
        self.mratio.text = re.search(mratiore, p4).group(1)
     
      v = re.search('Devices(\r\n [0-9]+: [^\r]+)+', p4)
      v2 = v.group(0).split('\r\n')
      
      for i in range(1, len(v2)):
        v3 = v2[i]
        self.device.addItem(re.search(' [0-9]+: ([\w\W]+)', v3).group(1))
        
      if(D is not None):
        didx = self.device.findText(D)
        if(didx >= 0):
          self.device.setCurrentIndex(didx)
        
      self.dectExeGood = True
      
      us.setValue("DECT/dect", p)
    except Exception as excpt:
      print (excpt)
      self.dectExeGood = False

  def setup(self):
    ScriptedLoadableModuleWidget.setup(self)
    
    self.dectExeGood = False

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
    
    self.alphaa = qt.QLineEdit()
    self.alphaa.text = ''
    parametersFormLayout.addRow("Density of a in A: ", self.alphaa)
    
    self.betaa = qt.QLineEdit()
    self.betaa.text = ''
    parametersFormLayout.addRow("Density of b in A: ", self.betaa)
    
    self.gammaa = qt.QLineEdit()
    self.gammaa.text = ''
    parametersFormLayout.addRow("Density of c in A: ", self.gammaa)
    
    # Add vertical spacer
    parametersFormLayout.addRow(qt.QLabel(" "), qt.QLabel(" "))
    
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

    self.alphab = qt.QLineEdit()
    self.alphab.text = ''
    parametersFormLayout.addRow("Density of a in B: ", self.alphab)
    
    self.betab = qt.QLineEdit()
    self.betab.text = ''
    parametersFormLayout.addRow("Density of b in B: ", self.betab)
    
    self.gammab = qt.QLineEdit()
    self.gammab.text = ''
    parametersFormLayout.addRow("Density of c in B: ", self.gammab)
    
    # Add vertical spacer
    parametersFormLayout.addRow(qt.QLabel(" "), qt.QLabel(" "))

    
    #
    # output volume selector
    #
    self.outputa = slicer.qMRMLNodeComboBox()
    self.outputa.nodeTypes = ["vtkMRMLScalarVolumeNode"]
    self.outputa.selectNodeUponCreation = True
    self.outputa.addEnabled = True
    self.outputa.removeEnabled = True
    self.outputa.noneEnabled = True
    self.outputa.renameEnabled = True
    self.outputa.showHidden = False
    self.outputa.showChildNodeTypes = False
    self.outputa.setMRMLScene( slicer.mrmlScene )
    self.outputa.setToolTip( "Pick the output to the algorithm." )
    parametersFormLayout.addRow("Output Volume a: ", self.outputa)
    
    self.outputb = slicer.qMRMLNodeComboBox()
    self.outputb.nodeTypes = ["vtkMRMLScalarVolumeNode"]
    self.outputb.selectNodeUponCreation = True
    self.outputb.addEnabled = True
    self.outputb.removeEnabled = True
    self.outputb.noneEnabled = True
    self.outputb.renameEnabled = True
    self.outputb.showHidden = False
    self.outputb.showChildNodeTypes = False
    self.outputb.setMRMLScene( slicer.mrmlScene )
    self.outputb.setToolTip( "Pick the output to the algorithm." )
    parametersFormLayout.addRow("Output Volume b: ", self.outputb)
    
    self.outputc = slicer.qMRMLNodeComboBox()
    self.outputc.nodeTypes = ["vtkMRMLScalarVolumeNode"]
    self.outputc.selectNodeUponCreation = True
    self.outputc.addEnabled = True
    self.outputc.removeEnabled = True
    self.outputc.noneEnabled = True
    self.outputc.renameEnabled = True
    self.outputc.showHidden = False
    self.outputc.showChildNodeTypes = False
    self.outputc.setMRMLScene( slicer.mrmlScene )
    self.outputc.setToolTip( "Pick the output to the algorithm." )
    parametersFormLayout.addRow("Output Volume c: ", self.outputc)

    # Add vertical spacer
    parametersFormLayout.addRow(qt.QLabel(" "), qt.QLabel(" "))

    self.outputm = slicer.qMRMLNodeComboBox()
    self.outputm.nodeTypes = ["vtkMRMLScalarVolumeNode"]
    self.outputm.selectNodeUponCreation = True
    self.outputm.addEnabled = True
    self.outputm.removeEnabled = True
    self.outputm.noneEnabled = True
    self.outputm.renameEnabled = True
    self.outputm.showHidden = False
    self.outputm.showChildNodeTypes = False
    self.outputm.setMRMLScene( slicer.mrmlScene )
    self.outputm.setToolTip( "Pick the output to the algorithm." )
    parametersFormLayout.addRow("Output Merge Volume: ", self.outputm)
    
    self.mratio = qt.QLineEdit()
    self.mratio.text = ''
    parametersFormLayout.addRow("Ratio (A:B) for merge: ", self.mratio)
    
    us = slicer.app.userSettings()
    
    # Add vertical spacer
    parametersFormLayout.addRow(qt.QLabel(" "), qt.QLabel(" "))

    self.enhanced = qt.QCheckBox()
    E = us.value("DECT/E")
    if(E is not None and E == "false"):
      self.enhanced.setChecked(False)
    else:
      self.enhanced.setChecked(True)
    parametersFormLayout.addRow("Enhanced quality (slower): ", self.enhanced)
    
    self.flip = qt.QCheckBox()
    F = us.value("DECT/F")
    if(F is not None and F == "true"):
      self.flip.setChecked(True)
    else:
      self.flip.setChecked(False)
    parametersFormLayout.addRow("Flip output 180 degrees: ", self.flip)

    # Add vertical spacer
    parametersFormLayout.addRow(qt.QLabel(" "), qt.QLabel(" "))
    
    self.dectapp = qt.QLineEdit()
    dectapp = us.value("DECT/dect")
    if(dectapp is not None):
      self.dectapp.text = dectapp
    else:
      self.dectapp.text = os.path.join(os.path.dirname(__file__), "dect.exe")
    parametersFormLayout.addRow("DECT executable:", self.dectapp)

    self.device = qt.QComboBox()
    parametersFormLayout.addRow("Device", self.device)
    
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
    self.onDectChanged()
    self.applyButton.connect('clicked(bool)', self.onApplyButton)
    self.inputa.connect("currentNodeChanged(vtkMRMLNode*)", self.onSelect)
    self.inputb.connect("currentNodeChanged(vtkMRMLNode*)", self.onSelect)
    self.outputa.connect("currentNodeChanged(vtkMRMLNode*)", self.onSelect)
    self.outputb.connect("currentNodeChanged(vtkMRMLNode*)", self.onSelect)
    self.outputc.connect("currentNodeChanged(vtkMRMLNode*)", self.onSelect)
    self.outputm.connect("currentNodeChanged(vtkMRMLNode*)", self.onSelect)
    self.alphaa.textChanged.connect(self.persistSettings)
    self.betaa.textChanged.connect(self.persistSettings)
    self.gammaa.textChanged.connect(self.persistSettings)
    self.alphab.textChanged.connect(self.persistSettings)
    self.betab.textChanged.connect(self.persistSettings)
    self.gammab.textChanged.connect(self.persistSettings)
    self.mratio.textChanged.connect(self.persistSettings)
    self.device.currentIndexChanged.connect(self.persistSettings)
   
    self.dectapp.textChanged.connect(self.onDectChanged)
    
    # Add vertical spacer
    self.layout.addStretch(1)

  def persistSettings(self):
    us = slicer.app.userSettings()
    us.setValue("DECT/aA", int(self.alphaa.text))
    us.setValue("DECT/bA", int(self.betaa.text))
    us.setValue("DECT/cA", int(self.gammaa.text))
    us.setValue("DECT/aB", int(self.alphab.text))
    us.setValue("DECT/bB", int(self.betab.text))
    us.setValue("DECT/cB", int(self.gammab.text))
    us.setValue("DECT/m", float(self.mratio.text))
    us.setValue("DECT/E", self.enhanced.isChecked())
    us.setValue("DECT/F", self.flip.isChecked())
    us.setValue("DECT/D", self.device.currentText)

  def cleanup(self):
    pass
    
  def onDectChanged(self):
    self.dectChanged()
    self.onSelect()

  def onSelect(self):
    self.applyButton.enabled = self.inputa.currentNode() and self.inputb.currentNode() and self.dectExeGood and \
      (self.inputa.currentNode() != self.inputb.currentNode()) and \
      (self.outputa.currentNode() or self.outputb.currentNode() or self.outputc.currentNode() or self.outputm.currentNode())    

  def onApplyButton(self):
    logic = slicerdectLogic()
    logic.run(self.inputa.currentNode(), self.inputb.currentNode(), int(self.alphaa.text), int(self.betaa.text), int(self.gammaa.text), int(self.alphab.text), int(self.betab.text), int(self.gammab.text), self.outputa.currentNode(), self.outputb.currentNode(), self.outputc.currentNode(), self.outputm.currentNode(), float(self.mratio.text), self.enhanced.isChecked(), self.flip.isChecked(), self.dectapp.text, self.device.currentIndex, self.progbar)

#
# slicer-dectLogic
#

class slicerdectLogic(ScriptedLoadableModuleLogic):
  """This class should implement all the actual
  computation done by your module.  The interface
  should be such that other python code can import
  this class and make use of the functionality without
  requiring an instance of the Widget.
  Uses ScriptedLoadableModuleLogic base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def run(self, inputa, inputb, alphaa, betaa, gammaa, alphab, betab, gammab, outputa, outputb, outputc, outputm, mratio, enhanced, flip, p, device, pb = None):
    """
    Run the actual algorithm
    """

    logging.info('Processing started')
    
    if pb is None:
      pass
    else:
      pb.setValue(0)
      slicer.app.processEvents()
      
    td = slicer.util.tempDirectory("dect")
    ia = os.path.join(td, "a.tiff")
    ib = os.path.join(td, "b.tiff")
    ox = os.path.join(td, "x.tiff")
    oy = os.path.join(td, "y.tiff")
    oz = os.path.join(td, "z.tiff")
    om = os.path.join(td, "m.tiff")
    
    spacing = inputa.GetSpacing()
    origin = inputa.GetOrigin()
    
    iasn_origfn = None
    ibsn_origfn = None
    if(inputa.GetStorageNode() is not None):
      iasn_origfn = inputa.GetStorageNode().GetFileName()
    if(inputb.GetStorageNode() is not None):
      ibsn_origfn = inputb.GetStorageNode().GetFileName()
    
    slicer.util.saveNode(inputa, ia)
    slicer.util.saveNode(inputb, ib)
    
    iasn = inputa.GetStorageNode()
    ibsn = inputb.GetStorageNode()
    
    if(iasn_origfn is not None):
      iasn.SetFileName(iasn_origfn)
    else:
      iasn.SetFileName(iasn.GetFileName().replace(".tiff", ".nrrd"))
    
    if(ibsn_origfn is not None):
      ibsn.SetFileName(ibsn_origfn)
    else:
      ibsn.SetFileName(ibsn.GetFileName().replace(".tiff", ".nrrd"))
    
    startupinfo = None
    if os.name == 'nt':
      startupinfo = subprocess.STARTUPINFO()
      startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
      
    pargs = [ p, '-A', ia, '-B', ib, '-a', str(alphaa), '-b', str(betaa), '-c', str(gammaa), '-d', str(alphab), '-e', str(betab), '-f', str(gammab), '-D', str(device), '-M', om, '-x', ox, '-y', oy, '-z', oz, '-r', str(mratio), '-Z', '-s' ]
    if(enhanced):
      pargs.append('-E')
    if(flip):
      pargs.append('-F')
      
    logging.info(pargs)
    
    p2 = subprocess.Popen(pargs, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, startupinfo=startupinfo)

    p2.wait()
    
    if(outputa is not None):
      sn = slicer.vtkMRMLVolumeArchetypeStorageNode()
      sn.SetFileName(ox)
      sn.ReadData(outputa)
      outputa.SetSpacing(spacing)
      outputa.SetOrigin(origin)
      sn.SetFileName(ox.replace(".tiff", ".nrrd"))
   
    if(outputb is not None):
      sn = slicer.vtkMRMLVolumeArchetypeStorageNode()
      sn.SetFileName(oy)
      sn.ReadData(outputb)
      outputb.SetSpacing(spacing)
      outputb.SetOrigin(origin)
      sn.SetFileName(ox.replace(".tiff", ".nrrd"))
   
    if(outputc is not None):
      sn = slicer.vtkMRMLVolumeArchetypeStorageNode()
      sn.SetFileName(oz)
      sn.ReadData(outputc)
      outputc.SetSpacing(spacing)
      outputc.SetOrigin(origin)
      sn.SetFileName(ox.replace(".tiff", ".nrrd"))
   
    if(outputm is not None):
      sn = slicer.vtkMRMLVolumeArchetypeStorageNode()
      sn.SetFileName(om)
      sn.ReadData(outputm)
      outputm.SetSpacing(spacing)
      outputm.SetOrigin(origin)
      sn.SetFileName(ox.replace(".tiff", ".nrrd"))
   
    logging.info('Processing completed')
      
    if pb is None:
      pass
    else:
      pb.setValue(100)
      slicer.app.processEvents()

    if outputm is not None:
      slicer.util.setSliceViewerLayers(background=outputm)
      for sliceViewName in slicer.app.layoutManager().sliceViewNames():
        sw = slicer.app.layoutManager().sliceWidget(sliceViewName)
        sw.sliceLogic().FitSliceToAll()

    return True


class slicerdectTest(ScriptedLoadableModuleTest):
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
    self.test_slicerdect1()

  def test_slicerdect1(self):
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
    logic = slicerdectLogic()
    self.assertTrue( logic.hasImageData(volumeNode) )
    self.delayDisplay('Test passed!')
