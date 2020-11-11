import sys
from os import environ
from os import getcwd
import string

sys.path.append(environ["PYTHON_MODULE_PATH"])


import CompuCellSetup
CompuCellSetup.setSimulationXMLFileName("2D_vasculogenesis_control.xml")

sim,simthread = CompuCellSetup.getCoreSimulationObjects()
CompuCellSetup.initializeSimulationObjects(sim,simthread)
import CompuCell #notice importing CompuCell to main script has to be done after call to getCoreSimulationObjects()

pyAttributeAdder,listAdder=CompuCellSetup.attachListToCells(sim)          ####Assigns a list to each cell in the cellList inventory. List elements are later attached to each cell.
pyAttributeAdderDict,dictAdder=CompuCellSetup.attachDictionaryToCells(sim)      ####Assigns a dictionary to each cell in the cellList inventory. Dictionary elements are later attached to each cell.

#Add Python steppables here
from PySteppablesExamples import SteppableRegistry
steppableRegistry=SteppableRegistry()

from steppableBasedMitosisSteppables_control import VolumeParamSteppable
volumeParamSteppable=VolumeParamSteppable(sim,10)
steppableRegistry.registerSteppable(volumeParamSteppable)

from steppableBasedMitosisSteppables_control import MitosisSteppable
mitosisSteppable=MitosisSteppable(sim,10)
steppableRegistry.registerSteppable(mitosisSteppable)

# from steppableBasedMitosisSteppables import ProteinExpression
# proteinExpression=ProteinExpression(sim,10)
# steppableRegistry.registerSteppable(proteinExpression)

from steppableBasedMitosisSteppables_control import SecretionCellSteppable
secretionCellSteppable=SecretionCellSteppable(sim,10)
steppableRegistry.registerSteppable(secretionCellSteppable)


CompuCellSetup.mainLoop(sim,simthread,steppableRegistry)



