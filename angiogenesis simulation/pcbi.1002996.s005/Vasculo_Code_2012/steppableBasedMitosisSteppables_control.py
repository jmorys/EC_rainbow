from PySteppables import *
from PySteppablesExamples import MitosisSteppableBase
import CompuCell
import sys
from random import *
from math import * 
import time
import os

current_path = os.getcwd()

run_time = time.strftime("%m_%d_%Y_%I_%M_%S", time.localtime())

## For Hex
lmfLength=1*sqrt(2.0/(3.0*sqrt(3.0)))*sqrt(3.0)
xScale=1.0
yScale=1*sqrt(3.0)/2.0
zScale=1*sqrt(6.0)/3.0
## For Hex

class SecretionCellSteppable(SteppableBasePy):
    def __init__(self,_simulator,_frequency=1):
        SteppableBasePy.__init__(self,_simulator, _frequency)
        self.simulator=_simulator
        self.inventory=self.simulator.getPotts().getCellInventory()
        self.cellList=CellList(self.inventory)
        self.nTrackerPlugin=CompuCell.getNeighborTrackerPlugin()
        self.fieldNameExternalVEGF = 'VEGF_ext'
        self.fieldNameSolubleVEGF = 'VEGF_sol'
        self.fieldNameCXCL10 = 'CXCL10'
        self.fieldNameCCL2 = 'CCL2'

    def start(self):
        for cell in self.cellList:
            vasculo_attributes=CompuCell.getPyAttrib(cell)

    def step(self,mcs):
        x=time.clock()
        print 'Secretion Cell Steppable start:'
        VEGFsolSecretor=self.getFieldSecretor("VEGF_sol")
        VEGFextSecretor=self.getFieldSecretor("VEGF_ext")
        CCL2Secretor=self.getFieldSecretor("CCL2")
        CXCL10Secretor=self.getFieldSecretor("CXCL10")
        solFlt1Secretor=self.getFieldSecretor("sol_Flt1")
        ProteaseSecretor=self.getFieldSecretor("Protease")
        
        for cell in self.cellList:
            vasculo_attributes=CompuCell.getPyAttrib(cell)
            if cell.type==1:  
                totalArea = 0
                totalTIPArea = 0               
                pt=CompuCell.Point3D()
                pt.x=int(round(cell.xCM/max(float(cell.volume),0.001)))
                pt.y=int(round(cell.yCM/max(float(cell.volume),0.001)))
                pt.z=int(round(cell.zCM/max(float(cell.volume),0.001)))
                cellNeighborList=CellNeighborListAuto(self.nTrackerPlugin,cell)
                for neighborSurfaceData in cellNeighborList:
          #Check to ensure cell neighbor is not medium or other cell types
                    if neighborSurfaceData.neighborAddress:
                        if neighborSurfaceData.neighborAddress.type == 4:               
            #sum up common surface area of cell with its neighbors
                            totalTIPArea+=neighborSurfaceData.commonSurfaceArea
                #print 'cell id:', cell.id, ' Area shared with tip cell is:', totalTIPArea
                if totalTIPArea>5:
                    solFlt1Secretor.secreteOutsideCellAtBoundary(cell,0.01)  #Flt1 inhibition, can implement 10fold drop in secretion
                
                for neighborSurfaceData in cellNeighborList:
          #Check to ensure cell neighbor is not medium or other cell types
                    if neighborSurfaceData.neighborAddress:               
            #sum up common surface area of cell with its neighbors
                        if neighborSurfaceData.neighborAddress.type == 1 or neighborSurfaceData.neighborAddress.type == 2 or neighborSurfaceData.neighborAddress.type == 3:
                            totalArea+=neighborSurfaceData.commonSurfaceArea
                     
                
        print "test sec", time.clock()-x        


              

class VolumeParamSteppable(SteppableBasePy):
    def __init__(self,_simulator,_frequency=1):
        SteppableBasePy.__init__(self,_simulator,_frequency)
        self.simulator=_simulator
        self.inventory=self.simulator.getPotts().getCellInventory()
        self.cellList=CellList(self.inventory)
        self.nTrackerPlugin=CompuCell.getNeighborTrackerPlugin()
        self.fieldNameExternalVEGF = 'VEGF_ext'
        self.fieldNameSolubleVEGF = 'VEGF_sol'
        self.fieldNameCXCL10 = 'CXCL10'
        self.fieldNameCCL2 = 'CCL2'
    def start(self):
        self.pt=CompuCell.Point3D()  # set uniform VEGF_ext field for ECM
        self.tempvar=os.getcwd()+"/vasculo_steppableBasedMitosis_py_"+run_time+"_Data.txt"
        
        totaldatafilename=open(self.tempvar, "w")
        totaldatafilename.write("MCS\tId\tType\tVolume\tSurfaceArea\tX_Location\tY_Location\tVEGF165\tVEGF121\tTotalVEGF\tCXCL10\tCCL2\tGrowing\tArrested\tQuiescent\tApoptotic\tTotal Cell Number\n") #first row, tab delimited
        totaldatafilename.close()
        
        for x in range(self.dim.x):
            for y in range(self.dim.y):
                CompuCell.getConcentrationField(self.simulator,"VEGF_ext").set(self.pt,.05)

        for cell in self.cellList:
            if cell.type==1:   # endothelial stalk cells
                cell.targetVolume=30
                cell.lambdaVolume=6.0
                cell.targetSurface=4*sqrt(cell.targetVolume)
                cell.lambdaSurface=4.0
            if cell.type==2:   # macrophage/inflammatory cells
                cell.targetVolume=40
                cell.lambdaVolume=6.0
                cell.targetSurface=4*sqrt(cell.targetVolume)
                cell.lambdaSurface=4.0
            if cell.type==3:    # mural/VSMC cells
                cell.targetVolume=50
                cell.lambdaVolume=6.0
                cell.targetSurface=4*sqrt(cell.targetVolume)
                cell.lambdaSurface=4.0
            if cell.type==4:   # endothelial tip cells
                cell.targetVolume=30
                cell.lambdaVolume=6.0
                cell.targetSurface=5*sqrt(cell.targetVolume)
                cell.lambdaSurface=8.0

    #Increase volume based on VEGF concentration and shared surface area            
    def step(self,mcs):
        fieldNeoVasc=CompuCell.getConcentrationField(self.simulator,self.fieldNameExternalVEGF)
        fieldNeoVascSol=CompuCell.getConcentrationField(self.simulator,self.fieldNameSolubleVEGF)
        fieldAntiVasc=CompuCell.getConcentrationField(self.simulator,self.fieldNameCXCL10)
        fieldProVasc=CompuCell.getConcentrationField(self.simulator,self.fieldNameCCL2)      
        ECapoptosisratevariable=20000
        ICapoptosisratevariable=100000   #PBMC cytotox
        MCapoptosisratevariable=100000	 #SMC cytotox
        totalNum = 0
        print "Start Volume Steppable"
        for cell in self.cellList:
            totalNum+=1
            vasculo_attributes=CompuCell.getPyAttrib(cell)            
            if cell.type==1:
                Growing = 0
                Arrested = 0
                Quiescent = 0
                Apoptotic = 0
                totalArea = 0
                pt=CompuCell.Point3D()
                pt.x=int(round(cell.xCM/max(float(cell.volume),0.001)))
                pt.y=int(round(cell.yCM/max(float(cell.volume),0.001)))
                pt.z=int(round(cell.zCM/max(float(cell.volume),0.001)))
                VEGFextconcentration=fieldNeoVasc.get(pt)
                VEGFsolconcentration=fieldNeoVascSol.get(pt)
                CXCL10concentration=fieldAntiVasc.get(pt)
                CCL2concentration=fieldProVasc.get(pt)
                VEGFconcentration=VEGFextconcentration+VEGFsolconcentration
                cellNeighborList=CellNeighborListAuto(self.nTrackerPlugin,cell)
                for neighborSurfaceData in cellNeighborList:
          #Check to ensure cell neighbor is not medium or other cell types
                    if neighborSurfaceData.neighborAddress:
                        if neighborSurfaceData.neighborAddress.type == 1 or neighborSurfaceData.neighborAddress.type == 4 or neighborSurfaceData.neighborAddress.type == 3:               
            #sum up common surface area of cell with its neighbors
                            totalArea+=neighborSurfaceData.commonSurfaceArea
                print 'total shared area:',totalArea       
                print "Heparin bound VEGF concentration:",VEGFextconcentration
                print "Soluble VEGF concentration:",VEGFsolconcentration
                print "Total VEGF concentration:",VEGFconcentration
                print "CXCL10 concentration:",CXCL10concentration
                print "CCL2 concentration:",CCL2concentration
                
                if  randint(1,ECapoptosisratevariable)==1:
                    cell.type=6
                    Apoptotic = 1
                if totalArea<26 and VEGFconcentration>0.4: #total area shared with other cells, VEGF threshold                    
                    if CXCL10concentration>0.3:
                        if CXCL10concentration>0.6:
                            if  randint(1,ECapoptosisratevariable/100)==1:
                                cell.type=6
                                Apoptotic = 1
                        
                        else:
                            cell.targetVolume-=0.001
                            cell.targetSurface=4*sqrt(cell.targetVolume)
                            #print 'Growth arrested by CXCL10'
                        
                            Arrested = 1
                    else:

                        if VEGFconcentration>1:
                            cell.type=4
                        else:
                            cell.targetVolume+=0.8
                            cell.targetSurface=4*sqrt(cell.targetVolume)
                        #print 'Growth due to VEGF'
                            Growing = 1
                else:
                    if totalArea>45:
                        if  randint(1,ECapoptosisratevariable/10)==1:
                            cell.type=6
                            Apoptotic = 1
                    else:
                    #print 'Cell quiescent'
                        Quiescent = 1
                
                if mcs%50==0:
                    namelist=[]
                    namelist.append(str(mcs))                                                       ###Current MCS
                    namelist.append(str(cell.id))
                    namelist.append(str(cell.type))
                    namelist.append(str(cell.volume))
                    namelist.append(str(cell.surface))
                    namelist.append(str(cell.xCM/(cell.volume*lmfLength*xScale)))
                    namelist.append(str(cell.yCM/(cell.volume*lmfLength*yScale)))
                    namelist.append(str(VEGFextconcentration))
                    namelist.append(str(VEGFsolconcentration))
                    namelist.append(str(VEGFconcentration))
                    namelist.append(str(CXCL10concentration))
                    namelist.append(str(CCL2concentration))
                    namelist.append(str(Growing))
                    namelist.append(str(Arrested))
                    namelist.append(str(Quiescent))
                    namelist.append(str(Apoptotic))
                    namelist.append(str(totalNum))
                    
                    totaldatafilename=open(self.tempvar, "a")
                    totaldatafilename.write('\t'.join(namelist))
                    totaldatafilename.write('\n')  
                    totaldatafilename.close()
            if cell.type==2:
                randommultiplier=uniform(0.8,1.2)
                #if mcs>2000: # and mcs<12000:  ## add delays/adjustments in proliferation
                cell.targetVolume+=(0.01*randommultiplier)   #Inflammatory Cell Proliferation
                cell.targetSurface=4*sqrt(cell.targetVolume)
                    
                if  randint(1,ICapoptosisratevariable)==1:
		    cell.type=6
            if cell.type==3:
		randommultiplier=uniform(0.8,1.2)
		#if mcs>2000: ## delay in mural proliferation
		cell.targetVolume+=(0.01*randommultiplier)  ##Mural Cell Proliferation
		cell.targetSurface=4*sqrt(cell.targetVolume)
		totalECArea = 0
		pt=CompuCell.Point3D()
		pt.x=int(round(cell.xCM/max(float(cell.volume),0.001)))
		pt.y=int(round(cell.yCM/max(float(cell.volume),0.001)))
		pt.z=int(round(cell.zCM/max(float(cell.volume),0.001)))
		cellNeighborList=CellNeighborListAuto(self.nTrackerPlugin,cell)

		for neighborSurfaceData in cellNeighborList:
			#Check to ensure cell neighbor is not medium or other cell types
		    if neighborSurfaceData.neighborAddress:
                        if neighborSurfaceData.neighborAddress.type == 1: #or neighborSurfaceData.neighborAddress.type == 4:               
            #sum up common surface area of cell with its neighbors
                            totalECArea+=neighborSurfaceData.commonSurfaceArea
                # print 'total EC area:',totalECArea       
                # print "Heparin bound VEGF concentration:",VEGFextconcentration
                # print "Soluble VEGF concentration:",VEGFsolconcentration
                # print "Total VEGF concentration:",VEGFconcentration
                

                    if totalECArea>5:                                  
                        cell.targetVolume+=(0.02*randommultiplier)  #Mural Cell Proliferation
                        cell.targetSurface=4*sqrt(cell.targetVolume)
                            # print 'VSMC Growth due to VEGF'
                    if randint(1,MCapoptosisratevariable)==1:
                        cell.type=6
				
            if cell.type==6:
                print "test",cell.targetVolume
                if cell.targetVolume>1:
                    cell.targetVolume-=0.5
                else:
                    cell.targetVolume=0
                    cell.targetSurface=4*sqrt(cell.targetVolume)
                    print "test2"
        #print "emd test"
            #if mcs>1000:
            if cell.type==4:
                totalArea = 0
                freeArea = cell.surface
                pt=CompuCell.Point3D()
                pt.x=int(round(cell.xCM/max(float(cell.volume),0.001)))
                pt.y=int(round(cell.yCM/max(float(cell.volume),0.001)))
                pt.z=int(round(cell.zCM/max(float(cell.volume),0.001)))
                cellNeighborList=CellNeighborListAuto(self.nTrackerPlugin,cell)
                for neighborSurfaceData in cellNeighborList:
          #Check to ensure cell neighbor is not medium or other cell types
                    if neighborSurfaceData.neighborAddress:
                        if neighborSurfaceData.neighborAddress.type == 1 or neighborSurfaceData.neighborAddress.type == 4:               
            #sum up common surface area of cell with its neighbors
                            totalArea+=neighborSurfaceData.commonSurfaceArea
                freeArea-=totalArea            
                #print 'total free area:',freeArea
                if freeArea<0.001:
                    cell.type=1
                
                

#

                    
                    
class MitosisSteppable(MitosisSteppableBase):

    def __init__(self,_simulator,_frequency=1):
        MitosisSteppableBase.__init__(self,_simulator, _frequency)
        self.fieldNameExternalVEGF = 'VEGF_ext'
        self.fieldNameSolubleVEGF = 'VEGF_sol'
        
    def step(self,mcs):
        print "INSIDE MITOSIS STEPPABLE"
        self.fieldNeoVasc=CompuCell.getConcentrationField(self.simulator,self.fieldNameExternalVEGF)
        self.fieldNeoVascSol=CompuCell.getConcentrationField(self.simulator,self.fieldNameSolubleVEGF)
        
        cells_to_divide=[]
        
        for cell in self.cellList:
            vasculo_attributes=CompuCell.getPyAttrib(cell)
            
            if cell.type==1:
                if cell.volume>55:
                    cells_to_divide.append(cell)   
            if cell.type==3:
                if cell.volume>90:
                    cells_to_divide.append(cell)   
            if cell.type==2:
                if cell.volume>75:
                    cells_to_divide.append(cell)   
                 
        for cell in cells_to_divide:
            # to change mitosis mode leave one of the below lines uncommented
            self.divideCellRandomOrientation(cell)                  
            # self.divideCellOrientationVectorBased(cell,1,1,0)                 # this is a valid option
            # self.divideCellAlongMajorAxis(cell)          

    def updateAttributes(self):
        parentCell=self.mitosisSteppable.parentCell
        childCell=self.mitosisSteppable.childCell
        child_vasculo_attributes=CompuCell.getPyAttrib(childCell)
        parent_vasculo_attributes=CompuCell.getPyAttrib(parentCell)
        pt=CompuCell.Point3D()
        pt.x=int(round(childCell.xCM/max(float(childCell.volume),0.001)))
        pt.y=int(round(childCell.yCM/max(float(childCell.volume),0.001)))
        pt.z=int(round(childCell.zCM/max(float(childCell.volume),0.001)))
        VEGFextconcentration=self.fieldNeoVasc.get(pt)
        VEGFsolconcentration=self.fieldNeoVascSol.get(pt)
        VEGFconcentration=VEGFextconcentration+VEGFsolconcentration
                        
        parentCell.targetVolume/=2
        parentCell.lambdaVolume=5
        parentCell.targetSurface=4*sqrt(parentCell.targetVolume)
        parentCell.lambdaSurface=4
        
        childCell.targetVolume=parentCell.targetVolume
        childCell.lambdaVolume=5
        childCell.targetSurface=4*sqrt(childCell.targetVolume)
        childCell.lambdaSurface=4
        
             
        
        if parentCell.type==1:
            if randint(1,5)==1 and VEGFconcentration>0.4:
                childCell.type=4  #stalk cell
            else:
                childCell.type=1  #tip cell
            
        if parentCell.type==3:
            childCell.type=3
        if parentCell.type==2:
            childCell.type=2
            
        child_vasculo_attributes.update(parent_vasculo_attributes)    
        # if parentCell.type==2:
            # childCell.type=2

class ProteinExpression(SteppableBasePy):
    def __init__(self,_simulator,_frequency=1):
        SteppableBasePy.__init__(self,_simulator, _frequency)
    
    def start(self):
        for cell in self.cellList:
            vasculo_attributes=CompuCell.getPyAttrib(cell)
            
            if cell.type==1:
                vasculo_attributes['cell.receptor.VEGFR1']=100
                vasculo_attributes['cell.receptor.VEGFR2']=1000
    
    def step(self,mcs):
        print "INSIDE Protein Expression STEPPABLE"
        randommultiplier1=uniform(0.95,1.05)
        randommultiplier2=uniform(0.95,1.05)
        for cell in self.cellList:
            vasculo_attributes=CompuCell.getPyAttrib(cell)
            if cell.type==1:
               
                vasculo_attributes['cell.receptor.VEGFR1']*=randommultiplier1
                vasculo_attributes['cell.receptor.VEGFR2']*=randommultiplier2


                              
