# -*- coding: utf-8 -*-
"""
Contains methods to create, end, and control SUMO simulation.
last major edit Michael 12/21/15
"""
import subprocess, sys, os, constants

import traci
from math import radians, sin, cos, pi, fmod
from time import sleep

class Sumo:
    
    def __init__(self, configFile, gui = False, outFile=None):
        """ starts the simulation
            inputs:
            configFile = string, name of the configuration you're using
            gui = boolean (optional), if True, the SUMO gui will pop up
            outFile = string (optional), file for SUMO to save data
        """

        PORT = 8800     
        
        # from sumo demo code (vehicleControl.py)
        try:
            from sumolib import checkBinary
        except ImportError:
            def checkBinary(name):
                return name 
        
        # write terminal command
        completeCommand = [checkBinary("sumo")]
        if gui:
            completeCommand = [checkBinary("sumo-gui")]
        sumoConfig = configFile + "/" + configFile + ".sumocfg"
        completeCommand += ["-c", sumoConfig]
        completeCommand += ["--remote-port",str(PORT)]
        if outFile is not None:
            completeCommand += ["--fcd-output","./Results/%s.xml" % outFile ]
        
        self._C = completeCommand        
        
        ## start SUMO
        self.sumoProcess = subprocess.Popen(completeCommand,
                                            stdout=subprocess.PIPE,#sys.stdout,#
                                            stderr=subprocess.STDOUT)
        traci.init(PORT, 10)
        
        self.outFile = outFile
    
        
    def createVehicle(self,vehID, laneID, pos=0):
        routeID = _edge(laneID)
        laneIndex = int( laneID[laneID.rfind('_')+1:] )
        traci.vehicle.add(vehID,routeID,pos=pos,lane=laneIndex)
        traci._sendIntCmd(traci.constants.CMD_SET_VEHICLE_VARIABLE,
                          traci.constants.VAR_SPEEDSETMODE, vehID, 0)
        traci.vehicle.setLaneChangeMode(vehID, 0)
        self.simulationStep()
        return 0

        
    def moveVehicle(self,vehID, lane, pos=0.):
        prevLane = traci.vehicle.getLaneID(vehID)
        
        if lane is not prevLane:
            thisroute = traci.vehicle.getRoute(vehID)
            if not _edge(lane) in thisroute: # have to change route
                changedRoute = False
                for r in traci.route.getIDList():
                    redges = traci.route.getEdges(r)
                    if _edge(prevLane) in redges and _edge(lane) in redges:
                        traci.vehicle.setRouteID(vehID, r)
                        changedRoute=True
                        break
                if not changedRoute:
                    print("failed to find route, probably crashing")
        
        traci.vehicle.moveTo(vehID, lane, pos)
        traci.vehicle.setSpeed(vehID,0)
        self.simulationStep()
        return 0
        
        
    def moveVehicleAlong(self,vehID, dist, lane=None):
        prevLane = traci.vehicle.getLaneID(vehID)
        prevPos = traci.vehicle.getLanePosition(vehID)
        
        if lane is None: # assume you can remain on current lane
            lane = prevLane           
            
        if lane[0]==':': #intersection, can't select lane because SUMO is dumb
            necessarySpeed=dist
            traci.vehicle.setSpeed(vehID,necessarySpeed*10.)
            for i in range(1):
                self.simulationStep()
            traci.vehicle.setSpeed(vehID,0.)
            return 0
            
        if lane is prevLane: # staying in same lane
            currPos = prevPos + dist
        else: # changing lanes, assume you complete old lane
            thisroute = traci.vehicle.getRoute(vehID)
            if not _edge(lane) in thisroute: # have to change route
                changedRoute = False
                for r in traci.route.getIDList():
                    redges = traci.route.getEdges(r)
                    if _edge(prevLane) in redges and _edge(lane) in redges:
                        traci.vehicle.setRouteID(vehID, r)
                        changedRoute=True
                        break
                if not changedRoute:
                    print("failed to find route, probably crashing")
            currPos = prevPos + dist - traci.lane.getLength(prevLane)
            if currPos < 0 and dist > 0:
                currPos = prevPos + dist
                lane = prevLane # don't change lanes if you haven't reached next one
        
        if dist == 0:
            return 0
            
        traci.vehicle.moveTo(vehID, lane, currPos)
        traci.vehicle.setSpeed(vehID,0)
        self.simulationStep()
        return 0
        
    def removeVehicle(self,vehID):
        traci.vehicle.remove(vehID)
        self.simulationStep()
        return 0
    
    
    def getVehicleState(self,vehID):
        """ input: vehID = string \n
            output: [laneID = string, position on lane = num,
                     coordinate position = (num,num), angle = num (deg)]
        """
        try:
            sf = traci.vehicle.getLaneID(vehID)
        except:
            return [None]*4
        return [sf,#traci.vehicle.getLaneID(vehID),
                traci.vehicle.getLanePosition(vehID),
                traci.vehicle.getPosition(vehID),
                radians(traci.vehicle.getAngle(vehID))]
        
    def getLaneInfo(self,lane):
        """ input: laneID = string \n
            output: [lane length = num, lane width = num,
                     IDs of upcoming connected lanes = [string] ]
        """
        # traci returns extra information, as below...
        ##getLinks(string) -> list((string, bool, bool, bool))
        ##A list containing ids of successor lanes together with
        ##priority, open and foe.
        
        successorlinks = traci.lane.getLinks(lane)
        linknames = [a[0] for a in successorlinks]
        return [traci.lane.getLength(lane),
                traci.lane.getWidth(lane),
                linknames]
    
    def end(self):
        self.sumoProcess.terminate()
        if "" in traci._connections:
            traci._connections[""].close()
            del traci._connections[""]
            
        if self.outFile is not None: # transport results to csv
            thisSumoOutFile = "./Results/" + self.outFile + ".xml"
            thisCsvFile = "./Results/" + self.outFile + ".csv"
            os.system("python " + toolsPathName + "/xml/xml2csv.py " +
                        thisSumoOutFile + " --output " + thisCsvFile)
                        
    def route(self,vehID):
        return traci.vehicle.getRoute(vehID)
        
    def addCrashSymbol(self, aState, bState, ncrashes):
        acenterx = aState.x - 5./2.*sin(aState.angle)
        bcenterx = bState.x - 5./2.*sin(bState.angle)
        acentery = aState.y + 5./2.*cos(aState.angle)
        bcentery = bState.y + 5./2.*cos(bState.angle)
        center = (acenterx/2. + bcenterx/2., acentery/2. + bcentery/2.)
        circlePoints = []
        for k in range(16):
            angle = k*2*pi/16.
            radius = 2. + fmod(k,2)*1.
            circlePoints += [(center[0] + radius*cos(angle),
                              center[1] + radius*sin(angle))]
        crashcircle = 'crashcircle'+str(ncrashes % 4 + 1)
        traci.polygon.setShape(crashcircle, circlePoints)
        
    def addSensorSymbol(self):
        traci.polygon.setColor('sensor',(0,255,0,0))
    def removeSensorSymbol(self):
        traci.polygon.setColor('sensor',(255,255,255,0)) 
       
    def track(self, vehID):
        traci.gui.trackVehicle("View #0", vehID)
        
    def screen(self, x1, x2):
        traci.gui.setOffset('View #0', x2/2.+x1/2., 8.)
        
    def simulationStep(self):
        sleep(.01)
        traci.simulationStep()
        sleep(.01)
    
def _edge(laneID):
    return laneID[:laneID.rfind('_')]
