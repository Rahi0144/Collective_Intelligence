# -*- coding: utf-8 -*-
from sumoMethodsWindows import Sumo as Simulator
#from ownSimulator import OwnSim as Simulator
import pandas as pd
import numpy as np
import time
import collisionCheck
from Qsimformat import formatState
from QsimInit import initialize
### import QTableIndex
from Agent_2405 import Agents as Agent
from Agent_2405 import agent_totalNumStates
import crossingTTC_rect as xxt
from QtableUpdates import updateQvalue_simple, updateQvalue_2step

numiter = 3
trainingTable = False
#QTable = np.zeros((agent_totalNumStates,21)) # run these two lines to initialize
#Qcount = np.zeros((agent_totalNumStates,21))
#QTable = np.load('qtable_4way_l.npy.npy')
#Qcount = np.load('qtable_4way_l.npy_count.npy')
#Qcount = np.load('qtable_4way_count.npy')


Stats_train=pd.Series([0.]*6,index=['Collision','Untrained Collision','Reward',
                                      'Time','WrongPath','Cars'])
VEHsize = (5.,2.) # meters, length by width
DELTAT = .1
MS2KPH = 3.6
ttcCheck_speedMargins = 10. / MS2KPH # m/s

roadOrder = pd.DataFrame({'left':[4,3,1,2],'right':[3,4,2,1],
                          'straight':[2,1,4,3]},index=[1,2,3,4])
intersectionLookUp = [['1i_0','3o_0',':0_12_0'] , ['1i_0','2o_0',':0_13_0'] ,
                      ['1i_0','2o_1',':0_13_1'] , ['1i_1','4o_1',':0_15_0'] ,
                      ['2i_0','4o_0',':0_4_0'] , ['2i_0','1o_0',':0_5_0'] ,
                      ['2i_0','1o_1',':0_5_1'] , ['2i_1','3o_1',':0_7_0'] ,
                      ['3i_0','2o_0',':0_8_0'] , ['3i_0','4o_0',':0_9_0'] ,
                      ['3i_0','4o_1',':0_9_1'] , ['3i_1','1o_1',':0_11_0'] ,
                      ['4i_0','1o_0',':0_0_0'] , ['4i_0','3o_0',':0_1_0'] ,
                      ['4i_0','3o_1',':0_1_1'] , ['4i_1','2o_1',':0_3_0'] ]
intersectionInfo = xxt.getInterInfo('sumo')

def getRoadLength(lane):
    if lane[0]==':':
        inroad,outroad = next(((i,j) for i,j,k in intersectionLookUp if k==lane))
        roadDirections = roadOrder.loc[splitRoad(inroad)[0]]
        outroute = splitRoad(outroad)[0]
        if roadDirections['left'] == outroute:
            return 15.64
        elif roadDirections['right'] == outroute:
            return 5.
        else:
            return 16.32
    return 91.95

def splitRoad(lane):
    if lane[0]==':': # internal lane for intersection
        inroad = next((i for i,j,k in intersectionLookUp if k==lane))
        return [int(inroad[0]) , ':' , int(inroad[len(inroad)-1])]
    rt,ln = lane.split('_')
    return [int(rt[0]), rt[1], int(ln)]
    
def makeRoad(roadNum, io, ln):
    return str(roadNum)+io+'_'+str(ln)

def findFault(carA, carB):
    aheadLanes = []
    #futureAheadLane = None # without discounted reward this has no effect
    returnVal = 2
    if (carA['lane']== carB['lane']):
        if (carA['lanepos'] > carB['lanepos']):
            returnVal = 1
        else:
            returnVal = 0
            
    for joint in intersectionLookUp:
        if joint[0] == carA['lane']:
            aheadLanes += [joint[2]]
            #if joint[1] == car['dest']: # include ahead lane only if it is known dest.
            #    futureAheadLane = joint[1]
        if joint[2] == carA['lane']:
            aheadLanes = [joint[1]]
    for aheadLane in aheadLanes:    
        if carB['lane']==aheadLane:
            returnVal = 0
#    
    aheadLanes = []
    #futureAheadLane = None # without discounted reward this has no effect
    for joint in intersectionLookUp:
        if joint[0] == carB['lane']:
            aheadLanes += [joint[2]]
            #if joint[1] == car['dest']: # include ahead lane only if it is known dest.
            #    futureAheadLane = joint[1]
        if joint[2] == carB['lane']:
            aheadLanes = [joint[1]]
    for aheadLane in aheadLanes:    
        if carA['lane']==aheadLane:
            returnVal = 1

    if returnVal < 2:
        if (carA['lanechanged']>0):
            return 0
        elif(carB['lanechanged']>0):
            return 1
        return returnVal

    adir = 1 # 0 left 1 straight or right
    bdir = 1

    route,gradeA,lane = splitRoad(carA['lane'])
    route,gradeB,lane = splitRoad(carB['lane'])
    if gradeA == ':' and gradeB == 'o':
        return 0
    if gradeB == ':' and gradeA == 'o':
        return 1
        
    if gradeA == ':': # in intersection lane
        beforeA,afterA = next(((i,j) for i,j,k in intersectionLookUp if carA['lane']==k))
        beforeA,aa,bb = splitRoad(beforeA)
        afterA,aa,bb = splitRoad(afterA)
        if roadOrder.loc[beforeA,'left'] == afterA: # A turning left
            adir = 0
    if gradeB == ':': # in intersection lane
        beforeB,afterB = next(((i,j) for i,j,k in intersectionLookUp if carB['lane']==k))
        beforeB,aa,bb = splitRoad(beforeB)
        afterB,aa,bb = splitRoad(afterB)
        if roadOrder.loc[beforeB,'left'] == afterB: # B turning left
            bdir = 0
    if adir == 0 and bdir == 1:
        return 0
    elif adir == 1 and bdir == 0:
        return 1
    if carA['priority'] == 0 and carB['priority'] == 1:
        return 0
    elif carA['priority'] == 1 and carB['priority'] == 0:
        return 1
    return 2
            
                
    


def gatherStateInfo(cars, trajectories, carID):
    car = cars.iloc[carID]
    route,grade,ln = splitRoad(car['lane'])
    destroute,o,destln = splitRoad(car['dest'])
    
    carUpdate = pd.Series()
    carUpdate['ID'] = carID
    # whether the vehicle is entering the intersection (or already entered)
    if grade=='o':
        carUpdate['Car Location'] = 2
    elif grade=='i':
        if car['lanepos'] <= 89.:
            carUpdate['Car Location'] = 0
        else:
            carUpdate['Car Location'] = 1
    else:
        carUpdate['Car Location'] = 3
    # road and lane number for the vehicle (0 is right, 1 left)
    carUpdate['Road'] = route
    carUpdate['Lane Number'] = ln
    # destination road and lane
    carUpdate['Destination Road'] = destroute
    carUpdate['Destination Lane'] = destln
    # turn direction for the vehicle (0 left, 1 straight, 2 right)
    if destroute == roadOrder.loc[route].left:
        carUpdate['Turn Direction'] = 0
    elif destroute == roadOrder.loc[route].straight:
        carUpdate['Turn Direction'] = 1
    else:
        carUpdate['Turn Direction'] = 2
    if grade == 'o':
        carUpdate['Turn Direction'] = 1
    # whether lane is correct for destination
    if ln == 0:
        carUpdate['Correct Lane'] = carUpdate['Turn Direction']>=1
    else:
        carUpdate['Correct Lane'] = carUpdate['Turn Direction']==0
    if grade == 'o':   # are they at the correct destination
        if route == destroute:
            carUpdate['Correct Lane'] = 1
        else:
            carUpdate['Correct Lane'] = 0
    # speed (discretized)
    carUpdate['Speed'] = car['speed']*MS2KPH
    # whether car collided
    carUpdate['Collision'] = (car['status'] >= 2) & (car['caratfault']==1)
   # carUpdate['Collision'] = car['status'] >= 2
    carUpdate['Colliding Vehicle'] = car['status'] - 2
    # adjacent lane is free (no collision if there is a lane change)
    if (grade=='i' and car['lanepos']>5.) or grade=='o':
        timeToLookAhead = 1.
        carUpdate['Adjacent Lane Free'] = 1 + ln
        adjlane = makeRoad(route,grade,1 - ln)
        searchAdjacent = (cars['status']==0)&(cars['lane']==adjlane)
        for otherCarID in np.where(searchAdjacent)[0]:
            alt = cars.iloc[otherCarID]
            carpositions = (np.arange(0,timeToLookAhead,DELTAT) *
                            car['speed'] + car['lanepos'])   
            altpositions = (np.arange(0,timeToLookAhead,DELTAT) *
                            alt['speed'] + alt['lanepos'])                
            if np.any(np.abs(altpositions - carpositions) <= VEHsize[0]+1.):
                carUpdate['Adjacent Lane Free'] = 0
        aheadLanePos = car['lanepos'] + car['speed']*DELTAT - 86.95
        if aheadLanePos > 0: # added 5/25, temp fix
            adjacentAheadLanes = list((k for i,j,k in intersectionLookUp if i==adjlane))
            searchAdjacentAhead = cars['status']==0
            for adjacentAheadLane in adjacentAheadLanes:
                searchAdjacentAhead = searchAdjacentAhead | (
                                            cars['lane'] == adjacentAheadLane)
            searchAdjacentAhead = searchAdjacentAhead & (cars['status']==0)
            searchAdjacentAhead = searchAdjacentAhead & (
                            cars['lanepos'] <= aheadLanePos)
            if np.any(searchAdjacentAhead):
                carUpdate['Adjacent Lane Free'] = 0
    else:
        carUpdate['Adjacent Lane Free'] = 0
        
    carUpdate['Time to Nearest Vehicle'] = -1
    # find closest crossing vehicles        
    tNV = 100
    closestCar = -1
    secondTNV = 100
    slowTNV = 100
    fastTNV = 100
    DTAC = 100
    for crossing in trajectories[carID]: 
        altRoad = crossing[0]
        collideVehicles = (cars['status']==0)
        collideVehicles = collideVehicles & ((cars['ilane']==altRoad) > 0)
        blockingDist = 100.  ## blocking change starts here
        blocking = False
        tNVcc = 100
        closestCarcc = -1
        secondTNVcc = 100
        slowTNVcc = 100
        fastTNVcc = 100
        DTACcc = 100
        for altcar in np.where(collideVehicles)[0]:
            for altcrossing in trajectories[altcar]:
                origRoad = altcrossing[0]
                if origRoad == car['ilane']:
                    (ttc, slowttc, fastttc)  = xxt.checkTrajectories(crossing,
                                            altcrossing, ttcCheck_speedMargins)
                    if altcrossing[2] < blockingDist and altcrossing[3] >= 0:
                        blockingDist = altcrossing[2]
                        blocking = altcrossing[1] == 0 # only blocking if stopped
                        if blocking: DTAC = xxt.distanceToCrash(crossing,altcrossing,False)
                    elif blocking:
                        continue
                    if ttc < tNVcc or blocking:
                        secondTNVcc = tNVcc
                        tNVcc = ttc
                        closestCarcc = altcar
                        if ttc < 6:
                            DTACcc = xxt.distanceToCrash(crossing, altcrossing,
                                                       move = True)
                    elif ttc < secondTNVcc:
                        secondTNVcc = ttc
                    if slowttc < slowTNVcc or blocking: slowTNVcc = slowttc
                    if fastttc < fastTNVcc or blocking: fastTNVcc = fastttc
                    if tNVcc > 6:
                        dtac = xxt.distanceToCrash(crossing, altcrossing,False)
                        DTACcc = min(DTACcc, dtac)
        if tNVcc < tNV:
            secondTNV = tNV
            tNV = tNVcc
            closestCar = closestCarcc
            if tNVcc < 6:
                DTAC = DTACcc
        elif tNVcc < secondTNV:
            secondTNV = tNVcc
        if slowTNVcc < slowTNV: slowTNV = slowTNVcc
        if fastTNVcc < fastTNV: fastTNV = fastTNVcc
        if tNV > 6:
            DTAC = min(DTAC, DTACcc)## change ends here
                        
    carUpdate['Time to Crossing Car'] = tNV
    carUpdate['Crossing Car'] = closestCar
    carUpdate['Time to 2nd Crossing Car'] = secondTNV
    carUpdate['Slow Time Crossing'] = slowTNV
    carUpdate['Fast Time Crossing'] = fastTNV
    
    # locate vehicles ahead of this one
    aheadLanes = []
    #futureAheadLane = None # without discounted reward this has no effect
    for joint in intersectionLookUp:
        if joint[0] == car['lane']:
            aheadLanes += [joint[2]]
            #if joint[1] == car['dest']: # include ahead lane only if it is known dest.
            #    futureAheadLane = joint[1]
        if joint[2] == car['lane']:
            aheadLanes = [joint[1]]
    aheadVehicles = (cars['status']==0) & (cars['lane']==car['lane'])
    aheadVehicles = aheadVehicles & (cars.index != carID) # ensure this vehicle is not included
    for aheadLane in aheadLanes:    
        aheadVehicles = aheadVehicles | (
                (cars['status']==0) & (cars['lane']==aheadLane) )
    aheadVehicles = np.where(aheadVehicles)[0]
    # locate closest and 2nd closest vehicles
    closestPos = 1000
    closestCar = None
    secondClosestPos = 1000
    secondClosestCar = None
    for j in aheadVehicles:
        aheadCar = cars.iloc[j]
        if aheadCar['lane'] == car['lane']:
            position = aheadCar['lanepos'] - car['lanepos']
            if position < 0:
                position = 1001
        else: # in lane ahead
            position = aheadCar['lanepos'] - car['lanepos'] +\
                           getRoadLength(car['lane'])
        # for lane further ahead:
        #position = aheadCar['lanepos'] - car['lanepos'] +\
        #           getRoadLength(car['lane']) + getRoadLength(aheadLanes[0])
        if position < closestPos:
            secondClosestPos = closestPos
            secondClosestCar = closestCar
            closestPos = position
            closestCar = j
        elif position < secondClosestPos:
            secondClosestPos = position
            secondClosestCar = j
    closestPos = closestPos - VEHsize[0]
    secondClosestPos = secondClosestPos - VEHsize[0]
    # gather info on ahead vehicles
    carUpdate['Next to Intersection'] = False
    carUpdate['2nd Next to Intersection'] = False    
    carUpdate['Ahead Car'] = -1
    carUpdate['Speed of Ahead Car'] = -1
    carUpdate['Distance to Ahead Car'] = 1000
    carUpdate['Time to Ahead Car'] = 100
    carUpdate['Speed of 2nd Ahead Car'] = -1
    carUpdate['Distance to 2nd Ahead Car'] = -1
    carUpdate['Time to 2nd Ahead Car'] = 100
    carUpdate['Slow Time Ahead'] = 100
    carUpdate['Fast Time Ahead'] = 100
    if closestCar is None:
        if carUpdate['Car Location'] == 0:
            carUpdate['Next to Intersection'] = True        
    else:
        closestCarSpeed = cars.iloc[closestCar].speed
        carUpdate['Ahead Car'] = closestCar        
        carUpdate['Speed of Ahead Car'] = closestCarSpeed*MS2KPH
        carUpdate['Distance to Ahead Car'] = closestPos
        carUpdate['Time to Ahead Car'] = xxt.timeForDist(closestPos,
                                            car['speed'] - closestCarSpeed)
        carUpdate['Slow Time Ahead'] = xxt.timeForDist(closestPos,
                        car['speed'] - closestCarSpeed - ttcCheck_speedMargins)
        carUpdate['Fast Time Ahead'] = xxt.timeForDist(closestPos,
                        car['speed'] - closestCarSpeed + ttcCheck_speedMargins)
        if carUpdate['Car Location'] == 0:
            if splitRoad(cars.iloc[closestCar].lane)[1] != 'i':
                cars['Next to Intersection'] = True
        if secondClosestCar is None:
            carUpdate['Speed of 2nd Ahead Car'] = -1
            carUpdate['Distance to 2nd Ahead Car'] = -1
            if (carUpdate['Car Location'] == 0 and 
                carUpdate['Next to Intersection'] == 0):
                carUpdate['2nd Next to Intersection'] = True
        else:
            secondCCSpeed = cars.iloc[secondClosestCar].speed
            carUpdate['Speed of 2nd Ahead Car'] = secondCCSpeed * MS2KPH
            carUpdate['Distance to 2nd Ahead Car'] = secondClosestPos
            carUpdate['Time to 2nd Ahead Car'] = xxt.timeForDist(secondClosestPos,
                                            car['speed'] - secondCCSpeed)

    # locate behind vehicles
    behindLanes = []
    #pastBehindLane = None # without discounted reward this has no effect
    for joint in intersectionLookUp:
        if joint[1] == car['lane']:
            behindLanes += [joint[2]]
        if joint[2] == car['lane']:
            behindLanes = [joint[0]]
    behindVehicles = (cars['status']==0) & (cars['lane']==car['lane'])
    behindVehicles = behindVehicles & (cars.index != carID) # ensure this vehicle is not included
    if len(behindLanes) >= 1:
        behindVehicles = behindVehicles | (
                (cars['status']==0) & (cars['lane']==behindLanes[0]) )
    if len(behindLanes) == 2:
        behindVehicles = behindVehicles | (
                (cars['status']==0) & (cars['lane']==behindLanes[1]) )
    behindVehicles = np.where(behindVehicles)[0]
    # locate closest and 2nd closest vehicles
    closestPos = 1000
    closestCar = None
    secondClosestPos = 1000
    secondClosestCar = None
    for j in behindVehicles:
        behindCar = cars.iloc[j]
        if behindCar['lane'] == car['lane']:
            position = car['lanepos'] - behindCar['lanepos']
            if position < 0:
                position = 1001
        elif len(behindLanes) >= 1 and behindCar['lane'] == behindLanes[0]:
            position = car['lanepos'] - behindCar['lanepos'] +\
                       getRoadLength(behindLanes[0])
        elif len(behindLanes) == 2 and behindCar['lane'] == behindLanes[1]:
            position = car['lanepos'] - behindCar['lanepos'] +\
                       getRoadLength(behindLanes[1])
            #position = car['lanepos'] - behindCar['lanepos'] +\
            #           getRoadLength(behindLanes[0]) + getRoadLength(behindLanes[1])
        if position < closestPos:
            secondClosestPos = closestPos
            secondClosestCar = closestCar
            closestPos = position
            closestCar = j
        elif position < secondClosestPos:
            secondClosestPos = position
            secondClosestCar = j
    carUpdate['Behind Car'] = -1
    carUpdate['Time to Behind Car'] = 100
    carUpdate['Time to 2nd Behind Car'] = 100
    if closestCar is not None:
        carUpdate['Behind Car'] = closestCar        
        tNV =  xxt.timeForDist(closestPos - VEHsize[0] ,
                            cars.iloc[closestCar].speed - car['speed'])
        if tNV < 0: # never colliding
            tNV = 100
        carUpdate['Time to Behind Car'] = tNV
    if secondClosestCar is not None:      
        secondTNV = xxt.timeForDist(secondClosestPos - VEHsize[0],
                cars.iloc[secondClosestCar].speed - car['speed'])
        if secondTNV < 0: # never colliding
            secondTNV = 100
        carUpdate['Time to 2nd Behind Car'] = secondTNV
        
    carUpdate['On Grid'] = True
    carUpdate['TTNV Slower'] = -1
    carUpdate['TTNV Faster'] = -1
    carUpdate['Distance to Crossing Car'] = DTAC
    
    carUpdate['Intersection Open'] = 1
    carUpdate['Intersection Fill Car'] = -1
    if grade != 2:
        nconflicts = 0
        conflictCar = -1
        for altcar in np.where(cars['status']==0)[0]:
            aroute,agrade,aln = splitRoad(cars.loc[altcar,'lane'])
            if agrade==':' or (agrade=='i' and cars.loc[altcar,'lanepos']>=89):
                if (aroute<3) != (route<3):
                    nconflicts += 1
                    conflictCar = altcar
        if nconflicts >= 1:
            carUpdate['Intersection Open'] = 0
        if nconflicts == 1:
            carUpdate['Intersection Fill Car'] = conflictCar
    carUpdate['Priority'] = int(car['priority'])
    return carUpdate


## main code

iteration = 1
err = 0

## run iteration
while iteration <= numiter and err == 0:    
    starttime = time.time()
    print 'iteration '+ str(iteration)
    # set up parameters and vehicles
    cars = initialize(iteration)
    #cars1 = cars
    agents = {}
    for carID in range(cars.shape[0]):
        car = cars.iloc[carID]
        agents[carID] = Agent(carID, car['lane'],car['dest'],car['speed'],car['time'])
    
    ncars = cars.shape[0]
    cars['status'] = [-1]*ncars #-1 not created, 0 created, 1 exited, 2 crashed
    cars['lanepos'] = [-1]*ncars # not given to Agent but needed for calculation
    cars['x'] = [-1]*ncars
    cars['y'] = [-1]*ncars
    cars['angle'] = [-1]*ncars
    cars['ilane'] = ['']*ncars
    cars['caratfault'] = [0]*ncars
    cars['lanechanged'] = [0]*ncars
    cars['previousTurn'] = [-1]*ncars
    cars['zeroAction'] = [0]*ncars
    cars['priority'] = [-1]*ncars
    #cars['indexS'] = [-1]*ncars # previous state index in Qtable
    #cars['indexA'] = [-1]*ncars # previous action index in Qtable

    collisionCount = 0
    zeroCollisionCount = 0
    TotReward = 0
    WrongPath = 0
    CarTime = 0
    ttime = 0
    maxTime = 100 # seconds
       
    # start simulator
    Sim = Simulator('Qsim', gui = True)    

    ## run simulation
    while ttime < maxTime and err == 0 and np.any(cars['status']<=0):
        
        priorityRoadTimer = int(ttime // 2.)%12
        priorityRoad1_2 = priorityRoadTimer < 5
        priorityRoad3_4 = priorityRoadTimer >= 6 and priorityRoadTimer < 11
        print "timer "+str(priorityRoadTimer)
        
        # create new vehicles
        for carID in np.where(cars['status']<0)[0]:
            car = cars.iloc[carID]
            if car['time'] <= ttime:
                # search through lane and see if another vehicle is taking the spot
                otherCars = (cars['status'] == 0) | (cars['status'] >= 2)
                addThisCar = True
                for otherID in np.where(otherCars)[0]:
                    if cars['lane'].iloc[otherID] == car['lane']:
                        if cars['lanepos'].iloc[otherID] <= 10.:
                            addThisCar = False
                if addThisCar:
                    err += Sim.createVehicle(str(carID), car['lane'], 0.)
                    cars.loc[carID,'status'] = 0
                    if err > 0:
                        break

                #err += Sim.moveVehicleAlong(str(carID), .01)
        
        # gather vehicle info
        for carID in np.where(cars['status']==0)[0]:        
            carLane,carLanePos,carPos,carAngle = Sim.getVehicleState(str(carID))
            carAngle = carAngle # fix on 4/23/16
            
            if carLane == '':
                print '!!!! lane is empty'+ str(carID) + cars.loc[carID,'lane']
                carLane = cars.loc[carID,'lane']
                carLanePos = 0.0
                carPos = (-6.*carID,-6.*carID)
                carAngle = 0.
            elif carLane == ":0_16_0":
                carLane = ":0_3_0"
            elif carLane == ":0_17_0":
                carLane = ":0_11_0"
            elif carLane is None:
                print '!!!! lane is none '+ str(carID)
                cars.loc[carID,'status']=1
                continue
            
            #if carLane[1] == 'o':
            #    cars.loc[carID,'status'] = 1
            cars.loc[carID,'lane'] = carLane
            cars.loc[carID,'lanepos'] = carLanePos
            cars.loc[carID,'x'] = carPos[0]
            cars.loc[carID,'y'] = carPos[1]
            cars.loc[carID,'angle'] = carAngle
            if carLane[1]=='o' and carLanePos < 5.: # SUMO sends crazy angles here
                if carLane[0]=='1':
                    cars.loc[carID,'angle'] = -1.5708
                if carLane[0]=='2':
                    cars.loc[carID,'angle'] = 1.5708
                if carLane[0]=='3':
                    cars.loc[carID,'angle'] = 0.
                if carLane[0]=='4':
                    cars.loc[carID,'angle'] = 3.14159
            
            # finding the intersection lane at which to look for collisions
            # slightly tricky because we only want to pick one lane
            carroute,cargrade,carln = splitRoad(carLane)
            if cargrade == 'i':
                if carln == 0: # right and straight are both options
                    if cars.loc[carID,'previousTurn'] == 1:
                        currentDestination = roadOrder.loc[carroute,'right']
                    elif cars.loc[carID,'previousTurn'] == 0:
                        currentDestination = roadOrder.loc[carroute,'straight']
                    else: # no decision yet, use desired destination
                        currentDestination,aa,bb = splitRoad(cars.loc[carID,'dest'])
                        if currentDestination == roadOrder.loc[carroute,'left']:
                            # all else fails, just check straight ahead
                            currentDestination = roadOrder.loc[carroute,'straight']
                    cars.loc[carID,'ilane'] =next(k for i,j,k in
                                        intersectionLookUp if i==carLane and
                                        int(j[0])==currentDestination)
                else:
                    cars.loc[carID,'ilane'] =next(k for i,j,k in
                                            intersectionLookUp if i==carLane)
            elif cargrade==':':
                cars.loc[carID,'ilane'] = carLane
                
            # brake for light            
            brakeEl1 = cargrade=='i'
            brakeEl2 = roadOrder.loc[carroute,'right']!=cars.loc[carID,'dest']
            brakeEl3 = (carroute<3) and not priorityRoad1_2
            brakeEl4 = (carroute>2) and not priorityRoad3_4
            carSpeedInUnit = cars.loc[carID,'speed']*MS2KPH/10.
            hardBrakeDist = carSpeedInUnit**2. /MS2KPH*DELTAT*5
            softBrakeDist = carSpeedInUnit**2. /MS2KPH*DELTAT*10
            brakeEl5 = carLanePos + hardBrakeDist < 92.
            brakeEl6 = carLanePos + softBrakeDist > 90.
            #brakeEl7 = carLanePos >= 90. and carLanePos <= 92.
            pbrake = brakeEl1 and brakeEl2 and (brakeEl3 or brakeEl4) and\
                     brakeEl5 and brakeEl6
            cars.loc[carID,'priority'] = not pbrake
            if pbrake: print "car "+str(carID)+" brake"
                
        
        # cars that collided last time are now completely removed
        carsThatCollidedLastStep = np.where(cars['status']>=2)[0]
        for carID in carsThatCollidedLastStep:
            
            cars.loc[carID, 'status'] = 1
            Sim.removeVehicle(str(carID))
        
        # check for collisions
        activeCars = np.where(cars['status']==0)[0]
        for carNum in range(len(activeCars)):
            carID = activeCars[carNum]
            car = cars.iloc[carID]
            for altNum in range(carNum):
                altID = activeCars[altNum]
                alt = cars.iloc[altID]
                carObject = pd.Series([car['x'],car['y'],car['angle'],
                                       VEHsize[0],VEHsize[1]],
                                       index=collisionCheck.collisionVars)
                altObject = pd.Series([alt['x'],alt['y'],alt['angle'],
                                       VEHsize[0],VEHsize[1]],
                                       index=collisionCheck.collisionVars)
                if collisionCheck.check(carObject, altObject):
                    cars.loc[carID, 'status'] = 2 + altID
                    cars.loc[altID, 'status'] = 2 + carID
                    collisionCount += 1
                    zeroCollisionCount += car['zeroAction']+alt['zeroAction']
                    carAtFault = findFault(cars.iloc[carID],cars.iloc[altID])
                    # carAtFault = 0 -> carID at fault
                    # carAtFault = 1 -> altID at fault
                    if carAtFault == 0:
                        cars.loc[carID, 'caratfault'] = 1
                    elif carAtFault ==1  :
                        cars.loc[altID, 'caratfault'] = 1
                    elif carAtFault == 2:
                        cars.loc[carID, 'caratfault'] = 1
                        cars.loc[altID, 'caratfault'] = 1
                        collisionCount += 1
                    #Sim.addCrashSymbol(car, alt, collisionCount/2)
                    #time.sleep(2.) # just to show that cars collided
        
        # gather when the vehicles reach potential collision points
        trajectories = {}
        for carID in np.where((cars['status']==0)|(cars['status']>=2))[0]:
            car = cars.iloc[carID]
            route,grade,ln = splitRoad(car['lane'])
            
            if grade=='i': # not in intersection yet
                initDist = car['lanepos'] - getRoadLength(car['lane'])
            elif grade=='o': # left intersection (but maybe tail end is still there)
                initDist = car['lanepos'] + getRoadLength(car['ilane'])            
            else:
                initDist = car['lanepos']
            
            trajectory = []
            for crossIndex in range(intersectionInfo.shape[0]):
                thisCross = intersectionInfo.iloc[crossIndex]
                if thisCross['lane'] == car['ilane']:
                    trajectory += xxt.gatherTrajectory(car['speed'], initDist,
                                             thisCross, ttcCheck_speedMargins)
            trajectories[carID] = trajectory
         
         
        carsToUpdate = np.where((cars['status']==0) | (cars['status']>=2))[0]        
         
        # update QTable
        Qstates = {}
        prevIndices = {}
        for carID in carsToUpdate:
            carUpdate = gatherStateInfo(cars, trajectories, carID)
            Qstates[carID] = carUpdate
            prevIndices[carID] = (int(agents[carID].state), int(agents[carID].action))
        # first gather normal reward
        #realReward = sum(reward(formatState(state)) for
        #                            state in Qstates.itervalues())
            
        realReward = 0
        for carID in carsToUpdate:
            agents[carID].state_space(*formatState(Qstates[carID]))
            realReward += agents[carID].reward(ttime)
            TotReward += realReward
        #print "all cars reward "+str(realReward)
        differenceRewards = [0.]*ncars
        # rewards without vehicle
        for carID in carsToUpdate:
            differenceReward = 0
            for otherCar in carsToUpdate:
                if otherCar != carID:
                    agents[otherCar].state_space(*formatState(
                                    Qstates[otherCar], AgentToRemove=carID))
                    differenceReward += agents[otherCar].reward_def(cars.loc[otherCar,'caratfault'] )
            differenceRewards[carID] = realReward - differenceReward
        for carID in carsToUpdate: # have to redo this to reset all state variables
            finalformat = formatState(Qstates[carID])
            agents[carID].state_space(*finalformat)
            printString = "car "+str(carID)
            #printString += " NTI "+str(finalformat.iloc[7])
            printString += " cross "+str(Qstates[carID]['Time to Crossing Car'])
            printString += " ahead "+str(Qstates[carID]['Time to Ahead Car'])
            #printString += " TTNV "+str(finalformat.iloc[7])
            printString += " Speed "+str(finalformat.iloc[4])
            #printString += " behind "+str(Qstates[carID]['Time to Behind Car'])
            #printString += " correctLane "+str(finalformat.iloc[3])
            #printString += " correctLane "+str(formatState(Qstates[carID]).iloc[3])
            printString += " D 2 Ahead "+str(Qstates[carID]['Distance to Ahead Car'])
            printString += " D 2 Cross "+str(Qstates[carID]['Distance to Crossing Car'])
            printString += " slowTTNV "+str(finalformat['TTNV Slower'])
            printString += " fastTTNV "+str(finalformat['TTNV Faster'])
            print printString
            
            printString = "car "+str(carID) + ": reward "
            printString += str(differenceRewards[carID])
            if trainingTable:
#                printString += updateQvalue_simple(QTable, prevIndices[carID], 
#                                    differenceRewards[carID])
                printString += updateQvalue_2step(QTable,Qcount, prevIndices[carID], 
                                    differenceRewards[carID],
                                    agents[carID].state, finalformat['Collision'])
            #print printString
                       
        ## make actions
        carsThatAct = cars['status'] == 0
        for carID in np.where(carsThatAct)[0]:
            car = cars.iloc[carID]
            route,grade,ln = splitRoad(car['lane'])
            # get action from agent
            laneChange, speedChange, turn, zeroAction = agents[carID
                                ].Action_training(QTable, trainingTable, None)
            cars.loc[carID,'zeroAction'] = zeroAction
            if zeroAction:
                print "car "+str(carID)+"has action 0!!!!!!!!!!"
            suppressLaneChange = (grade == 'o' and car['lanepos']<=5.) or (grade == 'i' and car['lanepos'] >= 89.)
            #suppressLaneChange = (grade == 'i' and car['lanepos'] >= 89.)
            if suppressLaneChange:
                laneChange = 0;
            suppressSpeedChange = False#grade == 'o' and (route == 4) or (grade == ':' and route==1 and turn ==2 and carID==2)
            if suppressSpeedChange:
                speedChange = 2;
            printString = "car action "+str(carID)+": lane "+str(int(laneChange))
            printString += " speed "+str(int(speedChange))+" turn "+str(turn)+"\n"
            #print printString
            #cars['indexA'][carID] = indexA
            
            if laneChange==1 and (grade =='i' or grade=='o'):
                newlane = makeRoad(route,grade,0)
                err += Sim.moveVehicle(str(carID), newlane, car['lanepos'])
                cars.loc[carID,'lane'] = newlane
                if err > 0:
                    break
            if laneChange==2 and (grade =='i' or grade=='o'):
                newlane = makeRoad(route,grade,1)
                err += Sim.moveVehicle(str(carID), newlane, car['lanepos'])
                cars.loc[carID,'lane'] = newlane
                if err > 0:
                    break
            
            if laneChange == 1 and ln == 1:
                
                cars.loc[carID,'lanechanged'] = 1
            elif laneChange == 2 and ln == 0:
                
                cars.loc[carID,'lanechanged'] = 2
            else:
                cars.loc[carID,'lanechanged']=0
                
            speedChange = [0,-10.,10.][int(speedChange)]/MS2KPH            
            cars.loc[carID,'speed'] = min(max(car['speed']+speedChange, 0.),60./MS2KPH)
            distance = cars.loc[carID,'speed']*DELTAT
            if (grade=='i' and car['lanepos'] > 89.):
                cars.loc[carID,'previousTurn'] = turn
            
            # the vehicle is about to make a transition to a new road
            if getRoadLength(car['lane']) - car['lanepos'] <= distance:
                
                if grade=='i': # entering intersection
                    if ln==0 and turn < 2: # the only case where the turn command matters
                        rightTurnRoute = roadOrder.loc[route].right
                        rightTurnLane = next((j for i,j,k in intersectionLookUp
                                                if i==car['lane'] and
                                                    int(j[0])==rightTurnRoute))
                        #currentlyTurningRight = splitRoad(car['dest'])[0] == rightTurnRoute
                        straightRoute = roadOrder.loc[route].straight
                        straightLane = next((j for i,j,k in intersectionLookUp
                                                if i==car['lane'] and
                                                    int(j[0])==straightRoute))
                        #currentlyStraight = splitRoad(car['dest'])[0] == straightRoute
                        
                        if turn == 0: # go straight
                            err += Sim.moveVehicleAlong(str(carID), 0.0, straightLane)
                        else: # turn right  ## also turning right if something wants to go left
                            err += Sim.moveVehicleAlong(str(carID), 0.0, rightTurnLane)
                    elif ln==1:
                        leftTurnRoute = roadOrder.loc[route].left
                        leftTurnLane = next((j for i,j,k in intersectionLookUp
                            if i==car['lane'] and int(j[0])==leftTurnRoute))
                        err += Sim.moveVehicleAlong(str(carID), 0.0, leftTurnLane)
                    err += Sim.moveVehicleAlong(str(carID), distance, ':')
                    #cars.loc[carID,'priority'] = int((route<3) == priorityRoads)
                
                elif grade=='o': # leaving simulation
                    Sim.removeVehicle(str(carID))
                    cars.loc[carID,'status'] = 1
                    WrongPath += int(route != splitRoad(car['dest'])[0])
                    CarTime += ttime - car['time']
                else: # going from intersection to exit lane
#                    newLane = next((j for i,j,k in intersectionLookUp if
#                                        k==car['lane']))
#                    newPos = distance - Sim.getLaneInfo(car['lane'])[0] + car['lanepos']
#                    err += Sim.moveVehicle(str(carID), newLane, newPos) 
                    err += Sim.moveVehicleAlong(str(carID), distance, ':')
                    
            elif (getRoadLength(car['lane']) - car['lanepos'] <= distance+5.
                   ) and grade=='o':
                # may still be leaving simulation
                Sim.removeVehicle(str(carID))
                cars.loc[carID,'status'] = 1
                WrongPath += int(route != splitRoad(car['dest'])[0])
                CarTime += ttime - car['time']
            elif distance > 0:
                if grade=='i' or (grade=='o' and car['lanepos']>5.):
                    err += Sim.moveVehicleAlong(str(carID), distance, car['lane'])
                else:
                    err += Sim.moveVehicleAlong(str(carID), distance, ':')
            if err > 0:
                break
        ttime += 0.1
            
    # end of iteration
    print "collisions "+str(collisionCount)
    print "wrong "+str(WrongPath)
    Stats_train=Stats_train+[collisionCount,zeroCollisionCount,TotReward,
                             CarTime,WrongPath,float(cars.shape[0])]
    
    if err == 0:
        # finish step
        Sim.end()        
        print "iteration "+str(iteration)+" : "+str(time.time()-starttime)
        time.sleep(.2)
        iteration += 1
    else:
        print "iteration "+str(iteration)+" failed"
        iteration += 1

# save info afterwards
iteration = iteration - 1
if trainingTable:
    np.save('qtable_4way_l.npy',QTable)
    np.save('qtable_4way_l_count.npy',Qcount)
Stats_train.loc['Collision'] = Stats_train.loc['Collision'] / Stats_train['Cars']
Stats_train.loc['Untrained Collision'] =(Stats_train.loc['Untrained Collision'] 
                                        / Stats_train['Cars'])
Stats_train.loc['Reward'] = Stats_train.loc['Reward'] / Stats_train['Cars']
Stats_train.loc['WrongPath'] = Stats_train.loc['WrongPath'] / Stats_train['Cars']
Stats_train.loc['Time'] = Stats_train.loc['Time'] / Stats_train['Cars']
statsSaveName = 'leftturn_train_'+str(iteration)+'.csv'
Stats_train.to_csv(statsSaveName)
