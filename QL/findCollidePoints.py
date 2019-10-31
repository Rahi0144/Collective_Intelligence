# -*- coding: utf-8 -*-
from sumoMethodsWindows import Sumo
import pandas as pd
#import numpy as np
import collisionCheck
import time
import collisionHull

VEHsize = (5.1,2.1) # meters, length by width
configuration = 'Qsim'

#intersectionLookUp = [['1i_0','3o_0',':0_12_0'] , ['1i_0','2o_0',':0_13_0'] ,
#                      ['1i_0','2o_1',':0_13_1'] , ['1i_1','4o_1',':0_15_0'] ,
#                      ['2i_0','4o_0',':0_4_0'] , ['2i_0','1o_0',':0_5_0'] ,
#                      ['2i_0','1o_1',':0_5_1'] , ['2i_1','3o_1',':0_7_0'] ,
#                      ['3i_0','2o_0',':0_8_0'] , ['3i_0','4o_0',':0_9_0'] ,
#                      ['3i_0','4o_1',':0_9_1'] , ['3i_1','1o_1',':0_11_0'] ,
#                      ['4i_0','1o_0',':0_0_0'] , ['4i_0','3o_0',':0_1_0'] ,
#                      ['4i_0','3o_1',':0_1_1'] , ['4i_1','2o_1',':0_3_0'] ]
intersectionLookUp = [['1i_0','3o_0',':0_12_0'] , ['1i_0','2o_0',':0_13_0'] ,
                      ['1i_1','4o_1',':0_15_0'] ,
                      ['2i_0','4o_0',':0_4_0'] , ['2i_0','1o_0',':0_5_0'] ,
                      ['2i_1','3o_1',':0_7_0'] ,
                      ['3i_0','2o_0',':0_8_0'] , ['3i_0','4o_0',':0_9_0'] ,
                      ['3i_1','1o_1',':0_11_0'] ,
                      ['4i_0','1o_0',':0_0_0'] , ['4i_0','3o_0',':0_1_0'] ,
                      ['4i_1','2o_1',':0_3_0'] ]
roadOrder = pd.DataFrame({'left':[4,3,1,2],'right':[3,4,2,1],
                          'straight':[2,1,4,3]},index=[1,2,3,4])
def getRoadLength(lane):
    if lane[0]==':':
        inroad,outroad = next(((i,j) for i,j,k in intersectionLookUp if k==lane))
        roadDirections = roadOrder.loc[int(inroad[0])]
        outroute = int(outroad[0])
        if roadDirections['left'] == outroute:
            return 15.64
        elif roadDirections['right'] == outroute:
            return 5.
        else:
            return 16.32
    return 91.95
    
class WriteFrame: # makes repeatedly adding to data frame easier
    def __init__(self, colnames=None):
        self.colnames = colnames
        self.restart=True
    def add(self, newrows):
        if self.restart:
            self.df = pd.DataFrame(newrows)
            self.restart=False
        else:
            self.df = self.df.append(newrows, ignore_index = True)
    def out(self):
        if self.colnames is not None:
            self.df.columns =  self.colnames
        #self.df.to_csv(fileName, sep=',', header=True, index=False)
        self.restart=True
        return self.df
    
err = 0
Sim = Sumo(configuration,gui=True)

# first gather the possible positions/angles of each intersection lane
allcars = {}
for start, end, road in intersectionLookUp:
    print "at "+road
    err += Sim.createVehicle(road, start, 91.)
    err += Sim.moveVehicleAlong(road, 0.1, end)
    thisdf = WriteFrame(['x','y','angle','length','width','lp'])    
    
    dist = 0.1
    status = -1
    totaldist = -50.
    while status < 1 and totaldist < 50:
        lane,lanepos,pos,angle = Sim.getVehicleState(road)
        status = 1*(lane[1]=='o' and lanepos > 5.) - 1*(lane[1]=='i')
        if lane[1]=='o' and lanepos < 5.: # must be included since back of vehicle
                if lane[0]=='1':         # is still in intersection
                    angle = -1.5708
                if lane[0]=='2':
                    angle = 1.5708
                if lane[0]=='3':
                    angle = 0.
                if lane[0]=='4':
                    angle = 3.1416
                lanepos = lanepos + getRoadLength(road)
        elif lane[0]==':' and totaldist < -10.:
            totaldist = lanepos
        if status==0:  #add vehicle info to the dataframe
            thisdf.add([[pos[0],pos[1],angle,VEHsize[0],VEHsize[1],totaldist]])
        Sim.moveVehicleAlong(road, dist, ':')
        totaldist += dist
        time.sleep(.05) # might guard against freeze
        
    allcars[road] = thisdf.out()
    Sim.removeVehicle(road)
Sim.end()

print "--- rect ---"
# now use dataframes of position+angle to find collision points
collisions = WriteFrame(['lane','lane2','begin_lp','end_lp'])
for an in range(len(intersectionLookUp)):
    for bn in range(an):
        print "checking "+str(an)+", "+str(bn)
        starta,enda,roada = intersectionLookUp[an]
        startb,endb,roadb = intersectionLookUp[bn]
        cara = allcars[roada]
        carb = allcars[roadb]
        
        abegin = 50
        aend = -1
        bbegin = 50
        bend = -1
        for acount in range(cara.shape[0]):
            apos = cara['lp'].iloc[acount]
            for bcount in range(carb.shape[0]):
                bpos = carb['lp'].iloc[bcount]
                if collisionCheck.check(cara.iloc[acount], carb.iloc[bcount]):
                    if abegin > apos:
                        abegin = apos
                    if bbegin > bpos:
                        bbegin = bpos
                    if aend < apos:
                        aend = apos
                    if bend < bpos:
                        bend = bpos
        if aend > 0 and not starta==startb: # ignore cars from same lane
            collisions.add([[roada,roadb,abegin,aend],[roadb,roada,bbegin,bend]])        
collisions.out().to_csv('collisionsFile_sumo_rect.csv', sep=",",
                header=True, index=False)
                
print "--- quad ---"
collisions = WriteFrame(['lane','lane2','p1','p2','p3','p4'])
for an in range(len(intersectionLookUp)):
    for bn in range(an):
        print "checking "+str(an)+", "+str(bn)
        starta,enda,roada = intersectionLookUp[an]
        startb,endb,roadb = intersectionLookUp[bn]
        cara = allcars[roada]
        carb = allcars[roadb]
        
        points = []
        for acount in range(cara.shape[0]):
            for bcount in range(carb.shape[0]):
                apos = cara['lp'].iloc[acount]
                bpos = carb['lp'].iloc[bcount]
                if collisionCheck.check(cara.iloc[acount], carb.iloc[bcount]):
                    points += [(apos, bpos)]
        if len(points) > 0:        
            points = collisionHull.getHull(points)
            collisions.add([[roada,roadb, points[0][0],points[1][0],
                             points[2][0],points[3][0]], [roadb,roada,
                             points[3][1],points[2][1],points[1][1],
                             points[0][1]]])
collisions.out().to_csv('collisionsFile_sumo_quad.csv',sep=",",
                            header=True, index=False)