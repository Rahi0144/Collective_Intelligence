# -*- coding: utf-8 -*-
"""
rectangle (simple) method of checking for intersection collisions
with speed margins included
5/13/16
"""
import pandas as pd

def getInterInfo(simname):
    return pd.read_csv('collisionsFile_'+simname+'_rect.csv',header=0)

def formatTrajectories(ttc, slowerttc, fasterttc):
#    fastOrSlow = 1 # 0 slow, 1 normal, 2 fast
#    if slowerttc < ttc - 2:
#        ttc = slowerttc
#        fastOrSlow = 0
#    if fasterttc < ttc - 2:
#        ttc = fasterttc
#        fastOrSlow = 2
#    return(ttc, fastOrSlow)
    return (ttc, slowerttc, fasterttc)

def timeForDist(dist, speed):
    if dist <= 0: return 0
    if speed <= 0: return 100
    return dist / speed

def gatherTrajectory(carspeed, initDist, crossing, speedMargin = 0.):
    if crossing['end_lp'] < initDist:
        return []        
    return [[crossing['lane2'], carspeed, (crossing['begin_lp'] - initDist),
                                (crossing['end_lp'] - initDist)]]
    
    
def checkTrajectories(crossing, altcrossing, speedMargin = 0.):
    ttc = 100
    speed, beginDist, endDist = crossing[1:]
    beginTime = timeForDist(beginDist, speed)
    endTime = timeForDist(endDist, speed)
    altspeed, altbeginDist, altendDist = altcrossing[1:]
    altbeginTime = timeForDist(altbeginDist, altspeed)
    altendTime = timeForDist(altendDist, altspeed)
    if endTime >= altbeginTime and altendTime >= beginTime:
        ttc = max(beginTime, altbeginTime)
    if speedMargin == 0:
        return formatTrajectories(ttc, 100, 100)
        
    slowerspeed = speed - speedMargin
    slowerBegin = timeForDist(beginDist, slowerspeed)
    slowerEnd = timeForDist(endDist, slowerspeed)
    fasterspeed = speed + speedMargin
    fasterBegin = timeForDist(beginDist, fasterspeed)
    fasterEnd = timeForDist(endDist, fasterspeed)
    slowerttc = 100
    fasterttc = 100
    if slowerEnd >= altbeginTime and altendTime >= slowerBegin:
        slowerttc = max(slowerBegin, altbeginTime)
    if fasterEnd >= altbeginTime and altendTime >= fasterBegin:
        fasterttc = max(fasterBegin, altbeginTime)
    return formatTrajectories(ttc , slowerttc , fasterttc)
    
def distanceToCrash(crossing, altcrossing, move=True):
    if move or (altcrossing[2] <= 0 and altcrossing[3] >= 0):
        return max(0., crossing[2])
    else:
        return 100