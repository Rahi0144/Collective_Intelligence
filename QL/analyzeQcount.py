# -*- coding: utf-8 -*-
"""
5/25/16
"""
import numpy as np
from matplotlib import pyplot

from Agent import agent_optionCounts, agent_totalNumStates, Agents
agent = Agents(0,0,0,0,0)
getInfo = agent.stateNum2Info
locationIndex = 6

indexer = list(np.prod(agent_optionCounts[:j]) for j in range(len(agent_optionCounts)))
locations = np.arange(580608)
locations = locations % indexer[locationIndex+1]
locations = locations // indexer[locationIndex]


Qcount = np.load('qtable_4way_count.npy')
#Qcount = np.load('qcount_2way_0206_3_count.npy')
#Qcount = np.load('qtable_4way_l_count.npy')

nupdates = float(np.sum(Qcount))

# analyze total counts
flattened = np.array([])
flattened = np.append(flattened, Qcount[(locations == 0)|(locations==2), :9])
flattened = np.append(flattened, Qcount[locations==1,9:18])
flattened = np.append(flattened, Qcount[locations==3,18:])

filled = float(np.sum(flattened > 0)) / float(len(flattened))
print "ratio of elements filled "+str(filled)
print "number of updates "+str(nupdates)
pyplot.hist(flattened, log=True)
pyplot.hist(flattened[flattened < 10], log=True)

ordered = np.sort(flattened)
float(np.sum(ordered[-10:]))/nupdates


# look at maximum value
np.amax(flattened)
maxstatenum = np.where(Qcount == np.amax(Qcount))[0]
maxstate = getInfo(maxstatenum)
Qcount[maxstatenum, :]

priorityswitch = maxstate
priorityswitch[12] = 1
priorityswitchnum = np.dot(indexer,priorityswitch)
Qcount[priorityswitchnum,:]



statecounts = np.sum(Qcount, axis=1)
ordered = np.sort(statecounts)
checkNum = 100
print "top "+str(checkNum)+" states have "+\
            str(float(np.sum(ordered[-100:]))/nupdates)+" of updates"
            
sum(statecounts > 0)