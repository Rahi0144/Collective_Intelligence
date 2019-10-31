# -*- coding: utf-8 -*-
"""
updating Qvalues based on new reward, and any other factors
returns a string that can be printed to give info about the updated values
5/24/16
"""
ratio = [.5, .5]

def updateQvalue_simple(Qtable,Qcount, index, reward):
    if index[0] >= 0:
        newval = ratio[0] * Qtable[index] + ratio[1] * reward
        Qtable[index] = newval
        return " new value "+str(newval)
    else:
        return " no update"

        
def updateQvalue_2step(Qtable,Qcount, index, reward, newState, collision):
    if index[0] >= 0:
        currentQvals = Qtable[newState, :]
        # find the right set of actions
        if collision or all(currentQvals==0):
            newval = 0.
        elif any(currentQvals[:9] < 0): # normal road
            newval = max(currentQvals[:9])
        elif any(currentQvals[9:18] < 0): # pre-intersection zone
            newval = max(currentQvals[9:18])
        else:
            newval = max(currentQvals[18:]) # in intersection
        ratio= 1/(1+Qcount[index])**.5
       # ratio= 1- ratio[0]
        newval = (1-ratio) * Qtable[index] + ratio * (reward + newval)
        Qtable[index] = newval
        Qcount[index] +=1
        return " new value "+str(newval)
    else:
        return " no update"


def updateQvalue_2stepAfter(Qtable, index, reward, prevIndex):
    if index[0] >= 0:
        newval = .5 * Qtable[index] + .5 * reward
        Qtable[index] = newval
        if prevIndex[0] >= 0:
            previousVal = Qtable[prevIndex] + .5*newval
            Qtable[prevIndex] = previousVal
            return " new value "+str(newval)+", last step value "+str(previousVal)
        else:
            return " new value "+str(newval)
    else:
        return " no update"
            
### things to add:
# discounting reward by number of states