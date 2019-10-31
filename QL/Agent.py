# -*- coding: utf-8 -*-

import numpy as np
agent_optionCounts = [4, 7, 3, 2, 3, 2, 4, 3, 2, 2, 3, 2, 2]
agent_totalNumStates = np.prod(agent_optionCounts)

class Agents(object):
    # total number of the cars on grid is drawn from a poisson distribution 
    # each car aslo assigned some initial speed which is drwan from a Gaussian distribution 
    # if 2 cars to start in the same lane, they should start with MinTimeInterval time differnce 
    def __init__(self, agent_id, source_lane, destination_lane, InitialSpeed,  MinTimeInterval):
        self.name = agent_id
        self.source = source_lane
        self.destination = destination_lane
        self.initial_speed = InitialSpeed
        self.time_interval = MinTimeInterval
        self.ASBR = 0.
        self.Rtime = 0
        self.Rdest = 0
        self.RTTC = 0
        self.RDTA = 0
        self.Rcoll = 0
        self.ETI = 0
        self.state = -1
        self.action = -1
        self.NLG = 0
        self.RLG = 0
        self.LaneChange =0
        self.PreviousStepInZone1 = False
        self.indexer = list(np.prod(agent_optionCounts[:j]) for j in range(len(agent_optionCounts)))
        
    # Agent Location={Pre-intersection, logical-intersection, Post-intersection, at-intersection }={0,1,2,3}
    # Agent Road = {left, right, down, up}={0,1,2,3}
    # Agent Lane=(in this case){right,left }= {0,1}
    # Destination Road = {left, right, down, up}={0,1,2,3}
    # Destination Lane=(in this case){right,left }= {0,1}    
    # Turn Direction = {left, stright,right} = {0,1,2}
    # Adjacent Lane is free ={none, left only, right only, both}={0,1,2,3}
    # Correct Lane = {no, yes} ={0,1}
    # Agent On  Grid = { yes,no} ={1,0}
    # speed = {0,10,20,30,40,50,60}={0,1,2,3,4,5,6}
    # collision ={no,yes}={0,1}
    # relative distance speed = {-3a,-2a,-a,0,a,2a,3a}={0,1,2,3,4,5,6,7}
    # DistancetoToCarOfInterest ={0,2,4,8,16,32,64,128,256 and more}
    # TNVoverestimation = {slower version collides, actual collision, faster} = {0,1,2}
    def state_space(self, AgentLocation, AgentLane, TurnDirection,
                    CorrectLane, Speed, Collision, Adj_LaneFree, TimeToNearestVehicle,
                    NextToIntersection,DistanceToAheadCar, AgentonGrid,
                    SlowerTNV, FasterTNV, DistanceToCrossCar, IntersectionOpen, Priority):
        
        self.Location = AgentLocation
        #self.Road = AgentRoad
        self.Lane = AgentLane
        #self.DRoad = DestinationRoad
        #self.DLane = DestinationLane
        self.Turn = TurnDirection
        self.RLane = CorrectLane
        self.collision = Collision
        self.AdjFreeLane = Adj_LaneFree
        self.OnGrid = AgentonGrid
        self.speed = Speed
        self.TTNV = TimeToNearestVehicle
        self.NTI = NextToIntersection
        self.DTAC = DistanceToAheadCar
        self.SlowerTNV = SlowerTNV
        self.FasterTNV = FasterTNV
        self.DistanceToCrossCar = DistanceToCrossCar
        self.NotTrainedFlag = False
        self.IntersectionOpen = IntersectionOpen
        self.Priority = Priority
        #self.SOAC = SpeedofAheadCar
       # self.DTAC = DistanceToAheadCar
#        self.relative_distance_speed = RelativeDistanceSpeed
#        self.distance_to_car_of_interest =  DistancetoToCarOfInterest
        allStateVariables = [self.TTNV, self.speed, self.AdjFreeLane,
                             self.RLane, self.Turn, self.Lane,
                             self.Location, self.DTAC, self.SlowerTNV,
                             self.FasterTNV, self.DistanceToCrossCar, 
                             self.IntersectionOpen, self.Priority]
        
        self.state = np.dot(self.indexer, allStateVariables)
        #self.state= self.TTNV + 4* self.speed + 4*7*self.OnGrid + 4*7*2*self.AdjFreeLane + 4*7*2*3*self.collision +\
        #            4*7*2*3*2*self.RLane + 4*7*2*3*2*2*self.Turn + 4*7*2*3*2*2*3*self.Lane+ 4*7*2*3*2*2*3*2*self.Location + 4*7*2*3*2*2*3*2*3*self.DTAC
    
    def stateNum2Info(self, num):
        stateVariables = [0]*len(self.indexer)
        for backwardsIndex in range(len(self.indexer)):
            index = len(self.indexer)-backwardsIndex-1
            stateVariables[index]= int(num)/self.indexer[index]
            num = num%self.indexer[index]
        return stateVariables
    
    def Action_training(self, Qtable, trainingOn, randomstate):
        if not randomstate is None:
            np.random.set_state(randomstate)
        if trainingOn:
            choiceProbabilities = [.80, .20]
        else:
            choiceProbabilities = [1., 0.]
        set_choose = np.random.choice([0,1],p=choiceProbabilities)
        if (self.OnGrid==1 and self.collision==0):
            if (self.Location==0 or self.Location==2):
                if (self.RLane==1):       # this can be changed as random later down the road to leave the car to do some manouver 
                    self.LaneChange = 0  # no lane change when on correct lane and before or after intersection 
                elif (self.RLane==0 and self.AdjFreeLane==0 ):
                    self.LaneChange = 0 # no Lane change when on wrong Lane and adjecent lane is busy
                elif (self.RLane==0 and (self.AdjFreeLane ==1 or self.AdjFreeLane==2) and self.Lane ==1) :
                    self.LaneChange = 1  #right lane change if you are on the wrong  and left lane
                elif (self.RLane==0 and (self.AdjFreeLane ==1 or self.AdjFreeLane==2) and self.Lane ==0) :
                    self.LaneChange = 2  #left lane change if you are on the wrong  and right lane
                else:
                    self.LaneChange = np.array([0,1,2])
                    
            if(self.Location==1 or (self.Location==0 and self.NTI ==1) ):
                if (self.Turn==1):       # this can be changed as random later down the road to leave the car to do some manouver 
                    self.MakeTurn = 0  # no turn when no turn is needed and at the intersection 
                elif (self.Turn == 2 and self.Lane==0 ):
                    self.MakeTurn = 1  # turn right when you are in right lane and need to make right
                elif (self.Turn ==0 and self.Lane ==1) :
                    self.MakeTurn = 2  # turn left lane if you are on the left lane and need to make left
                else:
                    self.MakeTurn = np.array([0,1,2])
                    
           
            if (self.Location==0 or self.Location==2):
                if (self.TTNV == 0 and self.speed >0):       # this can be changed as random later down the road to leave the car to do some manouver 
                    self.ChangeSpeed = 1  # if the time to nearest vehicle is small and speed greater than zero decrease the speed
                elif (self.TTNV == 0 and self.speed ==0):
                    self.ChangeSpeed = 0
                elif (self.TTNV == 3 and self.speed <6):       # this can be changed as random later down the road to leave the car to do some manouver 
                    self.ChangeSpeed = 2  # if the time to nearest vehicle is large and speed less than 70 increase the speed
                elif (self.TTNV == 3 and self.speed ==6):
                    self.ChangeSpeed = 0
                else:
                    self.ChangeSpeed = np.array([0,1,2])
                    
            elif(self.Location==1 or (self.Location==0 and self.NTI ==1) ):
                if (self.TTNV == 0 and self.speed >0):       # this can be changed as random later down the road to leave the car to do some manouver 
                    self.ChangeSpeed = 1  # if the time to nearest vehicle is small and speed greater than zero decrease the speed
                    #self.ChangeSpeed = np.array([0,1,2])
                elif (self.TTNV == 0 and self.speed ==0):
                    self.ChangeSpeed = 0
                  #  self.ChangeSpeed = np.array([0,1,2])
                elif (self.TTNV == 3 and self.speed <7 and self.Turn==1):       # this can be changed as random later down the road to leave the car to do some manouver 
                    self.ChangeSpeed = 2  # if the time to nearest vehicle is large and going stright and speed less than 70 increase the speed
                elif (self.TTNV == 3 and self.speed <5 and (self.Turn==0 or self.Turn==2) ):       # this can be changed as random later down the road to leave the car to do some manouver 
                    self.ChangeSpeed = 2  # if the time to nearest vehicle is large and going stright and speed less than 70 increase the speed
                elif (self.TTNV == 3 and self.speed ==5 and (self.Turn==0 or self.Turn==2) ):       # this can be changed as random later down the road to leave the car to do some manouver 
                    self.ChangeSpeed = 0  # if the time to nearest vehicle is large and going stright and speed less than 70 increase the speed
                elif (self.TTNV == 3 and self.speed >5 and (self.Turn==0 or self.Turn==2) ):       # this can be changed as random later down the road to leave the car to do some manouver 
                    self.ChangeSpeed = 1  # if the time to nearest vehicle is large and going stright and speed less than 70 increase the speed
                elif (self.TTNV == 2 and self.speed <5 and (self.Turn==0 or self.Turn==2) ):       # this can be changed as random later down the road to leave the car to do some manouver 
                    self.ChangeSpeed = 2  # if the time to nearest vehicle is large and going stright and speed less than 70 increase the speed
                elif (self.TTNV == 2 and self.speed ==5 and (self.Turn==0 or self.Turn==2) ):       # this can be changed as random later down the road to leave the car to do some manouver 
                    self.ChangeSpeed = 0  # if the time to nearest vehicle is large and going stright and speed less than 70 increase the speed
                elif (self.TTNV == 2 and self.speed >5 and (self.Turn==0 or self.Turn==2) ):       # this can be changed as random later down the road to leave the car to do some manouver 
                    self.ChangeSpeed = 1  # if the time to nearest vehicle is large and going stright and speed less than 70 increase the speed
                else:
                    self.ChangeSpeed = np.array([0,1,2])
                    
            elif(self.Location==3 ):
                if (self.TTNV == 0 and self.speed >0):       # this can be changed as random later down the road to leave the car to do some manouver 
                    self.ChangeSpeed = 1  # if the time to nearest vehicle is small and speed greater than zero decrease the speed
                    #self.ChangeSpeed = np.array([0,1,2])
                elif (self.TTNV == 0 and self.speed ==0):
                    self.ChangeSpeed = 0
                  #  self.ChangeSpeed = np.array([0,1,2])
                elif (self.TTNV == 3 and self.speed <7 and self.Turn==1):       # this can be changed as random later down the road to leave the car to do some manouver 
                    self.ChangeSpeed = 2  # if the time to nearest vehicle is large and going stright and speed less than 70 increase the speed
                elif (self.TTNV == 3 and self.speed <5 and (self.Turn==0 or self.Turn==2) ):       # this can be changed as random later down the road to leave the car to do some manouver 
                    self.ChangeSpeed = 2  # if the time to nearest vehicle is large and going stright and speed less than 70 increase the speed
                elif (self.TTNV == 3 and self.speed ==5 and (self.Turn==0 or self.Turn==2) ):       # this can be changed as random later down the road to leave the car to do some manouver 
                    self.ChangeSpeed = 0  # if the time to nearest vehicle is large and going stright and speed less than 70 increase the speed
                elif (self.TTNV == 3 and self.speed >5 and (self.Turn==0 or self.Turn==2) ):       # this can be changed as random later down the road to leave the car to do some manouver 
                    self.ChangeSpeed = 1  # if the time to nearest vehicle is large and going stright and speed less than 70 increase the speed
                elif (self.TTNV == 2 and self.speed <5 and (self.Turn==0 or self.Turn==2) ):       # this can be changed as random later down the road to leave the car to do some manouver 
                    self.ChangeSpeed = 2  # if the time to nearest vehicle is large and going stright and speed less than 70 increase the speed
                elif (self.TTNV == 2 and self.speed ==5 and (self.Turn==0 or self.Turn==2) ):       # this can be changed as random later down the road to leave the car to do some manouver 
                    self.ChangeSpeed = 0  # if the time to nearest vehicle is large and going stright and speed less than 70 increase the speed
                elif (self.TTNV == 2 and self.speed >5 and (self.Turn==0 or self.Turn==2) ):       # this can be changed as random later down the road to leave the car to do some manouver 
                    self.ChangeSpeed = 1  # if the time to nearest vehicle is large and going stright and speed less than 70 increase the speed
                else:
                    self.ChangeSpeed = np.array([0,1,2])

        else:
            self.LaneChange = 0
            self.MakeTurn = 0
            self.ChangeSpeed = 0
            
        if (self.Location==0 or self.Location==2):    
            if (np.array(self.LaneChange).size>=1 or np.array(self.ChangeSpeed).size >=1):
                x=np.array(self.LaneChange)
                y=np.array(self.ChangeSpeed)
                action_list= np.transpose([np.tile(x, y.size), np.repeat(y, x.size)]) 
                ActionInd=np.random.choice(len(action_list))
                Actionr= action_list[ActionInd,:]
                #ActionpInd=np.argmax(Qtable[self.state,0:])
                ActionpArr = np.ravel(np.where(Qtable[self.state,0:]==np.amax(Qtable[self.state,0:9])))
                ActionpArr = ActionpArr[ActionpArr[:]<9]
                #Qrow = Qtable[self.state,0:9] # alternative way
                #ActionpArr = np.ravel(np.where(Qrow==np.amax(Qrow)))
                if (len(ActionpArr)>=1 and trainingOn):
                   ActionpInd = np.random.choice(ActionpArr,1)
                elif(len(ActionpArr)>=1 and trainingOn == False):
                   ActionpInd = ActionpArr[0]
                   if Qtable[self.state,ActionpInd]== 0:
                       self.NotTrainedFlag = True 
                   
                else:
                    raise Exception("agent line 198, shouldn't be here")#ActionpInd =-1
                       
                Actionp = np.zeros(2)
                if (ActionpInd//9==0):
                   # if (np.array(self.LaneChange).size>1):                    
                    Actionp[0] = (ActionpInd)//3 
                    #else:
                    #    Actionp[0] = self.LaneChange
                    Actionp[1] = (ActionpInd)%3
                    
                    if (set_choose == 0):
                        self.Action = Actionp
                    else:
                        self.Action = Actionr                    
                else:
                    raise Exception("agent line 213, shouldn't be here")#self.Action = Actionr
                
                self.LaneChange = self.Action[0]
                self.ChangeSpeed = self.Action[1]
                if (self.Turn==1):     # this can be changed as random later down the road to leave the car to do some manouver 
                    self.MakeTurn = 0  # no turn when no turn is needed and at the intersection 
                elif (self.Turn == 2):
                    self.MakeTurn = 1  # turn right when you are in right lane and need to make right
                elif (self.Turn ==0) :
                    self.MakeTurn = 2  # turn left lane if you are on the left lane  turn left
            else:
                raise Exception("agent line 224, shouldn't be here")#if (self.Turn==1):     # this can be changed as random later down the road to leave the car to do some manouver 
#                    self.MakeTurn = 0  # no turn when no turn is needed and at the intersection 
#                elif (self.Turn == 2):
#                    self.MakeTurn = 1  # turn right when you are in right lane and need to make right
#                elif (self.Turn ==0) :
#                    self.MakeTurn = 2  # turn left lane if you are on the left lane  turn left
            if (self.AdjFreeLane==0):
                self.LaneChange = 0
            self.action= 3* self.LaneChange + self.ChangeSpeed
            #self.NotTrainedFlag = False
        elif (self.Location==1):
            if (np.array(self.MakeTurn).size>=1 or np.array(self.ChangeSpeed).size >=1):
                
                x=np.array(self.MakeTurn)
                y=np.array(self.ChangeSpeed)
                action_list= np.transpose([np.tile(x, y.size), np.repeat(y, x.size)]) 
                ActionInd=np.random.choice(len(action_list))
                Actionr= action_list[ActionInd,:]
                #ActionpInd=np.argmax(Qtable[self.state,0:])
                ActionpArr = np.ravel(np.where(Qtable[self.state,:]==np.amax(Qtable[self.state,9:18]))) # take care
                ActionpArr = ActionpArr[ActionpArr[:]>=9]
                ActionpArr = ActionpArr[ActionpArr[:]<18] 
                if (len(ActionpArr)>=1 and trainingOn):
                    ActionpInd = np.random.choice(ActionpArr,1)
                elif(len(ActionpArr)>=1 and trainingOn == False):
                    ActionpInd = ActionpArr[0]
                    if Qtable[self.state, ActionpInd]== 0:
                       self.NotTrainedFlag = True 
                else:
                    raise Exception("agent line 247, shouldn't be here")#ActionpInd =-1
                
                Actionp = np.zeros(2)
                if (ActionpInd//9==1):
                    Actionp[0] = (ActionpInd - 9)//3
                    Actionp[1] = (ActionpInd - 9)%3
                    if (set_choose == 0):
                        self.Action = Actionp
                    else:
                        self.Action = Actionr
                    
                else:
                    raise Exception("agent line 259, shouldn't be here")
                
                self.MakeTurn = self.Action[0]
                self.ChangeSpeed = self.Action[1]
                self.LaneChange = 0
            else:
                raise Exception("agent line 265, shouldn't be here")#
            
            self.LaneChange = 0
            self.action= 9 + 3* self.MakeTurn + self.ChangeSpeed
           # self.NotTrainedFlag = False
        
        elif (self.Location==3):
            if (np.array(self.ChangeSpeed).size >=1):
                
                #x=np.array(self.MakeTurn)
                y=np.array([self.ChangeSpeed])
                action_list= y
                ActionInd=np.random.choice(len(action_list))
                Actionr= action_list[ActionInd]
                Actionr= np.append([0],Actionr)
                #ActionpInd=np.argmax(Qtable[self.state,0:])
                ActionpArr = np.ravel(np.where(Qtable[self.state,0:]==np.amax(Qtable[self.state,18:]))) # take care
                ActionpArr = ActionpArr[ActionpArr[:]>=18]
                if (len(ActionpArr)>=1 and trainingOn):
                   ActionpInd = np.random.choice(ActionpArr,1)
                elif(len(ActionpArr)>=1 and trainingOn == False):
                   ActionpInd = ActionpArr[0]
                   if Qtable[self.state, ActionpInd]== 0:
                       self.NotTrainedFlag = True 
                else:
                    raise Exception("agent line 283, shouldn't be here")#ActionpInd =-1
                
                Actionp = np.zeros(2)
                if (ActionpInd//9==2):
                    Actionp[0] = 0 #no turn decision is allowed to be made at the physical intersection 
                    Actionp[1] = (ActionpInd - 18)%3
                    if (set_choose == 0):
                        self.Action = Actionp
                    else:
                        self.Action = Actionr
                    
                else:
                    raise Exception("agent line 295, shouldn't be here")
                
                self.MakeTurn = self.Action[0]
                self.ChangeSpeed = self.Action[1]
                self.LaneChange = 0
            else:
                raise Exception("agent line 301, shouldn't be here")
                self.LaneChange = 0
            self.action= 18 + self.ChangeSpeed
        else:
            raise Exception("agent line 305, shouldn't be here")#self.LaneChange = 0
            #self.ChangeSpeed = 0
            #self.MakeTurn = 0
            ##Actionp=np.argmax(self.Qtable[self.state,:])
            #self.action= 0
        printss ="Carid "+ str(self.name) + " State "+ str(self.state) + " Action selected "+ str(self.action)          
        print printss    
        self.ac_array = np.array([self.LaneChange, self.ChangeSpeed,
                                  self.MakeTurn, self.NotTrainedFlag])
        self.NotTrainedFlag = False
        if randomstate is None:
            return self.ac_array
        return [self.ac_array, np.random.get_state()]       
            

        
    def reward(self,iteration):
#        if self.OnGrid == 0:
#            if self.collision > 0:
#                return -1000
#            else:
#                return 0
        
        if (self.OnGrid ==1 and self.TTNV>0 and self.FasterTNV>0 and self.DTAC>0 and self.DistanceToCrossCar>0 and self.speed<2):
            self.Rtime += (7. - self.speed)*(-2.0)/7.0
        elif(self.OnGrid == 1):
            self.Rtime = (7. - self.speed)*(-2.0)/7.0
        # reward for staying in wrong lane (before making turn decision)
        if (self.OnGrid == 1 and self.Location <=1 and not
                                self.PreviousStepInZone1):
            if self.RLane == 0:
                self.Rdest = self.Rdest -2
            else:
                self.Rdest = 0
        # reward for making wrong turning decision
        elif (self.OnGrid == 1 and self.PreviousStepInZone1 ):
            if (not self.MakeTurn == (self.Turn+2)%3):
                self.Rdest = self.Rdest -15
            else:
                self.Rdest = 0
        else:
            self.Rdest = 0
            
        if(self.Location==3 and self.IntersectionOpen == 0 and self.Priority ==0 and  self.PreviousStepInZone1):
            self.ETI = -12
        else:
            self.ETI = 0
            
        self.PreviousStepInZone1 = (self.Lane == 0 and self.Location == 1)
            
        if self.LaneChange>0.5:
            self.NLG += 1
            if(self.NLG>2 and self.RLane == 0 and self.Location ==0  ):
                self.RLG = -20#-1.0 * self.NLG
            elif(self.NLG>2 and self.Location ==2):
                self.RLG = -20#-1.0 * self.NLG         
            else:
                self.RLG = 0
        else:
            self.RLG = 0  
        
            
#        if (self.OnGrid == 1 and self.DTAC==0):
#            self.DTVA = self.DTVA -3
#        else:
#            self.DTVA = 0 #self.RTTC #the new tt collision to be implemented 
#       
        if (self.DTAC == 0 and self.FasterTNV ==0 and self.speed>0):
            self.ASBR += -1.*(2. - self.DTAC) * self.speed * 1./3.
        else:
            self.ASBR = 0
            
        if (self.OnGrid == 1 and self.TTNV==0):
            self.RTTC = self.RTTC -4
        else:
            self.RTTC = 0 #self.RTTC #the new tt collision to be implemented 
        if (self.collision==1):
            self.Rcoll = -1000
                
        self.TR =self.Rtime + self.Rdest + self.RTTC +self.Rcoll+ self.RLG+self.ETI+self.ASBR
        
        return self.TR
        
    def reward_def(self,atfault):
        
        self.Rtime1 = self.Rtime
        self.Rdest1 = self.Rdest
        self.RLG1 = self.RLG
        self.ASBR1 = self.ASBR
        
        if(self.OnGrid ==1 and self.IntersectionOpen == 0 and self.Priority ==0 and self.speed>0):
            self.ETI1 = self.ETI
        else:
            self.ETI1 = 0
            
        if (self.OnGrid == 1 and self.TTNV==0):
            self.RTTC1 = self.RTTC
        else:
            self.RTTC1 = 0
       
#        if (self.OnGrid == 1 and self.DTVA==0):
#            self.DTVA1 = self.DTVA
#        else:
#            self.DTVA1 = 0
# 
        self.Rcoll1 = 0
        if (self.collision==1 or atfault>0):
            self.Rcoll1 = -1000
        self.TR1 =self.Rtime1 + self.Rdest1 + self.RTTC1 +self.Rcoll1 +self.RLG1+self.ETI1+self.ASBR1
        
        return self.TR1    
        
# location is a tuple to index the Qtable
def updateQvalue(oldval, reward):
    return .5 * oldval + .5 * reward
