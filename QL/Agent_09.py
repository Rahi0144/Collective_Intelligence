# -*- coding: utf-8 -*-
"""
Last mod Rahi 9/10/16 increased wrong turn penalty
"""
import GSettings
import numpy as np
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.linear_model import OrthogonalMatchingPursuitCV
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
        self.LaneChange = 0
        self.PreviousStepInZone1 = False
        self.PreviousPriority = False
        self.indexer = list(np.prod(agent_optionCounts[:j]) for j in range(len(agent_optionCounts)))
        self.SparseRec_index =0
        
    # Agent Location={Pre-intersection, logical-intersection, Post-intersection, at-intersection }={0,1,2,3}
    # Agent Lane=(in this case){right,left }= {0,1}   
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
        self.Lane = AgentLane
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
                             self.IntersectionOpen, 1]#self.Priority]
        
        self.state = np.dot(self.indexer, allStateVariables)
        
    
    def stateNum2Info(self, num):
        stateVariables = [0]*len(self.indexer)
        for backwardsIndex in range(len(self.indexer)):
            index = len(self.indexer)-backwardsIndex-1
            stateVariables[index]= int(num)/self.indexer[index]
            num = num%self.indexer[index]
        return stateVariables
        
    def ActionNum2Info(self, Ac_idx):
        if Ac_idx >=18:
            Action_array = np.array([0,0,Ac_idx%3])
        elif Ac_idx>=9:
            Action_array = np.array([(Ac_idx-9)/3,0,(Ac_idx-9)%3])
        else:
            Action_array = np.array([0,Ac_idx/3,Ac_idx%3])
        return Action_array
    
    def Action_training(self, Qtable, trainingOn, randomstate):
        if not randomstate is None:
            np.random.set_state(randomstate)
        if trainingOn:
            choiceProbabilities = [.99, .01]
            getActionFromTable = self.getActionFromTable_simple
        else:
            choiceProbabilities = [1., 0.]
            if (GSettings.SparseRecFlag == 0):
                rr=self.getActionFromSparseRecovery( Qtable,GSettings.SparseRecFlag,GSettings.SR_w,trainingOn)
                GSettings.SparseRecFlag = 100
                GSettings.SR_w = rr[0]
                getActionFromTable = self.getActionFromSparseRecovery
            else:
                getActionFromTable = self.getActionFromSparseRecovery
                
            #getActionFromTable = self.getActionFromTable_knn
            
        set_choose = np.random.choice([False,True],p=choiceProbabilities)
        
        if set_choose: # choose randomly from acceptable actions
            MakeTurn = [0,1,2]
            LaneChange = [0,1,2]
            ChangeSpeed = [0,1,2]
            
            # first define the set of acceptable actions for this state
            if self.Location == 0 or self.Location == 2:
                if (self.Turn == 1): 
                    MakeTurn = 0  # no turn when no turn is needed and at the intersection 
                elif (self.Turn == 2):
                    MakeTurn = 1  # turn right when you are in right lane and need to make right
                elif (self.Turn == 0):
                    MakeTurn = 2  # turn left lane if you are on the left lane  turn left                
                
                if (self.RLane==1):       # this can be changed as random later down the road to leave the car to do some manouver 
                    LaneChange = 0  # no lane change when on correct lane and before or after intersection 
                elif (self.RLane==0 and self.AdjFreeLane==0 ):
                    LaneChange = 0 # no Lane change when on wrong Lane and adjecent lane is busy
                elif (self.RLane==0 and (self.AdjFreeLane ==1 or self.AdjFreeLane==2) and self.Lane ==1) :
                    LaneChange = 1  #right lane change if you are on the wrong  and left lane
                elif (self.RLane==0 and (self.AdjFreeLane ==1 or self.AdjFreeLane==2) and self.Lane ==0) :
                    LaneChange = 2  #left lane change if you are on the wrong  and right lane
                    
                if (self.TTNV == 0 and self.speed >0):       # this can be changed as random later down the road to leave the car to do some manouver 
                    ChangeSpeed = 1  # if the time to nearest vehicle is small and speed greater than zero decrease the speed
                elif (self.TTNV == 0 and self.speed ==0):
                    ChangeSpeed = 0
                elif (self.TTNV == 3 and self.speed <6):       # this can be changed as random later down the road to leave the car to do some manouver 
                    ChangeSpeed = 2  # if the time to nearest vehicle is large and speed less than 70 increase the speed
                elif (self.TTNV == 3 and self.speed ==6):
                    ChangeSpeed = 0
                    
                x=np.array(LaneChange)
                y=np.array(ChangeSpeed)    
                action_map = np.transpose([np.repeat(x, y.size), np.tile(y, x.size)])
                action_map = np.array([[MakeTurn, act[0], act[1]] for act in action_map])
                action_list = np.dot(action_map, [0,3,1])
                    
            elif self.Location==1:
                if (self.Turn==1):       # this can be changed as random later down the road to leave the car to do some manouver 
                    MakeTurn = 0  # no turn when no turn is needed and at the intersection 
                elif (self.Turn == 2 and self.Lane==0 ):
                    MakeTurn = 1  # turn right when you are in right lane and need to make right
                elif (self.Turn ==0 and self.Lane ==1) :
                    MakeTurn = 2  # turn left lane if you are on the left lane and need to make left
                    
                if (self.TTNV == 0 and self.speed >0):       # this can be changed as random later down the road to leave the car to do some manouver 
                    ChangeSpeed = 1  # if the time to nearest vehicle is small and speed greater than zero decrease the speed
                    #self.ChangeSpeed = np.array([0,1,2])
                elif (self.TTNV == 0 and self.speed ==0):
                    ChangeSpeed = 0
                  #  self.ChangeSpeed = np.array([0,1,2])
                elif (self.TTNV == 3 and self.speed <7 and self.Turn==1):       # this can be changed as random later down the road to leave the car to do some manouver 
                    ChangeSpeed = 2  # if the time to nearest vehicle is large and going stright and speed less than 70 increase the speed
                elif (self.TTNV == 3 and self.speed <5 and (self.Turn==0 or self.Turn==2) ):       # this can be changed as random later down the road to leave the car to do some manouver 
                    ChangeSpeed = 2  # if the time to nearest vehicle is large and going stright and speed less than 70 increase the speed
                elif (self.TTNV == 3 and self.speed ==5 and (self.Turn==0 or self.Turn==2) ):       # this can be changed as random later down the road to leave the car to do some manouver 
                    ChangeSpeed = 0  # if the time to nearest vehicle is large and going stright and speed less than 70 increase the speed
                elif (self.TTNV == 3 and self.speed >5 and (self.Turn==0 or self.Turn==2) ):       # this can be changed as random later down the road to leave the car to do some manouver 
                    ChangeSpeed = 1  # if the time to nearest vehicle is large and going stright and speed less than 70 increase the speed
                elif (self.TTNV == 2 and self.speed <5 and (self.Turn==0 or self.Turn==2) ):       # this can be changed as random later down the road to leave the car to do some manouver 
                    ChangeSpeed = 2  # if the time to nearest vehicle is large and going stright and speed less than 70 increase the speed
                elif (self.TTNV == 2 and self.speed ==5 and (self.Turn==0 or self.Turn==2) ):       # this can be changed as random later down the road to leave the car to do some manouver 
                    ChangeSpeed = 0  # if the time to nearest vehicle is large and going stright and speed less than 70 increase the speed
                elif (self.TTNV == 2 and self.speed >5 and (self.Turn==0 or self.Turn==2) ):       # this can be changed as random later down the road to leave the car to do some manouver 
                    ChangeSpeed = 1  # if the time to nearest vehicle is large and going stright and speed less than 70 increase the speed
                    
                x = np.array(MakeTurn)
                y = np.array(ChangeSpeed)
                action_map = np.transpose([np.repeat(x, y.size), np.tile(y, x.size)])
                action_map = np.array([[act[0], 0, act[1]] for act in action_map])
                action_list = np.dot(action_map, [3,0,1]) + 9
              
            else:
                if (self.TTNV == 0 and self.speed >0):       # this can be changed as random later down the road to leave the car to do some manouver 
                    ChangeSpeed = [1]  # if the time to nearest vehicle is small and speed greater than zero decrease the speed
                    #self.ChangeSpeed = np.array([0,1,2])
                elif (self.TTNV == 0 and self.speed ==0):
                    ChangeSpeed = [0]
                  #  self.ChangeSpeed = np.array([0,1,2])
                elif (self.TTNV == 3 and self.speed <7 and self.Turn==1):       # this can be changed as random later down the road to leave the car to do some manouver 
                    ChangeSpeed = [2]  # if the time to nearest vehicle is large and going stright and speed less than 70 increase the speed
                elif (self.TTNV == 3 and self.speed <5 and (self.Turn==0 or self.Turn==2) ):       # this can be changed as random later down the road to leave the car to do some manouver 
                    ChangeSpeed = [2]  # if the time to nearest vehicle is large and going stright and speed less than 70 increase the speed
                elif (self.TTNV == 3 and self.speed ==5 and (self.Turn==0 or self.Turn==2) ):       # this can be changed as random later down the road to leave the car to do some manouver 
                    ChangeSpeed = [0]  # if the time to nearest vehicle is large and going stright and speed less than 70 increase the speed
                elif (self.TTNV == 3 and self.speed >5 and (self.Turn==0 or self.Turn==2) ):       # this can be changed as random later down the road to leave the car to do some manouver 
                    ChangeSpeed = [1]  # if the time to nearest vehicle is large and going stright and speed less than 70 increase the speed
                elif (self.TTNV == 2 and self.speed <5 and (self.Turn==0 or self.Turn==2) ):       # this can be changed as random later down the road to leave the car to do some manouver 
                    ChangeSpeed = [2]  # if the time to nearest vehicle is large and going stright and speed less than 70 increase the speed
                elif (self.TTNV == 2 and self.speed ==5 and (self.Turn==0 or self.Turn==2) ):       # this can be changed as random later down the road to leave the car to do some manouver 
                    ChangeSpeed = [0]  # if the time to nearest vehicle is large and going stright and speed less than 70 increase the speed
                elif (self.TTNV == 2 and self.speed >5 and (self.Turn==0 or self.Turn==2) ):       # this can be changed as random later down the road to leave the car to do some manouver 
                    ChangeSpeed = [1]  # if the time to nearest vehicle is large and going stright and speed less than 70 increase the speed
                    
                ChangeSpeed = np.array(ChangeSpeed)
                action_map = np.array([[0,0,speed] for speed in ChangeSpeed])
                action_list = ChangeSpeed + 18
                
            ActionIndex = np.random.choice(len(action_list))
            
            
        else: # choosing best option from table (out of allowed options)
            
            LaneChange = [0,1,2]
            ChangeSpeed = [0,1,2]
            MakeTurn = [0,1,2]
            
            if self.Location == 0 or self.Location == 2:
                # this can be changed as random later down the road to leave the car to do some manouver
                if (self.Turn == 1): 
                    MakeTurn = 0  # no turn when no turn is needed and at the intersection 
                elif (self.Turn == 2):
                    MakeTurn = 1  # turn right when you are in right lane and need to make right
                elif (self.Turn == 0):
                    MakeTurn = 2  # turn left lane if you are on the left lane  turn left
                if (self.AdjFreeLane==0):
                    LaneChange = 0
                if self.FasterTNV ==0 and self.DTAC <2:# and trainingOn == False:
                    ChangeSpeed = 1
                if self.Location == 0 and self.Priority == 0:
                    ChangeSpeed = 1
                x = np.array(LaneChange)
                y = np.array(ChangeSpeed)
                action_map = np.transpose([np.repeat(x, y.size), np.tile(y, x.size)])
                action_map = np.array([[MakeTurn, act[0], act[1]] for act in action_map])
                action_list = np.dot(action_map, [0,3,1])
                
            elif self.Location == 1:
                if self.FasterTNV ==0 and self.DTAC <2:# and trainingOn == False:
                    ChangeSpeed = 1
                if self.Priority == 0:
                    ChangeSpeed = 1
                x = np.array(MakeTurn)
                y = np.array(ChangeSpeed)            
                action_map = np.transpose([np.repeat(x, y.size), np.tile(y, x.size)])
                action_map = np.array([[act[0], 0, act[1]] for act in action_map])
                action_list = np.dot(action_map, [3,0,1]) + 9
                
            else:                
                if self.FasterTNV == 0 and self.DTAC < 2:# and trainingOn == False:
                    ChangeSpeed = [1]
                action_map = np.array([[0,0,speed] for speed in ChangeSpeed])
                action_list = np.array(ChangeSpeed) + 18
            
            rr = getActionFromTable(Qtable[:,action_list], GSettings.SparseRecFlag, GSettings.SR_w, trainingOn)
            ActionIndex =rr[1]
        # decompress chosen action into individual driving decisions
        self.action = action_list[ActionIndex]
        self.MakeTurn = action_map[ActionIndex, 0]
        self.LaneChange = action_map[ActionIndex, 1]
        self.ChangeSpeed = action_map[ActionIndex, 2]
        
        printss = "Carid "+ str(self.name) + " State "+ str(self.state) +\
                    " Action selected "+ str(self.action)
        #print printss
        self.ac_array = np.array([self.LaneChange, self.ChangeSpeed,
                                  self.MakeTurn, self.NotTrainedFlag])
        self.NotTrainedFlag = False # resetting
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
            self.Rtime = (7. - self.speed)*(-2.0)/7.0* 10.
        elif(self.OnGrid == 1):
            self.Rtime = (7. - self.speed)*(-2.0)/7.0
        # reward for staying in wrong lane (before making turn decision)
        if (self.OnGrid == 1 and self.Location <=1 and not
                                self.PreviousStepInZone1):
            if self.RLane == 0:
                self.Rdest = -10.
            else:
                self.Rdest = 0
        # reward for making wrong turning decision
        elif (self.OnGrid == 1 and self.PreviousStepInZone1 ):
            if (not self.MakeTurn == (self.Turn+2)%3):
                self.Rdest = -75
            else:
                self.Rdest = 0
        else:
            self.Rdest = 0
            
        if(self.Location==3 and self.PreviousPriority and self.PreviousStepInZone1):
            self.ETI = -100
        else:
            self.ETI = 0
            
        self.PreviousPriority = self.Priority==0 and self.IntersectionOpen==0
        self.PreviousStepInZone1 = self.Lane == 0 and self.Location == 1
            
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
            self.ASBR = -1.*(2. - self.DTAC) * self.speed *1.5
        else:
            self.ASBR = 0
            
        if (self.OnGrid == 1 and self.TTNV==0):
            self.RTTC =  -12
        else:
            self.RTTC = 0 #self.RTTC #the new tt collision to be implemented 
        if (self.collision==1):
            self.Rcoll = -1500
                
        self.TR =self.Rtime + self.Rdest + self.RTTC +self.Rcoll+ self.RLG+self.ETI+self.ASBR
        
        return self.TR
        
    def reward_def(self,atfault):
        
        self.Rtime1 = self.Rtime
        self.Rdest1 = self.Rdest
        self.RLG1 = self.RLG
        self.ASBR1 = self.ASBR
        
        if(self.OnGrid ==1 and self.PreviousPriority and self.speed>0):
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
        if self.collision==1:
            self.Rcoll1 = -1500
        if atfault > 0:
            self.Rcoll1 = -1400
        self.TR1 =self.Rtime1 + self.Rdest1 + self.RTTC1 +self.Rcoll1 +self.RLG1+self.ETI1+self.ASBR1
        
        return self.TR1    
        
        
    def getActionFromTable_simple(self, Qtable, trainingOn):
        ActionpArr = np.ravel(np.where(Qtable[self.state,:]==np.amax(Qtable[self.state,:])))
        assert(len(ActionpArr) >= 1)
        if trainingOn:
            ActionpInd = np.random.choice(ActionpArr)
        else:
            ActionpInd = ActionpArr[0]
        if Qtable[self.state, ActionpInd]== 0:
            self.NotTrainedFlag = True
        return ActionpInd
            
    
    def getActionFromTable_knn(self, Qtable, trainingOn):
        currentState = Qtable[self.state,:]
        currentStateList = self.stateNum2Info(self.state)       
        
        # perform standard action selection, unless there was no training on this state
        ActionpArr = np.ravel(np.where(currentState==np.amax(currentState)))
        assert(len(ActionpArr) >= 1)
        if trainingOn:
            ActionpInd = np.random.choice(ActionpArr)
        else:
            ActionpInd = ActionpArr[0]
        if currentState[ActionpInd] != 0:
            return ActionpInd
        
        self.NotTrainedFlag = True
        
        # gather a list of all states that can be compared to the current state
        # calculate the (weighted) distance of each state to the current state
        stateVarsToChangeList = [0,1,7,8,9,10]
        stateVarsToKeepList = [i for i in range(13) if not (i in stateVarsToChangeList)]
        stateWeights = [1.]*len(stateVarsToChangeList) # for now
        n_grids = np.mgrid[0:4,0:7,0:3,0:2,0:2,0:2]
        state_grid = n_grids[0] * 0.
        distance_grid = n_grids[0] * 0.
        for index,stateIndex in enumerate(stateVarsToChangeList):
            state_grid = state_grid + n_grids[index]*self.indexer[stateIndex]
            distance = n_grids[index] - currentStateList[stateIndex]
            distance_grid = distance_grid + np.abs(distance)*stateWeights[index]
        for stateIndex in stateVarsToKeepList:
            state_grid = state_grid +\
                        self.indexer[stateIndex]*currentStateList[stateIndex]
        all_allowed_states = state_grid.flatten()
        all_distances = distance_grid.flatten()
        
        weightByDistance = lambda dist: 1./(dist + .5)
        
        # score actions by how many nearby states chose this action
        actionScores = np.zeros([9,1])
        for idx, state in enumerate(all_allowed_states):
            if np.all(Qtable[state,:] == 0.):
                continue
            theseActions = Qtable[state,:].copy()
            theseActions[theseActions == 0.] = -100000.
            bestAction = np.argmax(theseActions)
            actionScores[bestAction] = actionScores[bestAction] +\
                                        weightByDistance(all_distances[idx])
                                    
        if np.all(actionScores == 0.):
            print "----- Totally zero action - "+str(self.state)
        return np.argmax(actionScores) # return highest-scoring action
        
    def getActionFromSparseRecovery(self, Qtable,SparseRecFlag,w,trainingOn):
        n_nonzero_coefs =31
        if SparseRecFlag==0:
            Train_index = np.where(Qtable<0)
            R=Qtable[Train_index]
           #Phi = []
            for i in range(len(Train_index[0])):
                State_array = self.stateNum2Info(Train_index[0][i])
                Action_arr = self.ActionNum2Info(Train_index[1][i])
                '''[self.TTNV, self.speed, self.AdjFreeLane,
                         self.RLane, self.Turn, self.Lane,
                         self.Location, self.DTAC, self.SlowerTNV,
                         self.FasterTNV, self.DistanceToCrossCar, 
                         self.IntersectionOpen, 1]#self.Priority]'''
                Psi= np.array([])
                Psi = np.concatenate((Psi,State_array[0:2]))
                for j in range(2,7):
                    Psi = np.concatenate((Psi,self.convertToOneHot(np.array([State_array[j],agent_optionCounts[j]-1]))[0]))
                Psi = np.concatenate((Psi, State_array[7:11]))
                Psi = np.concatenate((Psi, self.convertToOneHot(np.array([State_array[11],agent_optionCounts[11]-1]))[0]))
                
                for k in range(3):
                    Psi = np.concatenate((Psi,self.convertToOneHot(np.array([Action_arr[k],2]))[0]))
                if i==0 :
                    Phi = Psi
                else:
                    Phi = np.vstack((Phi,Psi))
                    
            omp = OrthogonalMatchingPursuit(n_nonzero_coefs=n_nonzero_coefs)
            omp.fit(Phi, R)
            w = omp.coef_
            idx_r, = w.nonzero()
            
            omp_cv = OrthogonalMatchingPursuitCV(copy=True, fit_intercept=True, normalize=True, max_iter=n_nonzero_coefs, cv=None, n_jobs=1, verbose=False)
            omp_cv.fit(Phi, R)
            w1 = omp_cv.coef_
            idx_r_cv, = w1.nonzero()
          #  return w
            #use to predict the other columns 
        
        ActionpArr = np.ravel(np.where(Qtable[self.state,:]==np.amax(Qtable[self.state,:])))
        assert(len(ActionpArr) >= 1)
        if trainingOn:
            ActionpInd = np.random.choice(ActionpArr)
        else:
            ActionpInd = ActionpArr[0]
        if Qtable[self.state, ActionpInd]== 0 :
            for i in  np.ravel(np.where(Qtable[self.state,:]==0)):
                State_array = np.array(self.stateNum2Info(int(self.state)))
                if self.Location ==0 or self.Location==2:
                    action = i
                elif self.Location ==1:
                    action = i + 9
                else:
                    action = i + 18
                    
                Action_arr = self.ActionNum2Info(action)
                '''[self.TTNV, self.speed, self.AdjFreeLane,
                         self.RLane, self.Turn, self.Lane,
                         self.Location, self.DTAC, self.SlowerTNV,
                         self.FasterTNV, self.DistanceToCrossCar, 
                         self.IntersectionOpen, 1]#self.Priority]'''
              
                Psi= np.array([])
                Psi = np.concatenate((Psi,State_array[0:2]))
                for j in range(2,7):
                    Psi = np.concatenate((Psi,self.convertToOneHot(np.array([State_array[j],agent_optionCounts[j]-1]).astype(int))[0]))
                Psi = np.concatenate((Psi, State_array[7:11]))
                Psi = np.concatenate((Psi, self.convertToOneHot(np.array([State_array[11],agent_optionCounts[11]-1]).astype(int))[0]))
                
                for k in range(3):
                    Psi = np.concatenate((Psi,self.convertToOneHot(np.array([Action_arr[k],2]))[0]))
                print w
                print Psi 
                Qtable[self.state , i ]= np.dot(Psi,w)
        ActionpArr = np.ravel(np.where(Qtable[self.state,:]==np.amax(Qtable[self.state,:])))
        assert(len(ActionpArr) >= 1)            
        if trainingOn:
            ActionpInd = np.random.choice(ActionpArr)
        else:
            ActionpInd = ActionpArr[0]
        if Qtable[self.state, ActionpInd]== 0:
            self.NotTrainedFlag = True
            
        return  [w , ActionpInd]


    def convertToOneHot(self, vector, num_classes=None):
        """
        Converts an input 1-D vector of integers into an output
        2-D array of one-hot vectors, where an i'th input value
        of j will set a '1' in the i'th row, j'th column of the
        output array.
    
        Example:
            v = np.array((1, 0, 4))
            one_hot_v = convertToOneHot(v)
            print one_hot_v
    
            [[0 1 0 0 0]
             [1 0 0 0 0]
             [0 0 0 0 1]]
        """
    
        assert isinstance(vector, np.ndarray)
        assert len(vector) > 0
    
        if num_classes is None:
            num_classes = np.max(vector)+1
        else:
            assert num_classes > 0
            assert num_classes >= np.max(vector)
    
        result = np.zeros(shape=(len(vector), num_classes))
        result[np.arange(len(vector)), vector] = 1
        return result.astype(int)
                        
