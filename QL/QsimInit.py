# -*- coding: utf-8 -*-
"""
Eventually we can set up more complex simulations from this file.
"""

import pandas as pd
import numpy as np 
# takes an iteration number
# return a pandas dataframe with ['time', 'lane', 'dest', 'speed'  ]
# 'time' = creation time for each vehicle, in .1s resolution
# 'lane' = exact lane the vehicle starts in, format 'Ai_B'
#           A is road [1,2,3,4]         B is lane [0,1]
# 'dest' = exact lane the vehicle wants to end in, format 'Ao_B'
# 'speed' vehicle's starting speed in meters per second
def initialize(iteration):
    
#    SourceList =  np.array(['1i_0','1i_1','2i_0','2i_1'])
#    DestList = np.array(['1o_0','1o_1','2o_0','2o_1','3o_0','3o_1','4o_0','4o_1'])
#    CarCount = 6
#    CarInit= pd.DataFrame(np.zeros([CarCount,4]),columns = ['time','lane','dest','speed'])
#    CarInit['lane'] = CarInit['lane'].astype(str)
#    CarInit.loc[0]= [0.0,'1i_1','4o_1',8.]
#    CarInit.loc[1]= [0.0,'2i_0','1o_0',8.]
#    CarInit.loc[2]= [2.0,'1i_1','4o_1',8.]
#    CarInit.loc[3]= [2.0,'2i_0','1o_0',8.]
#    CarInit.loc[4]= [4.0,'2i_0','1o_0',8.]
#    CarInit.loc[5]= [4.0,'1i_1','4o_1',8.]
#    return CarInit
     
#   # SourceList =  np.array(['1i_0','1i_1','2i_0','2i_1'])
#    SourceList = np.array(['1i_1','2i_0'])
#  #  DestList = np.array(['1o_0','1o_1','2o_0','2o_1','3o_0','3o_1','4o_0','4o_1'])
#    DestList = np.array(['1o_0','4o_1'])
#    N = int(np.sqrt(iteration) * 4.0)
#    CarCount = np.random.poisson(N, 1) + 1
#    CarInit= pd.DataFrame(np.zeros([CarCount,4]),columns = ['time','lane','dest','speed'])
#    CarInit['lane'] = CarInit['lane'].astype(str)
#    sd =np.zeros([2]).astype(int)
#    for i in range (CarCount):
#        sd[0] = np.random.randint(0,len(SourceList),1)
#        if sd[0] == 0:
#            sd[1] = 1
#        else:
#            sd[1] = 0
##        while sd[0]//2 == sd[1]//2:
##            sd = np.random.randint(0,min(len(SourceList),len(DestList)),2)
#        CarSpeed = np.random.randint(1,3)*10.0
#        s = SourceList[sd[0]]
#        d = DestList[sd[1]]
#        if np.shape(CarInit[CarInit['lane'].str.contains(s)])[0]>=1:
#            CarTime = (np.shape(CarInit[CarInit['lane'].str.contains(s)])[0]+1)*1.5
#        else:
#            CarTime = 0 
#        CarInit.loc[i]= [CarTime,s,d,CarSpeed]
#    return CarInit
     
    meanCarNum = 65
    SourceList =  np.array(['1i_0','1i_1','2i_0','2i_1','3i_0','3i_1','4i_0','4i_1'])
    DestList = np.array(['1o_0','1o_1','2o_0','2o_1','3o_0','3o_1','4o_0','4o_1'])
    CarCount = np.random.poisson(meanCarNum, 1) + 1
    CarInit= pd.DataFrame(np.zeros([CarCount,4]),columns = ['time','lane','dest','speed'])
    CarInit['lane'] = CarInit['lane'].astype(str)
    sd =np.zeros([2]).astype(int)
    for i in range (CarCount):
        sd[0] = np.random.randint(0,len(SourceList),1)
        sd[1] = np.random.randint(0,len(DestList),1)
        if (sd[0]==0):        
            sd[1] = np.random.choice(np.array([2,3,4,5]),1)
        elif(sd[0]==1):
            sd[1] = np.random.choice(np.array([6,7]),1)
        elif(sd[0]==2):
            sd[1] = np.random.choice(np.array([0,1,6,7]),1)
        elif(sd[0]==3):
            sd[1] = np.random.choice(np.array([4,5]),1)
        while (sd[0]//2 == sd[1]//2):
            sd[1] = np.random.randint(0,len(DestList),1)
        CarSpeed = np.random.randint(1,4)*10.0/3.6
        s = SourceList[sd[0]]
        st = SourceList[sd[0]][0:2]
        d = DestList[sd[1]]
        if np.shape(CarInit[CarInit['lane'].str.contains(st)])[0]>=1:
            CarTime = (np.shape(CarInit[CarInit['lane'].str.contains(st)])[0])* 1
        else:
            CarTime = 0
        CarTime = CarTime + np.random.uniform(0,.5)
        CarInit.loc[i]= [CarTime,s,d,CarSpeed]
    return CarInit     
     
#    meanCarNum = 29#(iteration)**.5 *4
#    rightsources = [['1i_0','3o_0'],['1i_1','3o_0'],['2i_0','4o_0'],
#                    ['2i_1','4o_0']]#,['3i_0','2o_0'],['4i_0','1o_0']]
#    othersources = [['1i_0','2o_0'],['1i_1','2o_0'],['1i_0','4o_1'],
#                    ['1i_1','4o_1'],['2i_0','1o_0'],['2i_1','1o_0'],
#                    ['2i_0','3o_1'],['2i_1','3o_1']]
#    CarCount = np.random.poisson(meanCarNum, 1) + 1
#    CarInit= pd.DataFrame(np.zeros([CarCount,4]),columns = ['time','lane','dest','speed'])
#    CarInit['lane'] = CarInit['lane'].astype(str)
#    sd =np.zeros([2]).astype(int)
#    for i in range (CarCount):
#        if np.random.uniform() > .2:
#            SourceList = othersources
#        else:
#            SourceList = rightsources
#        sd = np.random.randint(0,len(SourceList),1)
#        CarSpeed = np.random.randint(1,4)*10.0/3.6
#        s, d = SourceList[sd]
#        st = s[0:2]
#        if np.shape(CarInit[CarInit['lane'].str.contains(st)])[0]>=1:
#            CarTime = (np.shape(CarInit[CarInit['lane'].str.contains(st)])[0])* 1.
#        else:
#            CarTime = 0
#        CarTime = CarTime + np.random.uniform(0,.5)
#        CarInit.loc[i]= [CarTime,s,d,CarSpeed]
#    return CarInit 

#import pandas as pd
#import numpy as np 
## takes an iteration number
## return a pandas dataframe with ['time', 'lane', 'dest', 'speed'  ]
## 'time' = creation time for each vehicle, in .1s resolution
## 'lane' = exact lane the vehicle starts in, format 'Ai_B'
##           A is road [1,2,3,4]         B is lane [0,1]
## 'dest' = exact lane the vehicle wants to end in, format 'Ao_B'
## 'speed' vehicle's starting speed in meters per second
#def initialize(iteration):
##    SourceList = np.array(['1i_1','2i_0'])
##  #  DestList = np.array(['1o_0','1o_1','2o_0','2o_1','3o_0','3o_1','4o_0','4o_1'])
##    DestList = np.array(['1o_0','4o_1'])
##    N = int(np.sqrt(iteration) * 4.)
##    CarCount = np.random.poisson(N, 1) + 1
##    CarInit= pd.DataFrame(np.zeros([CarCount,4]),columns = ['time','lane','dest','speed'])
##    CarInit['lane'] = CarInit['lane'].astype(str)
##    sd =np.zeros([2]).astype(int)
##    for i in range (CarCount):
##        sd[0] = np.random.randint(0,len(SourceList),1)
##        if sd[0] == 0:
##            sd[1] = 1
##        else:
##            sd[1] = 0
###        while sd[0]//2 == sd[1]//2:
###            sd = np.random.randint(0,min(len(SourceList),len(DestList)),2)
##        CarSpeed = np.random.randint(1,3)*10.0
##        s = SourceList[sd[0]]
##        d = DestList[sd[1]]
##        if np.shape(CarInit[CarInit['lane'].str.contains(s)])[0]>=1:
##            CarTime = (np.shape(CarInit[CarInit['lane'].str.contains(s)])[0]+1)*1.5
##        else:
##            CarTime = 0
##        CarTime = CarTime + np.random.uniform(0.,.5) # added 5/30/16
##        CarInit.loc[i]= [CarTime,s,d,CarSpeed]
##    return CarInit
#    meanCarNum = 48 #(iteration)**.5 *4. 
#    SourceList =  np.array(['1i_0','1i_1','2i_0','2i_1'])#,'3i_0','3i_1','4i_0','4i_1'])
#    #SourceList =  np.array(['1i_1','2i_0'])
#    DestList = np.array(['1o_0','1o_1','2o_0','2o_1','3o_0','3o_1','4o_0','4o_1'])
#    #DestList = np.array(['1o_0','1o_1','4o_0','4o_1'])
#    CarCount = np.random.poisson(meanCarNum, 1) + 1
#    CarInit= pd.DataFrame(np.zeros([CarCount,4]),columns = ['time','lane','dest','speed'])
#    CarInit['lane'] = CarInit['lane'].astype(str)
#    sd =np.zeros([2]).astype(int)
#    for i in range (CarCount):
#        sd[0] = np.random.randint(0,len(SourceList),1)
#        sd[1] = np.random.randint(0,len(DestList),1)
#        while sd[0]//2 == sd[1]//2:
#            sd[1] = np.random.randint(0,len(DestList))
#        CarSpeed = np.random.randint(1,4)*10.0/3.6
#        s = SourceList[sd[0]]
#        d = DestList[sd[1]]
#        if np.shape(CarInit[CarInit['lane'].str.contains(s)])[0]>=1:
#            CarTime = (np.shape(CarInit[CarInit['lane'].str.contains(s)])[0])*1.5
#        else:
#            CarTime = 0
#        CarTime = CarTime + np.random.uniform(0,.5)
#        CarInit.loc[i]= [CarTime,s,d,CarSpeed]
#    return CarInit
initialize(48)