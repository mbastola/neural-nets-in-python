"""
Manil Bastola
The Brachistochrone Problem

Used OpenAI GYM's Mountain Car Continuous for reference:
http://incompleteideas.net/sutton/MountainCar/MountainCar1.cp
permalink: https://perma.cc/6Z2N-PFWC
"""

import sys
import math
import numpy as np
import brachistochrone as bc
from scipy.optimize import fsolve
from scipy.signal import savgol_filter
from gym.envs.classic_control import rendering

def polyfit(points):
    coeffs = np.polyfit(points[:,0], points[:,1], 2)
    print(coeffs)
    return coeffs;

#smooth data using convolution
def smooth(x):
    filter_size = 41
    filter_ = np.ones(filter_size)/filter_size
    x_smooth = np.convolve(x, filter_, mode='same')
    return x_smooth

def smooth1(x):
    filter_size = 41
    return savgol_filter(x,filter_size,4)

def cycloidEqnY(t, a):
    r = R_cycloid
    return (- r +  r * math.cos(t) - a[1] )

def cycloidEqnXY(p, a):
    r, t = p
    return ( ( r * t -  r * math.sin(t) - a[0] ), (- r +  r * math.cos(t) - a[1] ) )

def solveCycloidInit(a,b):
    a0 = [ (b[0] - a[0]), (b[1] - a[1]) ]
    r, t = fsolve( cycloidEqnXY , (1,1) , a0)
    return r , t
    
def solveCycloid(a,b):
    a0 = [ (b[0] - a[0]), (b[1] - a[1]) ]
    t = fsolve( cycloidEqnY , 1 , a0)
    return math.atan( ( - math.sin(t))/( 1 - math.cos(t)))/math.pi

def main():
    toRender = { "line":1, "circle":1, "parabola":0, "cycloid":1,"random":1, "rl": 0}
    
    if (len(sys.argv) == 2):
        #read actions from file
        global env4list
        #toRender["rl"] = 1
        #fin = open(sys.argv[1],"r")
        #line = fin.readline()
        env4list = np.load(sys.argv[1])
        env4list = smooth(env4list)
        toRender["rl"] = 1

        #fin.close()

    global gViewer;
    gViewer = rendering.Viewer(600, 600)
    saveVideo = True

    global env0, env0theta, env0done
    if toRender["random"]:
        env0 = bc.BrachistochroneEnv("random", gViewer, (0,0,0))
        if saveVideo:
         from gym.wrappers.monitor import Monitor
         env0 = Monitor(env0, './video-test', force=True)
         
        env0.reset()
        env0theta = 0
        env0done = False;
        env0.score_label.x  = gViewer.width-150
        env0.score_label.y  = gViewer.height-10  
    if toRender["line"]:
        global env1, env1theta, env1done
        env1 = bc.BrachistochroneEnv("line", gViewer, (1,0,0))
        if toRender["random"]:
            env1.setStartPosition(env0.start_position)
        env1done = False;
        env1theta = math.atan( (env1.goal_position[1]- env1.start_position[1]) / (env1.goal_position[0]- env1.start_position[0]) )/(math.pi)
        env1.reset()
        env1.score_label.x  = gViewer.width-150
        env1.score_label.y  = gViewer.height-25

        
    if toRender["circle"]:
        global env2, env2theta, env2done
        env2 = bc.BrachistochroneEnv("circle", gViewer, (0,0,1))
        if toRender["random"]:
            env2.setStartPosition(env0.start_position)
        env2done = False;
        env2theta = 2*math.atan( (env2.goal_position[1]- env2.start_position[1]) / (env2.goal_position[0]- env2.start_position[0]) )/(math.pi)
        env2.reset()
        env2.score_label.x  = gViewer.width-150
        env2.score_label.y  = gViewer.height-40
    
    if toRender["cycloid"]:
        global env3, env3theta, env3done, R_cycloid, T_Cycloid
        env3 = bc.BrachistochroneEnv("cycloid", gViewer, (0,0.75,0.25))
        if toRender["random"]:
            env3.setStartPosition(env0.start_position)
        R_cycloid, T_Cycloid = solveCycloidInit(env3.start_position, env3.goal_position)
        env3theta = 2*math.atan( (env3.goal_position[1]- env3.start_position[1]) / (env3.goal_position[0]- env3.start_position[0]) )/(math.pi)
        env3done = False;
        env3.reset()
        env3.score_label.x  = gViewer.width-150
        env3.score_label.y  = gViewer.height-55
    if toRender["rl"]:
        global env4, env4theta, env4done
        env4 = bc.BrachistochroneEnv("RL Agent", gViewer, (1,0.5,0))
        env4.reset()
        env4theta = 0
        env4done = False;
        env4.score_label.x  = gViewer.width-150
        env4.score_label.y  = gViewer.height-70        
        
    numsteps = 1000
    for i in range(numsteps):        
        
        toRender["random"] and env0.render()     
        toRender["line"] and env1.render()
        toRender["circle"] and env2.render()
        toRender["cycloid"] and env3.render()
        toRender["rl"] and env4.render()
        

        if toRender["random"] and not env0done:
            env0theta = env0.action_space.sample()
            _,_,env0done,_ = env0.step(np.float32(env0theta))
        if toRender["line"] and not env1done:
            _,_,env1done,_ = env1.step(np.float32([env1theta]))
        if toRender["circle"] and not env2done:
            _,_,env2done,_ = env2.step(np.float32([env2theta]))
            env2theta = 2*math.atan( (env2.goal_position[1]- env2.state[1]) / (env2.goal_position[0]- env2.state[0]) )/math.pi
        if toRender["cycloid"] and not env3done:
            _,_,env3done,_ = env3.step(np.float32([env3theta]))
            env3theta = solveCycloid( env3.start_position,  [env3.state[0], env3.state[1]])
        """
        if toRender["rl"] and not env5done:
            line = fin.readline()
            if line:
                env0theta = [float(line)]
                _,_,env0done,_ = env5.step(np.float32([env5theta]))
            else:
                env0done = True
        """
        if toRender["rl"] and not env4done:
            if i >= len(env4list):
                continue
            env4theta = env4list[i]
            _,_,env4done,_ = env4.step(np.float32([env4theta]))
            
    toRender["random"] and env0.close()
    toRender["line"] and env1.close()
    toRender["circle"] and env2.close()
    toRender["cycloid"] and env3.close()
    if toRender["rl"]:
        pts = env4.path
        print(pts)
        coeffs = polyfit(pts)
        env4.close()
    return
    
if __name__ == '__main__':
    main()
