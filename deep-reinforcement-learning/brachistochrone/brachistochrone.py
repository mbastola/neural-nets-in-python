"""
Manil Bastola
The Brachistochrone Problem

Used OpenAI GYM's Mountain Car Continuous for reference:
http://incompleteideas.net/sutton/MountainCar/MountainCar1.cp
permalink: https://perma.cc/6Z2N-PFWC
"""

import math

import numpy as np
import random
import gym
from gym import spaces
from gym.utils import seeding
from gym.envs.classic_control import rendering
import pyglet

class DrawText:
    def __init__(self, label:pyglet.text.Label):
        self.label=label
    def render(self):
        self.label.draw()

        
class BrachistochroneEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 24
    }

    def __init__(self, whoami, viewer = None, color=(0.5,0.5,0.5)):
        self.name = whoami
        self.color = color
        
        self.max_theta = 1
        self.min_theta = -1

        self.min_position = [0.0, 0.0]
        self.max_position = [1.0, 1.0]

        self.min_speed = 0;
        self.max_speed = 1e2;
        
        #self.start_position = [0.05,0.95]
        self.start_position = [0.2,0.8]
        self.goal_position = [0.8,0.2]
        #self.goal_position = [0.95,0.05]
        self.start_velocity = 0.0
        
        self.time_step = 0.08
        self.pos_step = 0.01
        self.particleRadius = 10
        
        self.total_time = 0
        self.gravity = 0.0098

        #self.path = [[0.0,self.start_position[1]]]
        self.path = [self.start_position]
        
        self.low_state = np.array([self.min_position[0], self.min_position[1], self.min_speed])
        self.high_state = np.array([self.max_position[0], self.max_position[1], self.max_speed])

        self.action_space = spaces.Box(low=self.min_theta, high=self.max_theta, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=self.low_state,high=self.high_state , dtype=np.float32)

        self.viewer = viewer
        self.seed()
        self.renderInit()
        self.reset()
        
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def getTimeTaken(self):
        return self.total_time

    def setStartPosition(self, pos):
        self.start_position = pos

    def setGoalPosition(self, pos):
        self.goal_position = pos

    def getStartPosition(self):
        return self.start_position

    def getGoalPosition(self):
        return self.goal_position
        
    def getName(self):
        return self.name

    #time steps 
    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        
        theta = math.pi*action[0]
        
        p0 = [self.state[0],self.state[1]]
        v0 = self.state[2]

        d = ( v0 * self.time_step ) - ( 0.5 * self.gravity * math.sin(theta) * self.time_step * self.time_step )
        
        p1 = [ p0[0] + d * math.cos(theta), p0[1] + d * math.sin(theta) ]

        descriminant = (v0 * v0) - (2 * self.gravity * ( p1[1] - p0[1] ))

        if descriminant < 0:
            descriminant = 0  #can occur due to floating point precision
        v1 = math.sqrt( descriminant )

        self.total_time += self.time_step
        
        self.path.append(p0)
        
        reward = 0
        
        done = self.isGoal();
        if done:
            reward += 1000
            print(self.name+": "+str(self.total_time)) 
        walled = self.isWall()
        if walled or v1 <= 0:
            reward = -100
            done = True
        #dist = abs(self.goal_position[0] - p1[0]) + abs(self.goal_position[1] - p1[1])
        dist = math.hypot((self.goal_position[0] - p1[0]),(self.goal_position[1] - p1[1]))
        #reward -= 1
        reward -= self.time_step
        reward -= dist
        
        self.state = np.array([p1[0], p1[1], v1])
        return self.state, reward, done, {}

    
    #position steps that minimzes time
    def pos_step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        theta = action[0]
        
        p0 = [self.state[0],self.state[1]]
        v0 = self.state[2]

        p1 = [ p0[0] + self.pos_step * math.cos(theta), p0[1] + self.pos_step * math.sin(theta)]

        descriminant = (v0 * v0) - (2 * self.gravity * ( p1[1] - p0[1] ))

        if descriminant < 0:
            descriminant = 0  #can occur due to floating point precision
        v1 = math.sqrt( descriminant )
        
        t = self.pos_step * ( v0 - v1 ) / ( self.gravity * ( p1[1] - p0[1] ) )
        if t < 0:
            t = self.pos_step * ( v0 + v1 ) / ( self.gravity * ( p1[1] - p0[1] ) )

        self.total_time += t
        
        p1[0] = np.clip(p1[0], self.min_position[0], self.max_position[0])
        p1[1] = np.clip(p1[1], self.min_position[1], self.max_position[1])

        self.path.append(p0)
        
        done = self.isGoal();

        reward = 0
        if done:
            reward = 100.0
        walled = self.isWall()
        if walled:
            reward = 0
            done = True

        reward -= (10*t - (abs(self.goal_position[0] - p1[0]) + abs(self.goal_position[1] - p1[1])))
        #minimize time reaching the goal

        self.old = p0;
        
        self.state = np.array([p1[0], p1[1], v1])
        return self.state, reward, done, {}
    
    def reset(self):
        self.total_time = 0
        #self.path = [[0.0,self.start_position[1]]]
        self.path = [self.start_position]
        self.state = np.array([self.start_position[0], self.start_position[1], self.start_velocity]);
        self.score_label.text = ""
        self.render()
        #self.renderInit()
        return self.state

    def isGoal(self):
        position = [self.state[0], self.state[1]]
        #
        eps = 2 * self.time_step * self.particleRadius
        return (math.hypot((position[0]-self.goal_position[0]),(position[1]-self.goal_position[1])) <= eps*self.pos_step)

    def isWall(self):
        position = [self.state[0], self.state[1]]
        return not ((position[0] < self.max_position[0]) and (position[0] >= self.min_position[0]) and (position[1] < self.max_position[1]) and (position[1] >= self.min_position[1]))

    def renderInit(self, mode='human'):
        if self.viewer is None:
            self.viewer = rendering.Viewer(600, 600)
        
        if self.color is None:
            self.color = (0.5,0.5,0.5) 
            
        screen = [self.viewer.width, self.viewer.height]
        world = [(self.max_position[0] - self.min_position[0]), (self.max_position[1] - self.min_position[1])] 
        self.scale = [(screen[0]/world[0]), (screen[1]/world[1])] 
        
        flagx = (self.goal_position[0])*self.scale[0]
        flagy1 = (self.goal_position[1])*self.scale[1]
        flagy2 = flagy1 + 50
        flagpole = rendering.Line((flagx, flagy1), (flagx, flagy2))
        self.viewer.add_geom(flagpole)
        
        flag = rendering.FilledPolygon([(flagx, flagy2), (flagx, flagy2-10), (flagx+25, flagy2-5)])
        flag.set_color(.8,.8,0)
        self.viewer.add_geom(flag)
        
        particle = rendering.make_circle(self.particleRadius)
        self.particletrans = rendering.Transform()

        self.particletrans.set_translation(-self.particleRadius,-self.particleRadius)
        self.particletrans.set_rotation(0)

        particle.add_attr(self.particletrans)
        particle.set_color(self.color[0], self.color[1], self.color[2])
        self.viewer.add_geom(particle)
        
        self.score_label = pyglet.text.Label("", font_size=10,x=(self.viewer.width - 150), y=(self.viewer.height-10),anchor_x='left', anchor_y='center',color=(int(255*self.color[0]),int(255*self.color[1]),int(255*self.color[2]), 255))
        self.viewer.add_geom(DrawText(self.score_label))
        return
        
    def render(self, mode='human'):
        pos = [ self.state[0], self.state[1] ]
        old_pos = self.path[-1]

        self.particletrans.set_translation((pos[0])*self.scale[0], (pos[1])*self.scale[1])
        self.particletrans.set_rotation(0)

        xs = np.array([old_pos[0], pos[0]])
        ys = np.array([old_pos[1], pos[1]])
        
        xys = list(zip(xs*self.scale[0], ys*self.scale[1]))
        path = rendering.make_polyline(xys)
        path.set_linewidth(2)
        path.set_color(self.color[0],self.color[1],self.color[2])
        self.viewer.add_geom(path)
        self.score_label.text = self.name+": %0.4fs" % self.total_time
        return self.viewer.render(return_rgb_array = mode=='rgb_array')
        
    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
