#!/usr/bin/python

''' Processes simulated files (e.g., draw samples from query points, create graphs, etc.) '''

import os
import yaml
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt

class Sampler():
    ''' class that takes in a simulation batch and can dissect it according to various input parameters '''
    def __init__(self, config_file):
        ''' initialize the sampler '''
        with open(config_file) as file:
            params = yaml.load(file) #creates a dictionary of parameters which describe the environment from a yaml
        
        self.Lx = params['Lx'] #length of environment (m)
        self.Ly = params['Ly'] #width/height of environment (m)
        self.T = params['T'] #total simulated time (s)
        self.dx = params['dx'] #length discretization (m)
        self.dy = params['dy'] #width/height discretization (m)
        self.dt = params['dt'] #time discretization (s)

        self.NI = int(self.Lx/self.dx)+1 #number of discrete cells in length
        self.NJ = int(self.Ly/self.dy)+1 #number of discrete cells in width/height

        self.history = np.load(params['file'])['arr_0']

        self.x, self.y = np.linspace(0, self.Lx, self.NI), np.linspace(0, self.Ly, self.NJ)
        self.X, self.Y = np.meshgrid(np.linspace(0, self.Lx, self.NI), np.linspace(0, self.Ly, self.NJ), indexing='xy') #define X,Y coordinates for cells for location lookup
        self.T = np.linspace(0, self.dt, self.T)

        self.interpolator = params['interpolator']

    def query_path(self, points):
        ''' returns the observation of the phenomenon for the x, y, t coordinate(s) passed in. '''
        obs = []
        for point in points:
            historyidx = np.abs(point[2] - self.T).argmin() #find the closest time snapshot
            world = self.history[:,historyidx].reshape((self.NI, self.NJ))
            xidx = np.abs(point[0] - self.x).argmin() #find the closest simulated cell
            yidx = np.abs(point[1] - self.y).argmin() #find the closest simulated cell
            obs.append(world[xidx, yidx])
        return obs

    def query_snapshot(self, dimensions, time):
        ''' returns a uniformly sampled snapshot of a phenomenon at a given time at the fidelity specified by dimensions '''
        historyidx = np.abs(time - self.T).argmin()
        world = self.history[:,historyidx].reshape((self.NI, self.NJ))
        f = interpolate.interp2d(self.x, self.y, world, kind=self.interpolator)
        self.snapX, self.snapY = np.linspace(0, self.Lx, dimensions[0]), np.linspace(0, self.Ly, dimensions[1])
        return f(self.snapX, self.snapY)

if __name__ == '__main__':
    sampler = Sampler('../config/simple_sampler.yaml')
    path = [(0, 0, 0), (5, 1, 1), (10, 2, 2), (10, 10, 3)]
    observations = sampler.query_path(path)
    snapshots = []
    for point in path:
        snapshots.append(sampler.query_snapshot((10,10), point[2]))
    
    for point, obs, snap in zip(path, observations, snapshots):
        plt.contourf(sampler.snapX, sampler.snapY, snap, cmap='viridis', vmin=np.nanmin(sampler.history), vmax=np.nanmax(sampler.history))
        plt.scatter(point[0], point[1], c=obs, cmap='viridis', vmin=np.nanmin(sampler.history), vmax=np.nanmax(sampler.history), edgecolor='red')
        plt.show()

