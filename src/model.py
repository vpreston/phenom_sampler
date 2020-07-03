#!/usr/bin/python

''' Defines the advection-diffusion model for a rectangular domain with 4 periodic boundaries '''

import os
import yaml
import numpy as np
import matplotlib.pyplot as plt

class Environment():
    ''' The simulated area '''
    def __init__(self, config_file):
        ''' initialize the environment '''
        with open(config_file) as file:
            params = yaml.load(file) #creates a dictionary of parameters which describe the environment from a yaml
        
        self.Lx = params['Lx'] #length of environment (m)
        self.Ly = params['Ly'] #width/height of environment (m)
        self.dx = params['dx'] #length discretization (m)
        self.dy = params['dy'] #width/height discretization (m)
        self.T = params['T'] #total time to simulate (s)
        self.t = 0 #current time
        self.dt = params['dt'] #time discretization(s)
        self.NI = int(self.Lx/self.dx)+1 #number of discrete cells in length
        self.NJ = int(self.Ly/self.dy)+1 #number of discrete cells in width/height

        self.X = np.linspace(0, self.Lx, self.NI)
        self.Y = np.linspace(0, self.Ly, self.NJ)

        self.world = np.ones(((self.NI)*(self.NJ),1)) #make the world an array of NI+2 x NJ+2 (add buffer around the world)
        self.u = np.ones_like(self.world) #set the velocity field as an array of NI x NJ
        self.v = np.ones_like(self.world) #set the velocity field as an array of NI x NJ
        self.sources = self.extract_sources(params['sources'])
        self.ufunc, self.vfunc = self.extract_velocity(params['velocity'])

        np.random.seed(params['seed']) #makes random behavior predictable, in order to recreate simulations
    
    def get_mesh(self):
        ''' helper function to return a meshgrid of points for the environment '''
        X, Y = np.meshgrid(np.linspace(0, self.Lx, self.NI), np.linspace(0, self.Ly, self.NJ), indexing='xy')
        return X, Y
    
    def get_xy(self, i):
        ''' from an index coordinate for the world grid, return the true value coordinate '''
        x = self.X[int((i)%self.NI)]
        y = self.Y[int((i)/self.NI)]
        return x, y
    
    def extract_sources(self, sfile):
        ''' helper function to define tracer source locations in the environment '''
        sources = np.zeros_like(self.world)
        for source, value in sfile.items():
            cx = value['lx']
            cy = value['ly']
            rx = value['rx']
            ry = value['ry']
            rate = value['rate']
            for i in range(0, self.NI*self.NJ):
                x, y = self.get_xy(i)
                if x < cx+rx and x > cx-rx and y < cy+ry and y > cy-ry:
                    sources[i] = rate
        return sources
    
    def extract_velocity(self, vfile):
        ''''helper function to define the velocity field '''
        
        def uvel(x, y, t):
            if vfile['u_field']['type'] == 'uniform':
                return vfile['u_field']['coeffs'][0] + 0.0001*int(vfile['corrupt'])*np.random.normal()
            elif vfile['u_field']['type'] == 'linear':
                return vfile['u_field']['coeffs'][0] * y + vfile['u_field']['coeffs'][1] + 0.0001*int(vfile['corrupt'])*np.random.normal()
            elif vfile['u_field']['type'] == 'quadratic':
                return vfile['u_field']['coeffs'][0] * y**2 + vfile['u_field']['coeffs'][1] * y + vfile['u_field']['coeffs'][2] + 0.0001*int(vfile['corrupt'])*np.random.normal()
            elif vfile['u_field']['type'] == 'periodic':
                return vfile['u_field']['coeffs'][0] * np.sin(np.pi*t/vfile['u_field']['coeffs'][1]) + 0.0001*int(vfile['corrupt'])*np.random.normal()
            else:
                return 0.01*int(vfile['corrupt'])*np.random.normal()
        
        def vvel(x, y, t):
            if vfile['v_field']['type'] == 'uniform':
                return vfile['v_field']['coeffs'][0] + 0.0001*int(vfile['corrupt'])*np.random.normal()
            elif vfile['v_field']['type'] == 'linear':
                return vfile['v_field']['coeffs'][0] * x + vfile['v_field']['coeffs'][1] + 0.0001*int(vfile['corrupt'])*np.random.normal()
            elif vfile['v_field']['type'] == 'quadratic':
                return vfile['v_field']['coeffs'][0] * x**2 + vfile['v_field']['coeffs'][1] * x + vfile['v_field']['coeffs'][2] + 0.0001*int(vfile['corrupt'])*np.random.normal()
            elif vfile['v_field']['type'] == 'periodic':
                return vfile['v_field']['coeffs'][0] * np.sin(np.pi*t/vfile['v_field']['coeffs'][1]) + 0.0001*int(vfile['corrupt'])*np.random.normal()
            else:
                return 0.0001*int(vfile['corrupt'])*np.random.normal()
        
        return uvel, vvel

    def apply_initial_condition(self, initial_state):
        ''' applies either a scalar value to the entire world field or applies a vector field to the world '''
        self.world = initial_state
    
        
    def get_velocity_field(self):
        ''' when called, computes a non-divergent velocity field for the environment.
            Notably applies boundary conditions under the assumption that the world is a channel '''
        for i in range(0, self.NI*self.NJ):
            x, y = self.get_xy(i)
            self.u[i] = self.ufunc(x, y, self.t)
            self.v[i] = self.vfunc(x, y, self.t)

        self.un = (self.u - np.abs(self.u))/2.0
        self.up = (self.u + np.abs(self.u))/2.0
        self.vn = (self.v - np.abs(self.v))/2.0
        self.vp = (self.v + np.abs(self.v))/2.0
    
    def compute_flux(self, i, world):
        ''' computes all of the fluxes in the world based on velocity and state params '''
        x, y = self.get_xy(i)
        if x > 0 and x < self.Lx and y > 0 and y < self.Ly:
            Fe = self.up[i]*world[i] + self.un[i+1]*world[i+1]
            Fw = self.up[i-1]*world[i-1] + self.un[i]*world[i]
            Fn = self.vp[i]*world[i] + self.vn[i+self.NI]*world[i+self.NI]
            Fs = self.vp[i-self.NI]*world[i-self.NI] + self.vn[i]*world[i]
            return Fe, Fw, Fn, Fs

        if x == self.Lx:
            Fe = self.up[i]*world[i] + self.un[i-self.NI+1]*world[i-self.NI+1]
            Fw = self.up[i-1]*world[i-1] + self.un[i]*world[i]
        elif x == 0:
            Fe = self.up[i]*world[i] + self.un[i+1]*world[i+1]
            Fw = self.up[i-1+self.NI]*world[i-1+self.NI] + self.un[i]*world[i]
        else:
            Fe = self.up[i]*world[i] + self.un[i+1]*world[i+1]
            Fw = self.up[i-1]*world[i-1] + self.un[i]*world[i]
        if y == self.Ly:
            Fn = self.vp[i]*world[i] + self.vn[i%self.NI]*world[i%self.NI]
            Fs = self.vp[i-self.NI]*world[i-self.NI] + self.vn[i]*world[i]
        elif y == 0:
            Fn = self.vp[i]*world[i] + self.vn[i+self.NI]*world[i+self.NI]
            Fs = self.vp[i-self.NI]*world[i-self.NI] + self.vn[i]*world[i]
        else:
            Fn = self.vp[i]*world[i] + self.vn[i+self.NI]*world[i+self.NI]
            Fs = self.vp[i-self.NI]*world[i-self.NI] + self.vn[i]*world[i]
        return Fe, Fw, Fn, Fs
        
    def step(self):
        ''' perform a forward euler step '''
        self.get_velocity_field()
        world = self.world
        for i in range(0, self.NI*self.NJ):
            Fe, Fw, Fn, Fs = self.compute_flux(i, world)
            self.world[i] = self.world[i] - (self.dt/self.dx*(Fe - Fw) + self.dt/self.dy*(Fn - Fs)) + self.dt*self.sources[i]
    
    def simulate(self):
        ''' perform forward euler '''
        history = np.zeros(shape=(self.world.shape[0], int(self.T/self.dt)))
        history[:,0] = self.world[:,0]
        self.t = self.dt
        n = 1

        while self.t <= self.T:
            if n % 100 == 0:
                print('Now simulating time = ', np.round(self.t), ' seconds')
            self.step()
            history[:,n] = self.world[:,0]
            self.t += self.dt
            n += 1
        
        return history

if __name__ == '__main__':
    channel = Environment('../config/simple_flow.yaml')
    history = channel.simulate()
    X, Y = channel.get_mesh()
    world = channel.world.reshape((channel.NI, channel.NJ))
    plt.contourf(X, Y, world)
    plt.show()

    np.savez('../data/history.npz', history)
