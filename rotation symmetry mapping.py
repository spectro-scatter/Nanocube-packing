# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 19:38:25 2021

@author: shunyu
"""

from scipy.spatial.transform import Rotation
import numpy as np
import ipyvolume as ipv
import matplotlib.pyplot as plt
from numba import jit
from tqdm import tqdm

##### R is the half length of a nanocube edge
R = 5.15
#####################################
#  D is 2 times of lattice constant
#####################################
D = 45.6
#### p is the corresponding reciprocal frequency to create the periodic structure in 
p = 2*np.pi/D


############## for a specific rotation #############
Rot_step = [[35.3, 30, 54.7]]

'''
############## for a series of ratation #############
Rot_step = []
for omiga in np.linspace(0, 90, num=9):
    for delta in np.linspace(0, 90, num=9):
        for theta in np.linspace(0, 90, num=9):
            Rot_step.append([omiga, delta, theta])
'''

Vector = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
RVector = []
for Rot in Rot_step:
    r = Rotation.from_euler('xyz', Rot, degrees=True)
    rVector = r.apply(Vector)
    RVector.append(rVector)


########### define reciprocal space ################

### defube wavevector range
axis = np.linspace(-3,3,301)

### constructe 3D reciprocal space
x, y, z = np.meshgrid(axis, axis, axis)

### calculate wavevector in spherical symmetry
q = np.sqrt(x**2+y**2+z**2)

### get 1D wavevector
hist, bin_edges = np.histogram(q, bins= 120, range = (0.1, 2.75))
qAxis = (bin_edges[1:] + bin_edges[:-1]) / 2.0


########### define form factor and structure factor ################
### define function to get the form factor of a cube
@jit(nopython=True)
def ff(scale, X, Y, Z, R):
    return scale*(8*np.sinc((X)*R/np.pi)*np.sinc((Y)*R/np.pi)*np.sinc((Z)*R/np.pi))**2

### define the Lorentzian function broadened FCC packing, which is a BCC in reciprocal space
@jit(nopython=True)
def fcc(p, o):
    return 4/(1+(np.abs(((x+y)/2)%(2*p)-p)-p)**2/o**2)/(1+(np.abs(((y+z)/2)%(2*p)-p)-p)**2/o**2)/(1+(np.abs(((x+z)/2)%(2*p)-p)-p)**2/o**2)

### calculate the structure factor of FCC
s_fcc = fcc(p, 0.01)

### create an array to collect all 1D Scattering profile of FCC-nanocube system
A_fcc = []

'''
### define the Lorentzian function broadened BCC packing, which is a FCC in reciprocal space
@jit(nopython=True)
def bcc(scale, X, Y, Z, R, p, o):
    return 2/(1+(np.abs(((x+y+z)/2)%(2*p)-p)-p)**2/o**2)/(1+(np.abs(((x-y+z)/2)%(2*p)-p)-p)**2/o**2)/(1+(np.abs(((-x+y+z)/2)%(2*p)-p)-p)**2/o**2)
   
### calculate the structure factor of BCC
s_bcc = bcc(p, 0.01)

### create an array to collect all 1D Scattering profile of BCC-nanocube system
A_bcc = []

### define the Lorentzian function broadened SC packing, which is a SC in reciprocal space
@jit(nopython=True)
def sc(p, o):
    return 1/(1+(np.abs((x)%(2*p)-p)-p)**2/o**2)/(1+(np.abs((y)%(2*p)-p)-p)**2/o**2)/(1+(np.abs((z)%(2*p)-p)-p)**2/o**2)

### calculate the structure factor of SC
s_sc = sc(p, 0.02)

### create an array to collect all 1D Scattering profile of FCC-nanocube system
A_sc = []

'''
for counter, vector_new in enumerate(tqdm(RVector)):

    
    X = vector_new[0][0]*x+vector_new[0][1]*y+vector_new[0][2]*z
    Y = vector_new[1][0]*x+vector_new[1][1]*y+vector_new[1][2]*z
    Z = vector_new[2][0]*x+vector_new[2][1]*y+vector_new[2][2]*z
    

    f_fcc = ff(1, X, Y, Z, R)*s_fcc
    #f_sc = ff(1, X, Y, Z, R)*s_sc
    #f_bcc = ff(1, R, 2*np.pi/35, 0.02)
    
    r_fcc, bin_number = np.histogram(q, bins = 120, weights=f_fcc, range = (0.1, 2.75))
    r_fcc = r_fcc/hist
    A_fcc.append(r_fcc)
    #r_sc, bin_number = np.histogram(q, bins = 120, weights=f_sc, range = (0.1, 2.75))
    #r_sc = r_sc/hist
    #A_sc.append(r_sc)
    #r_bcc, bin_number = np.histogram(q, bins = 120, weights=f_bcc, range = (0.1, 2.75))
    #r_bcc = r_bcc/hist
    #A_bcc.append(r_bcc)
