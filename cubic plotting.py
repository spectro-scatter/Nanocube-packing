# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 10:24:05 2021

@author: shunyu
"""

from scipy.spatial.transform import Rotation
import numpy as np
import ipyvolume as ipv
import matplotlib.pyplot as plt
from numba import jit
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

R = 5.15
p = 2*np.pi/30

Rot_step = []

for omiga in np.linspace(0, 45, num=5):
    for delta in np.linspace(0, 45, num=5):
        for theta in np.linspace(0, 45, num=5):
            Rot_step.append([omiga, delta, theta])

Vector = np.array([[-1, -1, -1],
                  [1, -1, -1 ],
                  [1, 1, -1],
                  [-1, 1, -1],
                  [-1, -1, 1],
                  [1, -1, 1 ],
                  [1, 1, 1],
                  [-1, 1, 1]])

RVector = []

for Rot in Rot_step:
    r = Rotation.from_euler('xyz', Rot, degrees=True)
    rVector = r.apply(Vector)
    RVector.append(rVector)

for count, Z in enumerate(RVector):
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter3D(Z[:, 0], Z[:, 1], Z[:, 2])
    
    verts = [[Z[0],Z[1],Z[2],Z[3]],
     [Z[4],Z[5],Z[6],Z[7]], 
     [Z[0],Z[1],Z[5],Z[4]], 
     [Z[2],Z[3],Z[7],Z[6]], 
     [Z[1],Z[2],Z[6],Z[5]],
     [Z[4],Z[7],Z[3],Z[0]]]
    
    # plot sides
    ax.add_collection3d(Poly3DCollection(verts, 
     facecolors='cyan', linewidths=1, edgecolors='b', alpha=.25))
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim3d(-2, 2)
    ax.set_ylim3d(-2, 2)
    ax.set_zlim3d(-2, 2)
    ax.set_title('Rotation {} {}'.format(count+1, Rot_step[count]))
    
    plt.show()