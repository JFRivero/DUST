#!/usr/bin/env python3 
"""
Derive Elastic forces for Yakoub 2-nodes elements
"""
import numpy as np

import yakoub_shape_function as shape_fun #generated with sym_shape_fun

import matplotlib.pyplot as plt; plt.ion()

def plot_3D(S,e,a,b,l,ax=None):

    if ax==None:
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    XX = np.vstack([S(x,0,0,l)@e for x in np.linspace(0,l,100)])
    
    ax.plot(XX[:,0],XX[:,1],XX[:,2],color="blue")


    for x in np.linspace(0,l,10):
        XX = np.vstack([S(x,-a/2,-b/2,l)@e,
                        S(x,-a/2,b/2,l)@e,
                        S(x,a/2,b/2,l)@e,
                        S(x,a/2,-b/2,l)@e,
                        S(x,-a/2,-b/2,l)@e])
                        
        ax.plot(XX[:,0],XX[:,1],XX[:,2],color="red")

    ax.axis("equal")


def plot_2D(S,e,a,b,l,ax=None, axes=(0,1)):

    if ax==None:
        fig, ax = plt.subplots()

    XX = np.vstack([S(x,0,0,l)@e for x in np.linspace(0,l,100)])
    
    ax.plot(XX[:,axes[0]],XX[:,axes[1]],color="blue")


    for x in np.linspace(0,l,10):
        XX = np.vstack([S(x,-a/2,-b/2,l)@e,
                        S(x,-a/2,b/2,l)@e,
                        S(x,a/2,b/2,l)@e,
                        S(x,a/2,-b/2,l)@e,
                        S(x,-a/2,-b/2,l)@e])
                        
        ax.plot(XX[:,axes[0]],XX[:,axes[1]],color="red")

    ax.axis("equal")
