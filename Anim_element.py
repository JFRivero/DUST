#!/usr/bin/env python3

"""
Return an animation of the element motion given a set of coordinates e.
@Author: Juan F Rivero-Rodriguez (UKAEA)
"""

import os
import sys
import importlib as imp

import numpy as np
from scipy import signal

import matplotlib.pyplot as plt;
import matplotlib.image as mpimg
from matplotlib.animation import FuncAnimation
from matplotlib.animation import ArtistAnimation
from matplotlib import gridspec

import plot_element

plt.ion()

plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)


def Anim3D_element(t,e,S,a,b,l,
                   dpi = 150, vmax = 300, time_interval = 200,
                   save=True, path='./'):                                                                                                    


    
    def update(i):

        # ax.collections = []
        # ax.lines = []
        ax.cla()
        
        plot_element.plot_3D(S,e[:,i],a,b,l,ax)
        ax.set_title("t = {:.3f} s".format(t[i]),fontsize=20)

    def init_func():
        plot_element.plot_3D(S,e[:,0],a,b,l,ax)
        ax.set_title("t = {:.3f} s".format(t[0]),fontsize=20)


    plt.ioff()

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    anim = FuncAnimation(fig, update, frames=t.size,
                         interval=time_interval, init_func = init_func,
                         repeat=True, repeat_delay=time_interval,blit=False)
    if save:
        index = 1
        while os.path.isfile(path+"Anim_{}.gif".format(index)):
            index += 1
        print("Saving " +path+ "Anim_{}.gif".format(index))
        anim.save(path+"Anim_{}.gif".format(index),dpi=dpi, writer="imagemagick")
        plt.close()
    else:
        plt.show()

    plt.ion()

if __name__ == "__main__":
    print("hello")
    # plot_traces_and_remap(47132, 0.25, 0.36, 59, vid=vid)
