#!/usr/bin/env python3
"""
Main script to integrate the dynamic of the Yakoub 2-noded element 
"""

import numpy as np

import scipy.integrate as sp_int

import elastic_force

import yakoub_mass_matrix

import constrains

import importlib as imp
imp.reload(elastic_force)

"""
Material properties
"""
E = 210e9 # Pa = N/m**2
poisson = 0.30 
lam = E*poisson/(1+poisson)/(1-2*poisson)
G = E/(2*(1+poisson))
rho = 7850 #kg/m**3

"""
Element dimentions
"""
l = 1 #m
a = 0.02 #m
b = 0.02 #m

"""
Initial conditions
"""
r0 = np.array([0,0,0]).T
r0_x = np.array([1,0,0]).T
r0_y = np.array([0,1,0]).T
r0_z = np.array([0,0,1]).T
r1 = np.array([l,0,0]).T
r1_x = np.array([1,0,0]).T
r1_y = np.array([0,1,0]).T
r1_z = np.array([0,0,1]).T

dr0 = np.array([0,0,0]).T
dr0_x = np.array([0,0,0]).T
dr0_y = np.array([0,0,0]).T
dr0_z = np.array([0,0,0]).T
dr1 = np.array([0,1,0]).T
dr1_x = np.array([0,0,0]).T
dr1_y = np.array([0,0,0]).T
dr1_z = np.array([0,0,0]).T

# Undeformed shape
e0 = np.hstack((r0,r0_x,r0_y,r0_z,r1,r1_x,r1_y,r1_z))

# Initial conditions: start from the undeformed shape with non-zero velocity at node 1
y0 = np.hstack((dr0,dr0_x,dr0_y,dr0_z,dr1,dr1_x,dr1_y,dr1_z,
                r0,r0_x,r0_y,r0_z,r1,r1_x,r1_y,r1_z)) 

"""
Integration parameters
"""
t0 = 0
tf = 1
t_eval = np.linspace(t0,tf,100)

"""
ODE definition
"""

Fe = elastic_force.Fe(a,b,l,e0,lam,G)
# Fe.eval_K1()
# Fe.eval_CK2()
Fe.load_K1()
Fe.load_CK2()

# def mass_matrix():
#     M = yakoub_mass_matrix.eval(rho, b, a, l)
#     Ce = constrains.eval_Ce()
    
#     return np.vstack((np.hstack((M,Ce.T)),np.hstack((Ce,np.zeros((Ce.shape[0],Ce.shape[0]))))))

# def fun(t,y):

#     Qd = constrains.eval_Qd() 
#     Q =  ## MISSING
#     A = mass_matrix()
#     b = np.hstack((Q,Qd))
#     X = np.linalg.solve(A, b)

#     return np.vstack([X[0:y.size], y[:y.size]])


# """
# Integration
# """

# sol = sp_int.solve_ivp(fun, (t0,tf), y0, method='LSODA', t_eval=t_eval)

