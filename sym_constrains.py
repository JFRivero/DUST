"""
Derive the constrain equations.
"""
import numpy as np

import sympy as sym

import inspect

import sympy2numpy

# Time
t = sym.symbols("t")

# Coordinates
x0 = sym.symbols("x0")
y0 = sym.symbols("y0")
z0 = sym.symbols("z0")

x0_x = sym.symbols("x0_x")
y0_x = sym.symbols("y0_x")
z0_x = sym.symbols("z0_x")

x0_y = sym.symbols("x0_y")
y0_y = sym.symbols("y0_y")
z0_y = sym.symbols("z0_y")

x0_z = sym.symbols("x0_z")
y0_z = sym.symbols("y0_z")
z0_z = sym.symbols("z0_z")

x1 = sym.symbols("x1")
y1 = sym.symbols("y1")
z1 = sym.symbols("z1")

x1_x = sym.symbols("x1_x")
y1_x = sym.symbols("y1_x")
z1_x = sym.symbols("z1_x")

x1_y = sym.symbols("x1_y")
y1_y = sym.symbols("y1_y")
z1_y = sym.symbols("z1_y")

x1_z = sym.symbols("x1_z")
y1_z = sym.symbols("y1_z")
z1_z = sym.symbols("z1_z")

r0 = sym.Matrix([x0, y0, z0]) # Can we define these as the symbolic variables?
r0_x = sym.Matrix([x0_x, y0_x, z0_x])
r0_y = sym.Matrix([x0_y, y0_y, z0_y])
r0_z = sym.Matrix([x0_z, y0_z, z0_z])

r1 = sym.Matrix([x1, y1, z1])
r1_x = sym.Matrix([x1_x, y1_x, z1_x])
r1_y = sym.Matrix([x1_y, y1_y, z1_y])
r1_z = sym.Matrix([x1_z, y1_z, z1_z])

e = sym.Matrix(np.concatenate((r0,r0_x,r0_y,r0_z,r1,r1_x,r1_y,r1_z)))

# First derivatives
dx0 = sym.symbols("dx0")
dy0 = sym.symbols("dy0")
dz0 = sym.symbols("dz0")

dx0_x = sym.symbols("dx0_x")
dy0_x = sym.symbols("dy0_x")
dz0_x = sym.symbols("dz0_x")

dx0_y = sym.symbols("dx0_y")
dy0_y = sym.symbols("dy0_y")
dz0_y = sym.symbols("dz0_y")

dx0_z = sym.symbols("dx0_z")
dy0_z = sym.symbols("dy0_z")
dz0_z = sym.symbols("dz0_z")

dx1 = sym.symbols("dx1")
dy1 = sym.symbols("dy1")
dz1 = sym.symbols("dz1")

dx1_x = sym.symbols("dx1_x")
dy1_x = sym.symbols("dy1_x")
dz1_x = sym.symbols("dz1_x")

dx1_y = sym.symbols("dx1_y")
dy1_y = sym.symbols("dy1_y")
dz1_y = sym.symbols("dz1_y")

dx1_z = sym.symbols("dx1_z")
dy1_z = sym.symbols("dy1_z")
dz1_z = sym.symbols("dz1_z")

dr0 = sym.Matrix([dx0, dy0, dz0])
dr0_x = sym.Matrix([dx0_x, dy0_x, dz0_x])
dr0_y = sym.Matrix([dx0_y, dy0_y, dz0_y])
dr0_z = sym.Matrix([dx0_z, dy0_z, dz0_z])

dr1 = sym.Matrix([dx1, dy1, dz1])
dr1_x = sym.Matrix([dx1_x, dy1_x, dz1_x])
dr1_y = sym.Matrix([dx1_y, dy1_y, dz1_y])
dr1_z = sym.Matrix([dx1_z, dy1_z, dz1_z])

de = sym.Matrix(np.concatenate((dr0,dr0_x,dr0_y,dr0_z,dr1,dr1_x,dr1_y,dr1_z)))

# Second derivatives
ddx0 = sym.symbols("ddx0")
ddy0 = sym.symbols("ddy0")
ddz0 = sym.symbols("ddz0")

ddx0_x = sym.symbols("ddx0_x")
ddy0_x = sym.symbols("ddy0_x")
ddz0_x = sym.symbols("ddz0_x")

ddx0_y = sym.symbols("ddx0_y")
ddy0_y = sym.symbols("ddy0_y")
ddz0_y = sym.symbols("ddz0_y")

ddx0_z = sym.symbols("ddx0_z")
ddy0_z = sym.symbols("ddy0_z")
ddz0_z = sym.symbols("ddz0_z")

ddx1 = sym.symbols("ddx1")
ddy1 = sym.symbols("ddy1")
ddz1 = sym.symbols("ddz1")

ddx1_x = sym.symbols("ddx1_x")
ddy1_x = sym.symbols("ddy1_x")
ddz1_x = sym.symbols("ddz1_x")

ddx1_y = sym.symbols("ddx1_y")
ddy1_y = sym.symbols("ddy1_y")
ddz1_y = sym.symbols("ddz1_y")

ddx1_z = sym.symbols("ddx1_z")
ddy1_z = sym.symbols("ddy1_z")
ddz1_z = sym.symbols("ddz1_z")

ddr0 = sym.Matrix([ddx0, ddy0, ddz0])
ddr0_x = sym.Matrix([ddx0_x, ddy0_x, ddz0_x])
ddr0_y = sym.Matrix([ddx0_y, ddy0_y, ddz0_y])
ddr0_z = sym.Matrix([ddx0_z, ddy0_z, ddz0_z])

ddr1 = sym.Matrix([ddx1, ddy1, ddz1])
ddr1_x = sym.Matrix([ddx1_x, ddy1_x, ddz1_x])
ddr1_y = sym.Matrix([ddx1_y, ddy1_y, ddz1_y])
ddr1_z = sym.Matrix([ddx1_z, ddy1_z, ddz1_z])

dde = sym.Matrix(np.concatenate((ddr0,ddr0_x,ddr0_y,ddr0_z,ddr1,ddr1_x,ddr1_y,ddr1_z)))

# Baumgarte parameters

a = sym.symbols("a")
b = sym.symbols("b")

# Contrains
C = sym.Matrix([r0[0],r0[1],r0[2],r0_x[1],r0_x[2],r0_y[2]])

Ce = C.jacobian(e)

dCe = Ce.diff(t)
for i in range(len(e)):
    dCe += Ce.diff(e[i])*de[i]

Ct = C.diff(t)

dCt = Ct.diff(t)
for i in range(len(e)):
    dCt += Ct.diff(e[i])*de[i]

Qd = - dCe*de + dCt

QBaum = 2*a*(Ce*e+Ct)-b**2*C

Ce = sym.simplify(Ce)
Qd = sym.simplify(Qd)
QBaum = sym.simplify(QBaum)

sympy2numpy.generate_lambdified_script("constrains.py", Ce, funname="eval_Ce")
sympy2numpy.generate_lambdified_script("constrains.py", Qd, funname="eval_Qd", addtofile=True)
variables_eval = (e, a, b)
sympy2numpy.generate_lambdified_script("constrains.py", QBaum, variables_eval,
                                       funname="eval_QBaum", addtofile=True)
