#!/usr/bin/env python3
"""
Derive Yakoub shape function from polynomials coefficients.
"""
import numpy as np

import sympy as sym

import inspect

import sympy2numpy

x = sym.symbols("x")
y = sym.symbols("y")
z = sym.symbols("z")
X = sym.Matrix([x, y, z])

l = sym.symbols("l")

xi = sym.symbols("xi")
eta = sym.symbols("eta")
zeta = sym.symbols("zeta")

S1 = [1, x, y, z, x*y, x*z, x**2, x**3]

A = sym.Matrix([np.concatenate((S1,np.zeros(16))),
                np.concatenate((np.zeros(8),S1,np.zeros(8))),
                np.concatenate((np.zeros(16),S1))])

A_x = sym.diff(A,x)
A_y = sym.diff(A,y)
A_z = sym.diff(A,z)

C = sym.Matrix([A.subs({x: 0, y: 0, z: 0}),
                A_x.subs({x: 0, y: 0, z: 0}),
                A_y.subs({x: 0, y: 0, z: 0}),
                A_z.subs({x: 0, y: 0, z: 0}),
                A.subs({x: l, y: 0, z: 0}),
                A_x.subs({x: l, y: 0, z: 0}),
                A_y.subs({x: l, y: 0, z: 0}),
                A_z.subs({x: l, y: 0, z: 0})])

S = A*C.inv()

# S = S.subs({x: xi * l, y: eta * l, z: zeta * l})

S = sym.simplify(S)

ncoord = S.shape[1]

e0 = sym.Matrix(sym.symbols([f"e0_{i}" for i in range(ncoord)]))
r0 = S@e0 #undeformed shape

J0 = r0.jacobian(X)
J0invT = J0.inv().T

Sx = S.diff(x)
Sy = S.diff(y)
Sz = S.diff(z)

dS_dr0 = [sym.Matrix([J0invT[j,:]@sym.Matrix([Sx[i,:], Sy[i,:], Sz[i,:]]) for i in range(3)])
          for j in range(3)]

dS_dr0 = [sym.simplify(dS_dr0[j]) for j in range(3)]


def get():
    return S

def generate_script():
    variables_eval = (x, y, z, l)
    sympy2numpy.generate_lambdified_script("yakoub_shape_function.py", S, variables_eval,
                                           funname="eval_S")
    variables_eval = (x, y, z, l, e0)
    sympy2numpy.generate_lambdified_script("yakoub_shape_function.py", dS_dr0[0],
                                           variables_eval,funname="eval_dS_dr0_0",
                                           addtofile=True)
    sympy2numpy.generate_lambdified_script("yakoub_shape_function.py", dS_dr0[1],
                                           variables_eval,funname="eval_dS_dr0_1",
                                           addtofile=True)
    sympy2numpy.generate_lambdified_script("yakoub_shape_function.py", dS_dr0[2],
                                           variables_eval,funname="eval_dS_dr0_2",
                                           addtofile=True)


if __name__ == "__main__":
    generate_script()
