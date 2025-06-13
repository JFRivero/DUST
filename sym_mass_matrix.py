"""
Derive Yakoub shape function from polynomials coefficients.
"""
import numpy as np

import sympy as sym

import inspect

import sympy2numpy

import sym_shape_fun

x = sym.symbols("x")
y = sym.symbols("y")
z = sym.symbols("z")

l = sym.symbols("l")
a = sym.symbols("a")
b = sym.symbols("b")

xi = sym.symbols("xi")
eta = sym.symbols("eta")
zeta = sym.symbols("zeta")

rho = sym.symbols("rho")

S = sym_shape_fun.get()

S = S.subs({x: xi * l, y: eta * l, z: zeta * l})

M = sym.integrate(l**3*rho*S.T@S,(xi,0,1))
M = sym.integrate(M,(eta,-a/2/l,a/2/l))
M = sym.integrate(M,(zeta,-b/2/l,b/2/l))

M = sym.simplify(M)

def get():
    return M

def generate_script():

    sympy2numpy.generate_lambdified_script("yakoub_mass_matrix.py", M)

if __name__ == "__main__":
    generate_script()

