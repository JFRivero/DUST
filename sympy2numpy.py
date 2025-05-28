"""
Converts symbolic expressions optimised with cse to numpy expressions and generates a .py file with the result. 
"""
import numpy as np

import sympy as sym

import inspect

import os

def sympy_to_numpy_code_generic(sympy_expr):
    """Converts SymPy expressions to Numpy."""
    code = str(sympy_expr)
    replacements = {
        '**': '**',
        'sqrt': 'np.sqrt',
        'sin': 'np.sin',
        'cos': 'np.cos',
        'tan': 'np.tan',
        'exp': 'np.exp',
        'log': 'np.log',
        'Abs': 'np.abs',
        'asin': 'np.arcsin',
        'acos': 'np.arccos',
        'atan': 'np.arctan',
        'atan2': 'np.arctan2',
        'floor': 'np.floor',
        'ceiling': 'np.ceil',
        'sign': 'np.sign',
        'conjugate': 'np.conjugate',
        're': 'np.real',
        'im': 'np.imag'
        # Add more substitution as they become necessary
    }
    for sympy_func, numpy_func in replacements.items():
        code = code.replace(sympy_func, numpy_func)
    return code

def generate_python_function(filename, sym_fun):
    """
    Generates a .py file with a function to evaluate the expressions resulting from sympy.cse.

    Args:
        filename (str): Name of the .py file to create.
        cse_output (tuple): Tuple output from sympy.cse (substitutions, expressions).
        variables (tuple): Tuple of input symbolic variables in the desired order
                           for the arguments of the generated function.
    """
    variables = sym_fun.free_symbols
    cse_output = sym.cse(sym_fun)
    substitutions, expressions = cse_output

    with open(filename, 'w', encoding='utf-8') as f:
        f.write("import numpy as np\n\n")
        f.write(f"def eval({', '.join(map(str, variables))}):\n")
        f.write("    \"\"\"Evaluates the generated expressions.\n\n")
        f.write("    Args:\n")
        for var in variables:
            f.write(f"        {var}: Value for the symbolic variable {var}.\n")
        f.write("    \n")
        f.write("    Returns:\n")
        f.write("        numpy.ndarray: Array of evaluated expression values.\n")
        f.write("    \"\"\"\n")

        # Calculate the substitutions
        for sub, val in substitutions:
            f.write(f"    {sub} = {sympy_to_numpy_code_generic(val)}\n")

        # Evaluate the expressions using substitutions
        f.write("    results = np.array([\n")
        for expr in expressions:
            f.write(f"        {sympy_to_numpy_code_generic(expr)},\n")
        f.write("    ])\n")
        f.write("    return results\n")


def generate_lambdified_script(filename, sym_fun, var=None, funname = "eval", addtofile=False):
    if var==None:
        var = list(sym_fun.free_symbols)

    lam_fun = sym.lambdify(var,sym_fun,modules="numpy",cse=True)

    openfile = "a" if addtofile else "w"

    with open(filename, openfile, encoding='utf-8') as f:
        f.write("\"\"\"\n")
        f.write("This script was automatically created with sympy2numpy.\n")
        f.write("\"\"\"\n\n") 
        f.write("from numpy import *\n\n")
        f.write(inspect.getsource(lam_fun).replace("_lambdifygenerated",funname))
        f.write("\n")
