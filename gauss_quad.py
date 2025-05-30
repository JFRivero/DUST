#!/usr/bin/env python3

import numpy as np
from numpy.polynomial.legendre import leggauss

def triple_integrator(func, x_limits, y_limits, z_limits, num_points_x, num_points_y, num_points_z):
    """
    Integrates a three-variable function f(x, y, z) over a parallelepiped region
    using Gaussian quadrature.

    Parameters:
    -----------
    func : callable
        The function to integrate. It must accept three arguments (x, y, z).
        Example: def func(x, y, z): return x**2 + y*z

    x_limits : tuple
        A tuple (x1, x2) defining the integration limits for x.

    y_limits : tuple
        A tuple (y1, y2) defining the integration limits for y.

    z_limits : tuple
        A tuple (z1, z2) defining the integration limits for z.

    num_points_x : int
        Number of Gaussian quadrature points for the x-dimension.

    num_points_y : int
        Number of Gaussian quadrature points for the y-dimension.

    num_points_z : int
        Number of Gaussian quadrature points for the z-dimension.

    Returns:
    --------
    float
        The approximated value of the triple integral.
    """

    # Get the Gaussian quadrature points and weights for each dimension.
    # These points and weights are defined over the standard interval [-1, 1].
    points_x, weights_x = leggauss(num_points_x)
    points_y, weights_y = leggauss(num_points_y)
    points_z, weights_z = leggauss(num_points_z)

    # Unpack the integration limits.
    x1, x2 = x_limits
    y1, y2 = y_limits
    z1, z2 = z_limits

    # Calculate the scaling factors to transform coordinates from [-1, 1]
    # to the actual integration limits [a, b].
    # The transformation is: x_actual = 0.5 * (b - a) * x_standard + 0.5 * (b + a)
    # The scaling factor is (b - a) / 2.
    factor_x = (x2 - x1) / 2.0
    factor_y = (y2 - y1) / 2.0
    factor_z = (z2 - z1) / 2.0

    approx_integral = 0.0

    # Perform the Gaussian quadrature sum for the triple integral.
    # The general formula is: Integral(f(x)) approx sum(w_i * f(x_i)).
    # For a triple integral, it extends to:
    # Integral(f(x,y,z)) approx sum_i sum_j sum_k (w_i * w_j * w_k * f(x_i_trans, y_j_trans, z_k_trans))
    # where x_i_trans, y_j_trans, z_k_trans are the transformed coordinates.

    for i in range(num_points_x):
        # Transform x-points from the standard interval [-1, 1] to [x1, x2].
        x_transformed = factor_x * points_x[i] + (x1 + x2) / 2.0

        for j in range(num_points_y):
            # Transform y-points from the standard interval [-1, 1] to [y1, y2].
            y_transformed = factor_y * points_y[j] + (y1 + y2) / 2.0

            for k in range(num_points_z):
                # Transform z-points from the standard interval [-1, 1] to [z1, z2].
                z_transformed = factor_z * points_z[k] + (z1 + z2) / 2.0

                # Evaluate the function at the transformed points and add to the result.
                # Multiply by the weights from each dimension.
                approx_integral += (weights_x[i] * weights_y[j] * weights_z[k] *
                                    func(x_transformed, y_transformed, z_transformed))

    # Multiply by the volume scaling factors (Jacobian of the transformation).
    approx_integral *= (factor_x * factor_y * factor_z)

    return approx_integral

# --- Example Usage ---

if __name__ == "__main__":
    # 1. Define the function to integrate.
    # Example: f(x, y, z) = x*y*z
    def my_function(x, y, z):
        return x * y * z

    # 2. Define the integration limits.
    x_lim = (0, 1)
    y_lim = (0, 2)
    z_lim = (0, 3)

    # 3. Define the number of Gaussian quadrature points for each dimension.
    # A higher number of points increases precision but also computational time.
    # For polynomial functions up to a certain degree, Gaussian quadrature can be exact.
    num_points = 5 # A common number to start with. You can try 2, 3, 4, etc.

    print(f"Function to integrate: f(x, y, z) = x*y*z")
    print(f"Integration limits: x in {x_lim}, y in {y_lim}, z in {z_lim}")
    print(f"Number of quadrature points (per dimension): {num_points}\n")

    # Perform the integration.
    result = triple_integrator(my_function, x_lim, y_lim, z_lim, num_points, num_points, num_points)

    print(f"The approximate result of the integral is: {result}")

    # To manually verify the result (Integral of x*y*z from 0 to 1, 0 to 2, 0 to 3)
    # Integral(x*y*z dx dy dz) = [x^2/2]_0^1 * [y^2/2]_0^2 * [z^2/2]_0^3
    #                          = (1/2 - 0) * (4/2 - 0) * (9/2 - 0)
    #                          = (1/2) * (2) * (9/2) = 9/2 = 4.5

    print("\n--- Another example: Non-polynomial function ---")
    def another_function(x, y, z):
        return np.sin(x) * np.cos(y) + np.exp(z)

    x_lim2 = (0, np.pi / 2)
    y_lim2 = (0, np.pi / 2)
    z_lim2 = (0, 1)
    num_points2 = 10 # More points are recommended for non-polynomial functions.

    print(f"Function to integrate: f(x, y, z) = sin(x)*cos(y) + exp(z)")
    print(f"Integration limits: x in {x_lim2}, y in {y_lim2}, z in {z_lim2}")
    print(f"Number of quadrature points (per dimension): {num_points2}\n")

    result2 = triple_integrator(another_function, x_lim2, y_lim2, z_lim2, num_points2, num_points2, num_points2)
    print(f"The approximate result of the integral is: {result2}")
