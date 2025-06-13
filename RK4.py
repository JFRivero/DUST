import numpy as np

def fixed_step_rk4(func, y0, t_points):
    """
    Solves an ODE using the 4th-order Runge-Kutta method with a fixed step size.

    Parameters:
        func (callable): The function defining the ODE, f(t, y).
        y0 (array): The initial condition.
        t_points (array): An array of time points where the solution will be computed.
                         The time step 'h' is inferred from these points.

    Returns:
        An array with the solution values 'y' at each point in 't_points'.
    """
    n = len(t_points)
    y = np.zeros((n, len(y0))) # Array to store the results
    y[0] = y0                  # Assign the initial condition

    # The time step 'h' is the difference between time points
    h = t_points[1] - t_points[0]

    # Integration loop
    for i in range(n - 1):
        t_current = t_points[i]
        y_current = y[i]

        # RK4 method formulas
        k1 = h * func(t_current, y_current)
        k2 = h * func(t_current + 0.5 * h, y_current + 0.5 * k1)
        k3 = h * func(t_current + 0.5 * h, y_current + 0.5 * k2)
        k4 = h * func(t_current + h, y_current + k3)

        # Calculate the next value of 'y'
        y[i+1] = y_current + (k1 + 2*k2 + 2*k3 + k4) / 6.0

    return y.T # Transpose so the shape is (num_variables, num_points) like solve_ivp
