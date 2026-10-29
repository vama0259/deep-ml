import numpy as np

def compute_partial_derivatives(func_name: str, point: tuple) -> tuple:
    if func_name == 'poly2d':            # x^2 y + x y^2
        x, y = point
        return (float(2*x*y + y**2), float(x**2 + 2*x*y))

    if func_name == 'exp_sum':           # e^(x+y)
        x, y = point
        e = np.exp(x + y)
        return (float(e), float(e))

    if func_name == 'product_sin':       # x sin(y)
        x, y = point
        return (float(np.sin(y)), float(x * np.cos(y)))

    if func_name == 'poly3d':            # x^2 y + y z^2
        x, y, z = point
        return (float(2*x*y), float(x**2 + z**2), float(2*y*z))

    if func_name == 'squared_error':     # (x-y)^2
        x, y = point
        return (float(2*(x - y)), float(-2*(x - y)))