import numpy as np

def compute_chain_rule_gradient(functions: list[str], x: float) -> float:
    funcs = {
        'square': lambda v: v ** 2,
        'sin':    np.sin,
        'exp':    np.exp,
        'log':    np.log,
    }
    derivs = {
        'square': lambda v: 2 * v,
        'sin':    np.cos,
        'exp':    np.exp,
        'log':    lambda v: 1 / v,
    }

    value = x      # current value (output of the functions applied so far)
    grad = 1.0     # running product of derivatives

    for name in reversed(functions):      # start from the innermost (last) function
        grad *= derivs[name](value)       # derivative evaluated at the current input
        value = funcs[name](value)        # then pass the output on to the next function

    return float(grad)