import numpy as np

def calc(arr, x, ask = 'normal'):
    len_x = len(arr)
    ans = 0
    if ask == 'normal':
        for a in range(len_x):
            ans += arr[a] * (x ** ((len_x - (a+1))))
    elif ask =='dash':
        for a in range(len_x -1):
            power_v = len_x - 1 - a
            ans += arr[a] * power_v * (x ** (power_v - 1))
    return ans
def quotient_rule_derivative(g_coeffs: list, h_coeffs: list, x: float) -> float:
    """
    Compute the derivative of f(x) = g(x)/h(x) at point x using the quotient rule.
    
    Args:
        g_coeffs: Coefficients of numerator polynomial in descending order
        h_coeffs: Coefficients of denominator polynomial in descending order
        x: Point at which to evaluate the derivative
        
    Returns:
        The derivative value f'(x)
    """
    g, h = calc(g_coeffs,x), calc(h_coeffs,x)
    g_dash, h_dash = calc(g_coeffs,x, ask = 'dash'), calc(h_coeffs,x, ask = 'dash')

    
    
    return (((g_dash*h)-(h_dash*g))/(h**2))
    # g_dash
