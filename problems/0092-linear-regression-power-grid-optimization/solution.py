import numpy as np
import math

def power_grid_forecast(consumption_data):
    n = len(consumption_data)
    Y = np.array(consumption_data, dtype=float)
    days = np.arange(1, n + 1, dtype=float)

    f = np.vectorize(lambda i: 10 * math.sin(2 * math.pi * i / 10))
    fluctuation = f(days)
    difference = Y - fluctuation

    m, c = np.polyfit(days, difference, 1)

    day15_base = m * 15 + c
    day15_prediction = day15_base + f(15)

    total_15 = round(day15_prediction)
    return math.ceil(total_15 * 1.05)