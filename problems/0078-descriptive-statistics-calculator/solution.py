import numpy as np
from collections import Counter


def descriptive_statistics(data: list | np.ndarray) -> dict:
    data = np.sort(np.asarray(data))
    length = len(data)

    mean = float(np.sum(data) / length)

    middle = length // 2
    if length % 2 == 0:
        median = float((data[middle - 1] + data[middle]) / 2)
    else:
        median = float(data[middle])

    mode, count = Counter(data.tolist()).most_common(1)[0]

    variance = float(np.sum((data - mean) ** 2) / length)
    std = float(np.sqrt(variance))

    p25, p50, p75 = np.percentile(data, [25, 50, 75])

    return {
        "mean": mean,
        "median": median,
        "mode": mode,
        "variance": variance,
        "standard_deviation": std,
        "25th_percentile": float(p25),
        "50th_percentile": float(p50),
        "75th_percentile": float(p75),
        "interquartile_range": float(p75 - p25),
    }