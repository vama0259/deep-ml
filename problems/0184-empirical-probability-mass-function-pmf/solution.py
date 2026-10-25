def empirical_pmf(samples):
    """
    Given an iterable of integer samples, return a list of (value, probability)
    pairs sorted by value ascending.
    """
    # TODO: Implement the function
    if len(samples) == 0:
        return []
    from collections import Counter
    number = len(samples)
    a = Counter(samples)
    b = {}

    for value, count in a.items():
        b[value] = count / number

    return sorted(b.items(), key=lambda item: item[0])