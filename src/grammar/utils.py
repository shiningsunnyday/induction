import numpy as np
from collections.abc import Iterable


def find_next(g, prefix=""):
    if prefix:
        g_nodes = [n[len(prefix) :] for n in g if n[: len(prefix)] == prefix]
    else:
        g_nodes = list(g)
    all_int = np.all([isinstance(n, int) for n in g_nodes])
    if all_int:
        if prefix:
            breakpoint()
        return max(list(g)) + 1
    else:
        key = str(max(list(map(int, g_nodes))) + 1)
        return f"{prefix}{key}"


def next(n):
    if isinstance(n, str):
        if ":" in n:
            prefix, n = n.split(":")
            key = str(int(n) + 1)
            return f"{prefix}:{key}"
        else:
            return str(int(n) + 1)
    else:
        return n + 1


def flatten(nested_iterable):
    if isinstance(nested_iterable, Iterable):
        return sum([flatten(iterable) for iterable in nested_iterable], [])
    else:
        return [nested_iterable]


def find_max(g):
    all_int = np.all([isinstance(n, int) for n in g])
    if all_int:
        return max(list(g))
    else:
        return str(max(list(map(int, g))))
