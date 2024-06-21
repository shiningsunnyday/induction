import numpy as np

def find_next(g):
    all_int = np.all([isinstance(n, int) for n in g])
    if all_int:
        return max(list(g))+1    
    else:
        return str(max(list(map(int, g)))+1)


def next(n):
    if isinstance(n, str):
        return str(int(n)+1)
    else:
        return n+1


def find_max(g):
    all_int = np.all([isinstance(n, int) for n in g])
    if all_int:
        return max(list(g))   
    else:
        return str(max(list(map(int, g))))
    