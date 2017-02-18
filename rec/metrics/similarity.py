import numpy as np

def pearson(u, v):
    u_offset = u - u.mean()
    v_offset = v - v.mean()
    return (u_offset * v_offset).sum() / \
           ( np.sqrt((u_offset**2).sum()) * np.sqrt((v_offset**2).sum()))

def cosine(u, v):
    return u * v / (np.sqrt(u * u) * np.sqrt(v * v))