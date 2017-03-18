import numpy as np


def pearson(u, v):
    u_offset = u - u.mean()
    v_offset = v - v.mean()
    numerator = (u_offset * v_offset).sum()
    u_rms = np.sqrt((u_offset ** 2).sum())
    v_rms = np.sqrt((v_offset**2).sum())
    denominator = u_rms * v_rms
    return numerator / denominator


def cosine(u, v):
    return u * v / (np.sqrt(u * u) * np.sqrt(v * v))
