"""vq.py

Script for MPGBP faster

luizfelipe.coelho@smt.ufrj.br
Oct 17, 2024
"""


import numpy as np
from numba import njit


@njit()
def mpgbp(x: np.ndarray, M_max: int, P: int, epsilon=1e-6):
    """"""
    x_hat = np.zeros((len(x),))
    residue = x.copy()
    M = 0
    while M < M_max:
        codeword = np.zeros((len(x),))
        idx_sorted = np.argsort(-np.abs(residue))
        for idx in range(P):
            codeword[idx_sorted[idx]] = np.sign(residue[idx_sorted[idx]])
        M += np.sum(np.abs(codeword))
        codeword_normalized = codeword/np.sum(np.abs(codeword))
        ip = np.sum(residue*codeword_normalized)
        powers = -np.ceil(np.log2(3/(4*ip)))
        spt = 2**powers
        residue -= spt*codeword
        x_hat += spt*codeword
        if np.sum(residue**2)**.5 <= epsilon:
            break

    return x_hat
