"""vq.py

Script for MPGBP faster

luizfelipe.coelho@smt.ufrj.br
Oct 17, 2024
"""


import numpy as np
from numba import njit


@njit()
def spt_count(x_hat: np.ndarray):
    K = len(x_hat)
    count = np.zeros((K,))
    for idx in range(K):
        residue = np.abs(x_hat[idx])
        while residue != 0:
            bitplane = np.floor(np.log2(residue))
            residue = np.abs(residue - 2**bitplane)
            count[idx] += 1
    
    return np.sum(count)


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
        codeword_normalized = codeword/np.sum(np.abs(codeword))
        ip = np.sum(residue*codeword_normalized)
        powers = -np.ceil(np.log2(3/(4*ip)))
        spt = 2**powers
        residue -= spt*codeword
        x_hat += spt*codeword
        M = spt_count(x_hat)
        if np.sum(residue**2)**.5 <= epsilon:
            break

    return x_hat
