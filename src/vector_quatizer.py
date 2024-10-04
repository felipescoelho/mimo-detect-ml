"""vector_quatizer.py

Script for the vector quatizer methods.

luizfelipe.coelho@smt.ufrj.br
Sep 25, 2024
"""


import numpy as np


def mpgbp(x: np.ndarray, N: int, P: int, bitplane_range: tuple, epsilon=1e-12):
    """
    Method to perform the Matching Pursuits with Generalized Bit Plane
    (MPGBP) algortihm for vector quantization.
    
    Args
    ----
    x : np.ndarray
        input vector
    N : int
        maximum number of non zero elements (SPT -- signed powers of two)
    P : int
        number of non zero elements in codeword components
    bitplane_range : tuple
        range for the bitplane (min_bp, max_bp) value for 2^{bit plane},
        i.e., the bit plane goes from 2^{min_bp} to 2^{max_bp}
    epsilon : float (default=1e-12)
        approximation threshold
    """

    x_hat = np.zeros((len(x),))
    residue = x.copy()
    count = 0
    while count < N:
        codeword = np.zeros((N,))
        idx_sort = np.argsort(-np.abs(residue))
        for idx in range(P):
            idx = idx_sort[idx]
            codeword[idx] = np.sign(residue[idx])
        codeword_normalized = codeword/np.linalg.norm(codeword)
        ip = np.dot(residue, codeword_normalized) / np.linalg.norm(codeword)
        powers = -np.ceil(np.log2(3/(4*ip)))
        spt = 2**powers
        residue -= spt*codeword
        x_hat += spt*codeword
        count += spt_count(x_hat, bitplane_range)
        if np.linalg.norm(residue) <= epsilon:
            break
    
    return x_hat


def spt_count(x_hat: np.ndarray, bitplane_range: tuple):
    """
    Method to count the number of signed powers of two in approximation.

    Args
    ----
    x_hat : torch.Tensor
        approximated input vector.
    bitplane_range : tuple
        range for the bitplane (min_bp, max_bp) value for 2^{bit plane},
        i.e., the bit plane goes from 2^{min_bp} to 2^{max_bp}
    """


    K = len(x_hat)
    count = np.zeros((K,))
    bitplane_axis = gen_bitplane(bitplane_range)
    for idx in range(K):
        residue = x_hat[idx]
        while residue != 0:
            bitplane = np.floor(np.log2(residue))
            bitplane_axis = allocate_power(bitplane_axis, bitplane,
                                           bitplane_range)
            residue = np.abs(residue - 2**bitplane)
            count[idx] += 1
    
    return count


def gen_bitplane(bitplane_range: tuple):
    """Method to generate bitplane from bitplane range."""

    min_bp, max_bp = bitplane_range
    bitplane = np.zeros((max_bp-min_bp+1,)) if max_bp > 0 and min_bp < 0 \
        else np.zeros((max_bp-min_bp,))
    
    return bitplane


def allocate_power(bitplane_axis: np.ndarray, bitplane: np.ndarray,
                   bitplane_range: tuple):
    """Method to allocate powers in bitplane."""

    min_bp, max_bp = bitplane_range
    idx = bitplane.copy()+1 if max_bp > 0 and min_bp < 0 else bitplane.copy()
    bitplane_axis[idx] = 1

    return bitplane_axis
