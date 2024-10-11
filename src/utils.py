"""utils.py

Script containing utilitary functions.

luizfelipe.coelho@smt.ufrj.br
Set 19, 2024
"""


import torch
import numpy as np
from math import floor
from tqdm import tqdm
from qtorch.quant import fixed_point_quantize
from .vector_quatizer import mpgbp


def onehot_map(x: np.ndarray):
    """Method to implement the one-hot mapping
    
    Args
    ----
    x : np.ndarray
        input data vector.
    """

    ensemble, KK = x.shape
    unique_values = np.unique(x)
    onehot_len = len(unique_values)
    out = np.zeros((ensemble, onehot_len, KK))
    print('One-hot encoding:')
    for idx0 in tqdm(range(ensemble)):
        for idx1 in range(KK):
            for idx2, val in np.ndenumerate(unique_values):
                if x[idx0, idx1] == val:
                    out[idx0, idx2, idx1] = 1

    return out, unique_values


def awgn(x: np.ndarray, snr_dB: float, seed: int | None = None):
    """Method to add white Gaussian noise to a signal.
    
    Args
    ----
    x : np.ndarray
        input array
    snr_dB : float
        signal to noise ratio in dB
    seed : int | None (default=None)
        a seed for the random number generator
    """

    rng = np.random.default_rng(seed)
    n = rng.standard_normal((len(x),)) + 1j* rng.standard_normal((len(x),))
    n *= np.sqrt(((np.vdot(x, x)/len(x))*10**(-.1*snr_dB))
                 / (np.vdot(n, n)/len(n)))
    
    return x + n


def matrix_55_toeplitz(N: int, K: int, seed: int | None = None):
    """Generate the matrix for the 0.55-Toeplitz channel.
    
    Args
    ----
    N : int
        number of rows in H
    K : int
        number of columns in H
    seed : int
        a seed for the random number generator
    """

    rng = np.random.default_rng(seed)
    h = np.array([.55**k for k in range(K)])
    HTH = np.array([np.roll(h, k) for k in range(K)])
    for k in range(K):
        for kk in range(k):
            HTH[k, kk] = HTH[kk, k]
    _, sigma, VT = np.linalg.svd(HTH)
    rand_mat = rng.standard_normal((N, K)) + 1j*rng.standard_normal((N, K))
    Q, _ = np.linalg.qr(rand_mat, mode='complete')
    H = Q @ np.vstack((np.diag(np.sqrt(sigma)),
                       np.zeros((N-K, K), dtype=np.complex128))) @ VT
    
    return H


def quantize_coefficients(state_dict: dict, MN_ratio: float, device: str):
    """Method to quatize NN coefficients.
    
    Args
    ----
    state_dict : dict
        neural network state dictionary containing trained coefficients
    MN_ratio : float
        ratio between vector length and number of SPT terms
    """

    quantized_state_dict = {}
    print(f'Quantizing for {MN_ratio}...')
    for key, value in tqdm(state_dict.items()):
        name = key.split('.')[1]
        if name == 'bias':
            quantized_state_dict[key] = value
        elif len(value.shape) == 2:
            N = value.shape[1]
            M_max = int(MN_ratio*N)
            tensor = torch.zeros(value.shape)
            for idx in range(value.shape[0]):
                tensor[idx, :] = mpgbp(value[idx, :], M_max, floor(N**.5), device)
            quantized_state_dict[key] = tensor
        else:
            N = len(value)
            M_max = int(MN_ratio*N)
            quantized_state_dict[key] = mpgbp(value, M_max, floor(N**.5), device)
    
    return quantized_state_dict


def quantize_coefficients2(state_dict: dict, MN_ratio: float, wl:int, fl: int,
                           device: str):
    """Method to quatize NN coefficients.
    
    Args
    ----
    state_dict : dict
        neural network state dictionary containing trained coefficients
    MN_ratio : float
        ratio between vector length and number of SPT terms
    """

    quantized_state_dict = {}
    print(f'Quantizing for {MN_ratio}...')
    for key, value in tqdm(state_dict.items()):
        name = key.split('.')[1]
        if name == 'bias':
            quantized_state_dict[key] = fixed_point_quantize(value, wl, fl)
        elif len(value.shape) == 2:
            N = value.shape[1]
            M_max = int(MN_ratio*N)
            tensor = torch.zeros(value.shape)
            for idx in range(value.shape[0]):
                tensor[idx, :] = mpgbp(value[idx, :], M_max, floor(N**.5), device)
            quantized_state_dict[key] = tensor
        else:
            N = len(value)
            M_max = int(MN_ratio*N)
            quantized_state_dict[key] = mpgbp(value, M_max, floor(N**.5), device)
    
    return quantized_state_dict


def keep_quatized(x: torch.Tensor, MN_ratio: float, device='cuda'):
    """Method to keep quatization.
    
    Args
    ----
    x : torch.Tensor
        input tensor
    MN_ratio : float
        ratio between number of elements in vector and number of SPT
        terms
    """
    if len(x.shape) == 2:
        out_tensor = torch.zeros(x.shape, device=device)
        N = x.shape[1]
        M_max = int(MN_ratio*N)
        for idx in tqdm(range(x.shape[0])):
            out_tensor[idx, :] = mpgbp(x[idx, :], M_max, floor(N**.5), device)
    else:
        N = len(x)
        M_max = int(MN_ratio*N)
        out_tensor = mpgbp(x, M_max, floor(N**.5), device)
    
    return out_tensor
