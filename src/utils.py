"""utils.py

Script containing utilitary functions.

luizfelipe.coelho@smt.ufrj.br
Set 19, 2024
"""


import torch
import numpy as np
from math import floor
from tqdm import tqdm
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


def onehot_demap(x: torch.Tensor, onehotmap: np.ndarray):
    """Metho to demap the onehot encoding"""

    xx = x.argmax(1)
    if len(xx.shape) == 2:  # data in batch
        batch_size, data_len = xx.shape
        X = np.zeros((batch_size, int(data_len/2)), dtype=np.complex128)
        for idx0 in range(batch_size):
            for idx1 in range(int(data_len/2)):
                X[idx0, idx1] = onehotmap[xx[idx0, idx1]] \
                    + 1j*onehotmap[xx[idx0, idx1+int(data_len/2)]]
    else:
        xx = x.argmax(0)
        data_len = len(xx)
        X = np.zeros((int(data_len/2),), dtype=np.complex128)
        for idx in range(int(data_len/2)):
            X[idx] = onehotmap[xx[idx]] + 1j*onehotmap[xx[idx+int(data_len/2)]]

    return X


def decide(x: torch.Tensor, mod_size: int):
    """Method to decide which symbols are in x.
    
    Args
    ----
    x : torch.Tensor
        input vector
    mod_size : int
        modulation size
    """

    match mod_size:
        case 16:
            symbols = torch.tensor((-3-1j*3, -3-1j, -3+1j, -3+1j*3, -1-1j*3,
                                    -1-1j, -1+1j, -1+1j*3, 1-1j*3, 1-1j, 1+1j,
                                    1+1j*3, 3-1j*3, 3-1j, 3+1j, 3+1j*3),
                                    device=x.device)
            eggs = symbols.tile(len(x), 1)
            spam = x.tile(16, 1)
            sausage = torch.abs(spam.T - eggs).argmin(1)
            
            return symbols[sausage]


def awgn(x: np.ndarray, P_in: float, snr_dB: float, seed: int | None = None):
    """Method to add white Gaussian noise to a signal.
    
    Args
    ----
    x : np.ndarray
        input array
    P_in : float
        reference signal power
    snr_dB : float
        signal to noise ratio in dB
    seed : int | None (default=None)
        a seed for the random number generator
    """

    rng = np.random.default_rng(seed)
    n = rng.standard_normal((len(x),)) + 1j*rng.standard_normal((len(x),))
    sigma2_n = P_in * 10**(-.1*snr_dB)
    n *= np.sqrt(sigma2_n/2)
    
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


def quantize_coefficients_mpgbp(state_dict: dict, MN_ratio: float,
                                device: str):
    """Method to quatize NN coefficients using the mpgbp algorithm.
    
    Args
    ----
    state_dict : dict
        neural network state dictionary containing trained coefficients
    MN_ratio : float
        ratio between vector length and number of SPT terms
    """

    quantized_state_dict = {}
    print(f'Quantizing for M_max/L = {int(MN_ratio)} ...')
    for key, value in tqdm(state_dict.items()):
        name = key.split('.')[1]
        if name == 'bias':
            quantized_state_dict[key] = value
        elif len(value.shape) == 2:
            N = value.shape[1]
            M_max = int(MN_ratio*N)
            tensor = torch.zeros(value.shape, device=device)
            for idx in range(value.shape[0]):
                normalizer = 2**torch.ceil(
                    torch.log2(value[idx, :].abs().max())
                ).item()
                tensor[idx, :] = mpgbp(value[idx, :]/normalizer, M_max,
                                       floor(N**.5), device)*normalizer
            quantized_state_dict[key] = tensor
        else:
            N = len(value)
            M_max = int(MN_ratio*N)
            normalizer = 2**torch.ceil(torch.log2(value.abs().max())).item()
            quantized_state_dict[key] = mpgbp(value/normalizer, M_max,
                                              floor(N**.5), device)*normalizer
    
    return quantized_state_dict


def quantize_coefficients_naive_mpgbp(state_dict: dict, wl:int, device: str):
    """Method to quatize NN coefficients using a naïve algorithm
    
    Args
    ----
    state_dict : dict
        neural network state dictionary containing trained coefficients
    MN_ratio : float
        ratio between vector length and number of SPT terms
    """

    quantized_state_dict_naive = {}
    quantized_state_dict_mpgbp = {}
    print(f'Quantizing ...')
    for key, value in tqdm(state_dict.items()):
        name = key.split('.')[1]
        if name == 'bias':
            quantized_state_dict_naive[key] = value
            quantized_state_dict_mpgbp[key] = value
        elif len(value.shape) == 2:
            tensor_naive = torch.zeros(value.shape, device=device)
            tensor_mpgbp = torch.zeros(value.shape, device=device)
            L = value.shape[1]
            for idx0 in range(value.shape[0]):
                M_max = 0
                for idx1 in range(L):
                    normalizer = 2**torch.ceil(
                        torch.log2(value[idx0, idx1].abs())
                    ).item()
                    approx, num_spt = twos_complement(
                        value[idx0, idx1].item()/normalizer, wl, device
                    )
                    tensor_naive[idx0, idx1] = normalizer*approx
                    M_max += num_spt
                normalizer = 2**torch.ceil(
                    torch.log2(value[idx0, :].abs().max())
                ).item()
                tensor_mpgbp[idx0, :] = mpgbp(value[idx0, :]/normalizer, M_max,
                                              floor(L**.5), device)*normalizer
            quantized_state_dict_naive[key] = tensor_naive
            quantized_state_dict_mpgbp[key] = tensor_mpgbp
        else:
            L = len(value)
            M_max = 0
            tensor_naive = torch.zeros((L,), device=device)
            for idx in range(L):
                normalizer = 2**torch.ceil(
                    torch.log2(value[idx].abs().max())
                ).item()
                approx, num_spt = twos_complement(value[idx].item()/normalizer,
                                                  wl, device)
                tensor_naive[idx] = approx*normalizer
                M_max += num_spt
            quantized_state_dict_naive[key] = tensor_naive
            normalizer = 2**torch.ceil(torch.log2(value.abs().max())).item()
            quantized_state_dict_mpgbp[key] = mpgbp(value/normalizer, M_max,
                                                    floor(L**.5),
                                                    device)*normalizer
    
    return quantized_state_dict_naive, quantized_state_dict_mpgbp


def twos_complement(x: float, wl: int, device: str):
    """Method to perform the two's complement representation and count
    number of SPT's necessary for computation.
    
    This is dog's math ...

    Args
    ----
    x : float
        input value (normalized by a power of two)
    wl : int
        wordlength (number of bits, considering the sign bit)
    device : str
        where the tensor is
    """

    x_2c = torch.zeros((wl,), device=device)
    spam = abs(x)
    for it in range(1, wl):
        spam *= 2
        if spam >= 1:
            x_2c[it] = 1
            spam -= 1
    if x < 0:
        x_2c = torch.logical_not(x_2c).type(torch.float)  # 1's complement
        x_2c[-1] += 1  # Add 1 to LSB to become 2's complement
        for bit in range(wl-2, 1, -1):
            if x_2c[bit+1] == 2:  # Propagate value
                x_2c[bit] -= 1
                x_2c[bit+1] = 0
            else:
                break
        if x_2c[1] == 2:
            print('Overflow! Why?')
            x_2c[1] = 0
    base = torch.tensor([2**-bit for bit in range(wl)], device=device).float()
    base[0] = -1
    approx_spt, spt_count = csd_representation(x_2c, device)

    return approx_spt, spt_count


def csd_representation(x: torch.Tensor, device: str):
    """
    Method to count the number of SPT terms in the SPT from the 2's
    complement number representation.
    
    This is still dog's math, but it's Border Collie level ...

    Args
    ----
    x : torch.Tensor
        input vector, number represented in 2's complement
    device : str
        where is the tensor
    """

    n = len(x)-1
    base = torch.tensor([2**-bit for bit in range(n+1)], device=device).float()
    delta = torch.tensor(False, device=device)
    x_in = torch.hstack((x, torch.tensor(False, device=device)))
    x_csd = torch.zeros((n+1,), device=device)
    for i in range(n, -1, -1):
        theta = torch.logical_xor(x_in[i], x_in[i+1])
        delta = torch.logical_and(torch.logical_not(delta), theta)
        if i == 0:
            x_csd[0] = (1 - 2*x[0])*delta
            approx = torch.dot(base, x_csd)

            return approx, x_csd.abs().sum().item()
        
        x_csd[i] = (1 - 2*x_in[i-1])*delta


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
