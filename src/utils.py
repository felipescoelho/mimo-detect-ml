"""utils.py

Script containing utilitary functions.

luizfelipe.coelho@smt.ufrj.br
Set 19, 2024
"""


import torch


def dft_mat(N: int):
    """DFT matrix.
    
    Args
    ----
    N : int
        number of DFT bins.
    """

    return torch.tensor([[torch.exp(-1j*2*torch.pi*n*k/N) for n in range(N)]
                         for k in range(N)])


def idft_mat(N: int):
    """IDFT matrix.
    
    Args
    ----
    N : int
        number of DFT bins
    """

    return torch.tensor([[torch.exp(1j*2*torch.pi*n*k/N)/N for n in range(N)]
                         for k in range(N)])
