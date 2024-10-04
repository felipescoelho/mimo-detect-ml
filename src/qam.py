"""qam.py

Script to handle QAM.

luizfelipe.coelho@smt.ufrj.br
Sep 17, 2024
"""


import torch
from math import log2
from tqdm import tqdm


def qammod(data: torch.Tensor, mod_size: int):
    """
    QAM modulation using gray coding and separating real from imaginary
    parts in the complex number.

    Args
    ----
    data : torch.Tensor
        bit stream
    mod_size : int
        mulation size, choose between 2, 4, and 16
    """

    spam = int(data.shape[1]/log2(mod_size))
    data_vect = data.reshape((data.shape[0], int(log2(mod_size)), spam))
    data_out = torch.zeros((data.shape[0], spam, 2))
    match mod_size:
        case 2:
            for idx0 in range(data.shape[0]):
                for idx1 in range(spam):
                    data_out[idx0, idx1, 0] = \
                        1 if data_vect[idx0, :, idx1] == 1 else -1
        case 4:
            for idx0 in range(data.shape[0]):
                for idx1 in range(spam):
                    sym0 = torch.tensor([0, 0])
                    sym1 = torch.tensor([0, 1])
                    sym2 = torch.tensor([1, 0])

                    symbol = data_vect[idx0, :, idx1]
                    if symbol.equal(sym0):  # 00
                        data_out[idx0, idx1, 0] = -1
                        data_out[idx0, idx1, 1] = -1
                    elif symbol.equal(sym1):  # 01
                        data_out[idx0, idx1, 0] = -1
                        data_out[idx0, idx1, 1] = 1
                    elif symbol.equal(sym2):  # 10
                        data_out[idx0, idx1, 0] = 1
                        data_out[idx0, idx1, 1] = -1
                    else:  # 11
                        data_out[idx0, idx1, 0] = 1
                        data_out[idx0, idx1, 1] = 1
        case 16:
            for idx0 in tqdm(range(data.shape[0])):
                for idx1 in range(spam):
                    sym00 = torch.tensor([0, 0, 0, 0])
                    sym01 = torch.tensor([0, 0, 0, 1])
                    sym02 = torch.tensor([0, 0, 1, 0])
                    sym03 = torch.tensor([0, 0, 1, 1])
                    sym04 = torch.tensor([0, 1, 0, 0])
                    sym05 = torch.tensor([0, 1, 0, 1])
                    sym06 = torch.tensor([0, 1, 1, 0])
                    sym07 = torch.tensor([0, 1, 1, 1])
                    sym08 = torch.tensor([1, 0, 0, 0])
                    sym09 = torch.tensor([1, 0, 0, 1])
                    sym10 = torch.tensor([1, 0, 1, 0])
                    sym11 = torch.tensor([1, 0, 1, 1])
                    sym12 = torch.tensor([1, 1, 0, 0])
                    sym13 = torch.tensor([1, 1, 0, 1])
                    sym14 = torch.tensor([1, 1, 1, 0])
                    
                    symbol = data_vect[idx0, :, idx1]
                    if symbol.equal(sym00):  # 0000
                        data_out[idx0, idx1, 0] = -3
                        data_out[idx0, idx1, 1] = -3
                    elif symbol.equal(sym01):  # 0001
                        data_out[idx0, idx1, 0] = -3
                        data_out[idx0, idx1, 1] = -1
                    elif symbol.equal(sym02):  # 0010
                        data_out[idx0, idx1, 0] = -3
                        data_out[idx0, idx1, 1] = 3
                    elif symbol.equal(sym03):  # 0011
                        data_out[idx0, idx1, 0] = -3
                        data_out[idx0, idx1, 1] = 1
                    elif symbol.equal(sym04):  # 0100
                        data_out[idx0, idx1, 0] = -1
                        data_out[idx0, idx1, 1] = -3
                    elif symbol.equal(sym05):  # 0101
                        data_out[idx0, idx1, 0] = -1
                        data_out[idx0, idx1, 1] = -1
                    elif symbol.equal(sym06):  # 0110
                        data_out[idx0, idx1, 0] = -1
                        data_out[idx0, idx1, 1] = 3
                    elif symbol.equal(sym07):  # 0111
                        data_out[idx0, idx1, 0] = -1
                        data_out[idx0, idx1, 1] = 1
                    elif symbol.equal(sym08):  # 1000
                        data_out[idx0, idx1, 0] = 3
                        data_out[idx0, idx1, 1] = -3
                    elif symbol.equal(sym09):  # 1001
                        data_out[idx0, idx1, 0] = 3
                        data_out[idx0, idx1, 1] = -1
                    elif symbol.equal(sym10):  # 1010
                        data_out[idx0, idx1, 0] = 3
                        data_out[idx0, idx1, 1] = 3
                    elif symbol.equal(sym11):  # 1011
                        data_out[idx0, idx1, 0] = 3
                        data_out[idx0, idx1, 1] = 1
                    elif symbol.equal(sym12):  # 1100
                        data_out[idx0, idx1, 0] = 1
                        data_out[idx0, idx1, 1] = -3
                    elif symbol.equal(sym13):  # 1101
                        data_out[idx0, idx1, 0] = 1
                        data_out[idx0, idx1, 1] = -1
                    elif symbol.equal(sym14):  # 1110
                        data_out[idx0, idx1, 0] = 1
                        data_out[idx0, idx1, 1] = 3
                    else:  # 1111
                        data_out[idx0, idx1, 0] = 1
                        data_out[idx0, idx1, 1] = 1
    
    return data_out


def qamdemod(data: torch.Tensor, mod_size: int):
    """
    QAM demoduation using Gray coding and considering that the real and
    imaginary parts are divided in the complex numbers.
    
    Args
    ----
    data : torch.Tensor
        input QAM modulated data
    mod_size : int
        modulation size
    """

    data_vect = torch.zeros((int(log2(mod_size)), data.shape[0]))
    match mod_size:
        case 2:
            for idx in range(data.shape[0]):
                data_vect[0, idx] = 1 if data[idx, 0] > 0 else 0
        case 4:
            symbols = torch.tensor([[-1, -1], [-1, 1], [1, -1], [1, 1]])
            for idx in range(data.shape[0]):
                distance = symbols - data[idx, :]

        case 16:
            symbols = torch.tensor([[-3, -3], [-3, -1], [-3, 3], [-3, 1],
                                    [-1, -3], [-1, -1], [-1, 3], [-1, 1],
                                    [3, -3], [3, -1], [3, 3], [3, 1],
                                    [1, -3], [1, -1], [1, 3], [1, 1]])
            for idx in range(data.shape[0]):
                distance = symbols - data[idx, :]
                print(distance)
    
    return data_vect.flatten()
