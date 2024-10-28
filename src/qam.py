"""qam.py

Script with QAM related functions - Numpy-based.

luizfelipe.coelho@smt.ufrj.br
Sep 30, 2024
"""


import numpy as np
from tqdm import tqdm


def qammod(data: np.ndarray, mod_size: int):
    """
    QAM modulation using Gray coding.

    Args
    ----
    data : np.ndarray
        bit stream
    mod_size : int
        modulation size, 2, 4, or 16.
    """

    out_len = int(len(data)/np.log2(mod_size))
    data_vect = data.reshape((out_len, int(np.log2(mod_size))))
    data_out = np.zeros((out_len,), dtype=np.complex128)
    match mod_size:
        case 2:
            for idx in range(out_len):
                data_out[idx] = 1 if data_vect[idx, :] == 1 else -1
        case 4:
            sym0 = np.array((0, 0))
            sym1 = np.array((0, 1))
            sym2 = np.array((1, 0))
            for idx in range(out_len):
                if np.array_equal(data_vect[idx, :], sym0):  # 00
                    data_out[idx] = -1-1j
                elif np.array_equal(data_vect[idx, :], sym1):  # 01
                    data_out[idx] = -1+1j
                elif np.array_equal(data_vect[idx, :], sym2):  # 10
                    data_out[idx] = 1-1j
                else:  # 11
                    data_out[idx] = 1+1j
        case 16:
            sym00 = np.array((0, 0, 0, 0))
            sym01 = np.array((0, 0, 0, 1))
            sym02 = np.array((0, 0, 1, 0))
            sym03 = np.array((0, 0, 1, 1))
            sym04 = np.array((0, 1, 0, 0))
            sym05 = np.array((0, 1, 0, 1))
            sym06 = np.array((0, 1, 1, 0))
            sym07 = np.array((0, 1, 1, 1))
            sym08 = np.array((1, 0, 0, 0))
            sym09 = np.array((1, 0, 0, 1))
            sym10 = np.array((1, 0, 1, 0))
            sym11 = np.array((1, 0, 1, 1))
            sym12 = np.array((1, 1, 0, 0))
            sym13 = np.array((1, 1, 0, 1))
            sym14 = np.array((1, 1, 1, 0))
            print('16QAM:')
            for idx in tqdm(range(out_len), leave=False):
                if np.array_equal(data_vect[idx, :], sym00):  # 0000
                    data_out[idx] = -3 - 1j*3
                elif np.array_equal(data_vect[idx, :], sym01):  # 0001
                    data_out[idx] = -3 - 1j
                elif np.array_equal(data_vect[idx, :], sym02):  # 0010
                    data_out[idx] = -3 + 1j*3
                elif np.array_equal(data_vect[idx, :], sym03):  # 0011
                    data_out[idx] = -3 + 1j
                elif np.array_equal(data_vect[idx, :], sym04):  # 0100
                    data_out[idx] = -1 - 1j*3
                elif np.array_equal(data_vect[idx, :], sym05):  # 0101
                    data_out[idx] = -1 - 1j
                elif np.array_equal(data_vect[idx, :], sym06):  # 0110
                    data_out[idx] = -1 + 1j*3
                elif np.array_equal(data_vect[idx, :], sym07):  # 0111
                    data_out[idx] = -1 + 1j
                elif np.array_equal(data_vect[idx, :], sym08):  # 1000
                    data_out[idx] = 3 - 1j*3
                elif np.array_equal(data_vect[idx, :], sym09):  # 1001
                    data_out[idx] = 3 - 1j
                elif np.array_equal(data_vect[idx, :], sym10):  # 1010
                    data_out[idx] = 3 + 1j*3
                elif np.array_equal(data_vect[idx, :], sym11):  # 1011
                    data_out[idx] = 3 + 1j
                elif np.array_equal(data_vect[idx, :], sym12):  # 1100
                    data_out[idx] = 1 - 1j*3
                elif np.array_equal(data_vect[idx, :], sym13):  # 1101
                    data_out[idx] = 1 - 1j
                elif np.array_equal(data_vect[idx, :], sym14):  # 1110
                    data_out[idx] = 1 + 1j*3
                else:
                    data_out[idx] = 1 + 1j
    
    return data_out


def qamdemod(data: np.ndarray, mod_size: int):
    """
    QAM demodulation using Gray coding.
    
    Args
    ----
    data : np.ndarray
        input QAM modulated data
    mod_size : int
        modulation size
    """

    data_out = np.zeros((len(data), int(np.log2(mod_size))))
    match mod_size:
        case 2:
            for idx, x in np.ndenumerate(data):
                data_out[idx, :] = 1 if x > 0 else 0
        case 4:
            symbols = np.array([-1-1j, -1+1j, 1-1j, 1+1j])
            bits = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
            for idx, x in np.ndenumerate(data):
                distance = np.abs(symbols - x)
                data_out[idx, :] = bits[np.argmin(distance)]
        case 16:
            symbols = np.array([-3-1j*3, -3-1j, -3+1j*3, -3+1j, -1-1j*3, -1-1j,
                                -1+1j*3, -1+1j, 3-1j*3, 3-1j, 3+1j*3, 3+1j,
                                1-1j*3, 1-1j, 1+1j*3, 1+1j])
            bits = np.array([[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0],
                             [0, 0, 1, 1], [0, 1, 0, 0], [0, 1, 0, 1],
                             [0, 1, 1, 0], [0, 1, 1, 1], [1, 0, 0, 0],
                             [1, 0, 0, 1], [1, 0, 1, 0], [1, 0, 1, 1],
                             [1, 1, 0, 0], [1, 1, 0, 1], [1, 1, 1, 0],
                             [1, 1, 1, 1]])
            for idx, x in np.ndenumerate(data):
                distance = np.abs(symbols - x)
                data_out[idx, :] = bits[np.argmin(distance)]
    
    return data_out
