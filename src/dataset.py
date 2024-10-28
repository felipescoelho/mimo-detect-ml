"""dataset.py

Script to manage dataset.

luizfelipe.coelho@smt.ufrj.br
Oct 18, 2024
"""


import os
import torch
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
from .qam import qammod
from .utils import onehot_map, awgn


class DatasetQAM(Dataset):
    def __init__(self, K: int, N: int, mod_size: int, ensemble: int,
                 H: np.ndarray, snr_dB: tuple, seed: int | None = None,
                 dataset_path: str | None = None):
        """Generates the dataset for the autoencoder.
        
        Args
        ----
        K : int
            number of symbols
        N : int
            number of antennas at the receiver
        mod_size : int
            modulation size for QAM
        ensemble : int
            number of iterations
        seed : int | None = None
            seed for RNG
        dataset_path : str | None = None
            path to dataset
        """

        super().__init__()
        self.rng = np.random.default_rng(seed)
        self.mod_size = mod_size
        if (dataset_path is not None) and os.path.isfile(dataset_path):
            self.X, self.Y, self.labels, self.onehot_map = \
                torch.load(dataset_path)
        else:
            source_bits = self.rng.integers(
                0, 1, size=(int(ensemble*K*np.log2(mod_size).item()),),
                endpoint=True
            )
            symbols = qammod(source_bits, mod_size).reshape((ensemble, K))
            x = np.zeros((ensemble, int(2*K)))
            y = np.zeros((ensemble, int(2*N)))
            print('Allocate data to save:')
            match mod_size:
                case 2:
                    P_sym = 1
                case 4:
                    P_sym = 2
                case 16:
                    P_sym = 10
            P_in = np.trace(np.conj(H).T @ H) * P_sym / N
            for idx in tqdm(range(ensemble)):
                if len(snr_dB) == 1:
                    received = awgn(H @ symbols[idx, :], P_in, snr_dB[0])
                else:
                    received = awgn(H @ symbols[idx, :], P_in,
                                    self.rng.uniform(snr_dB[0], snr_dB[1]))
                y[idx, :] = np.hstack((received.real, received.imag))
                x[idx, :] = np.hstack((symbols[idx, :].real,
                                       symbols[idx, :].imag))
            x_onehot, self.onehot_map = onehot_map(x)
            self.X = torch.from_numpy(x)
            self.Y = torch.from_numpy(y)
            self.labels = torch.from_numpy(x_onehot)
            if dataset_path is not None:
                torch.save([self.X, self.Y, self.labels, self.onehot_map],
                           dataset_path)

    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, index):
        y = self.Y[index, :]
        label = self.labels[index, :, :]
        return y, label


class DatasetBPSK(Dataset):
    def __init__(self, K: int, N: int, ensemble: int, H: np.ndarray,
                 snr_dB: tuple | list, seed: int | None = None,
                 dataset_path: str | None = None):
        """Generates the dataset for the autoencoder.
        
        Args
        ----
        K : int
            number of symbols
        N : int
            number of antennas at the receiver
        ensemble : int
            number of iterations
        seed : int | None = None
            seed for RNG
        dataset_path : str | None = None
            path to dataset
        """

        super().__init__()
        self.rng = np.random.default_rng(seed)
        if (dataset_path is not None) and os.path.isfile(dataset_path):
            self.X, self.Y, self.labels = torch.load(dataset_path)
        else:
            source_bits = self.rng.integers(0, 1, size=(int(ensemble*K),),
                                            endpoint=True)
            x = (2*source_bits - 1).reshape(ensemble, K)
            y = np.zeros((ensemble, N))
            labels = np.zeros((ensemble, 2, K))
            for idx in range(ensemble):
                if len(snr_dB) == 1:
                    snr = snr_dB[0]
                else:
                    snr = self.rng.triangular(snr_dB[0], snr_dB[1], snr_dB[1])
                snr_lin = 10**(snr/10)
                signal_power = snr_lin / 1
                w = self.rng.standard_normal((N,))
                y[idx, :] = H @ x[idx, :] * np.sqrt(signal_power) + w
                for k in range(K):
                    labels[idx, :, k] = np.array((1, 0)) if x[idx, k] == -1 \
                        else np.array((0, 1))
            self.X = torch.from_numpy(x)
            self.Y = torch.from_numpy(y)
            self.labels = torch.from_numpy(labels)
            if dataset_path is not None:
                torch.save([self.X, self.Y, self.labels], dataset_path)

    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, index):
        y = self.Y[index, :]
        label = self.labels[index, :, :]
        return y, label


class DatasetQPSK(Dataset):
    def __init__(self, K: int, N: int, ensemble: int, H: np.ndarray,
                 snr_dB: tuple | list, seed: int | None = None,
                 dataset_path: str | None = None):
        """Generates the dataset for the autoencoder.
        
        Args
        ----
        K : int
            number of symbols
        N : int
            number of antennas at the receiver
        ensemble : int
            number of iterations
        seed : int | None = None
            seed for RNG
        dataset_path : str | None = None
            path to dataset
        """

        super().__init__()
        self.rng = np.random.default_rng(seed)
        if (dataset_path is not None) and os.path.isfile(dataset_path):
            self.X, self.Y, self.labels = torch.load(dataset_path)
        else:
            source_bits_real = self.rng.integers(0, 1, size=(int(ensemble*K),),
                                                 endpoint=True)
            source_bits_imag = self.rng.integers(0, 1, size=(int(ensemble*K),),
                                                 endpoint=True)
            x_real = (2*source_bits_real - 1).reshape(ensemble, K)
            x_imag = (2*source_bits_imag - 1).reshape(ensemble, K)
            x = x_real + 1j*x_imag
            y = np.zeros((ensemble, N), dtype=np.complex128)
            labels_real = np.zeros((ensemble, 2, K))
            labels_imag = np.zeros((ensemble, 2, K))
            for idx in tqdm(range(ensemble)):
                if len(snr_dB) == 1:
                    snr = snr_dB[0]
                else:
                    snr = self.rng.triangular(snr_dB[0], snr_dB[1]-1,
                                              snr_dB[1])
                snr_lin = 10**(snr/10)
                signal_power = snr_lin / 1
                w = self.rng.standard_normal((N,)) \
                    + 1j*self.rng.standard_normal((N,))
                y[idx, :] = H @ x[idx, :] * np.sqrt(signal_power) + w
                for k in range(K):
                    labels_real[idx, :, k] = np.array((1, 0)) if \
                        x_real[idx, k] == -1 else np.array((0, 1))
                    labels_imag[idx, :, k] = np.array((1, 0)) if \
                        x_imag[idx, k] == -1 else np.array((0, 1))
            self.X = torch.from_numpy(x)
            self.Y = torch.from_numpy(y)
            self.labels = torch.from_numpy(np.concatenate((labels_real,
                                                           labels_imag),
                                                          axis=-1))
            if dataset_path is not None:
                torch.save([self.X, self.Y, self.labels], dataset_path)

    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, index):
        y = torch.concat((self.Y[index, :].real, self.Y[index, :].imag))
        label = self.labels[index, :]
        return y, label
