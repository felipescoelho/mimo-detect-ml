"""model.py

Script to contain autoencoder.

luizfelipe.coelho@smt.ufrj.br
Sep 17, 2024
"""


import os
import torch
import numpy as np
from math import log2
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from qtorch.quant import fixed_point_quantize
from .qam import qammod
from .utils import onehot_map, awgn, keep_quatized


class DatasetQAM(Dataset):
    def __init__(self, K: int, N: int, mod_size: int, ensemble: int,
                 H: np.ndarray, snr_dB: np.ndarray, seed: int | None = None,
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
        if os.path.isfile(dataset_path):
            self.X, self.Y, self.labels, self.onehot_map = torch.load(dataset_path)
        else:
            source_bits = self.rng.integers(
                0, 1, size=(int(ensemble*K*log2(mod_size)),), endpoint=True
            )
            symbols = qammod(source_bits, mod_size).reshape((ensemble, K))
            x = np.zeros((ensemble, int(2*K)))
            y = np.zeros((ensemble, int(2*N)))
            print('Allocate data to save:')
            for idx in tqdm(range(ensemble)):
                noisy_symbols = awgn(symbols[idx, :], self.rng.choice(snr_dB))
                received = H @ noisy_symbols
                y[idx, :] = np.hstack((received.real, received.imag))
                # for the reference:
                x[idx, :] = np.hstack((symbols[idx, :].real,
                                       symbols[idx, :].imag))
            x_onehot, self.onehot_map = onehot_map(x)
            
            # import pdb; pdb.set_trace()
            self.X = torch.from_numpy(x)
            self.Y = torch.from_numpy(y)
            self.labels = torch.from_numpy(x_onehot)

            torch.save([self.X, self.Y, self.labels, self.onehot_map], dataset_path)

    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, index):
        y = self.Y[index, :]
        label = self.labels[index, :, :]
        return y, label


class MLP(torch.nn.Module):
    """Multilayer Perceptron"""

    def __init__(self, N, K):
        super().__init__()
        input_size = int(2*N)
        hidden_features = int(10*K)
        output_size = int(2*K*4)
        self.fc1 = torch.nn.Linear(input_size, hidden_features)
        self.fc2 = torch.nn.Linear(hidden_features, hidden_features)
        self.fc3 = torch.nn.Linear(hidden_features, hidden_features)
        self.fc4 = torch.nn.Linear(hidden_features, hidden_features)
        self.fc5 = torch.nn.Linear(hidden_features, hidden_features)
        self.fc6 = torch.nn.Linear(hidden_features, output_size)
        self.logit_shape = (4, int(2*K))

    def forward(self, x):
        
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = torch.nn.functional.relu(self.fc3(x))
        x = torch.nn.functional.relu(self.fc4(x))
        x = torch.nn.functional.relu(self.fc5(x))
        x = self.fc6(x)
        x = x.reshape(x.shape[0], *self.logit_shape)

        return x


class MLPQuantized(torch.nn.Module):
    """Multilayer Perceptron but SOPOT"""

    def __init__(self, N: int, K: int, wl: int, fl: int):
        super().__init__()
        input_size = int(2*N)
        hidden_features = int(10*K)
        output_size = int(2*K*4)
        self.fc1 = torch.nn.Linear(input_size, hidden_features)
        self.fc2 = torch.nn.Linear(hidden_features, hidden_features)
        self.fc3 = torch.nn.Linear(hidden_features, hidden_features)
        self.fc4 = torch.nn.Linear(hidden_features, hidden_features)
        self.fc5 = torch.nn.Linear(hidden_features, hidden_features)
        self.fc6 = torch.nn.Linear(hidden_features, output_size)
        self.softmax_size = (4, int(2*K))
        self.wl = wl
        self.fl = fl

    def forward(self, x):
        
        x = torch.nn.functional.relu(self.fc1(x))
        x = fixed_point_quantize(x, self.wl, self.fl)
        x = torch.nn.functional.relu(self.fc2(x))
        x = fixed_point_quantize(x, self.wl, self.fl)
        x = torch.nn.functional.relu(self.fc3(x))
        x = fixed_point_quantize(x, self.wl, self.fl)
        x = torch.nn.functional.relu(self.fc4(x))
        x = fixed_point_quantize(x, self.wl, self.fl)
        x = torch.nn.functional.relu(self.fc5(x))
        x = self.fc6(x)
        x = fixed_point_quantize(x, self.wl, self.fl)
        
        x = x.reshape(x.shape[0], *self.softmax_size)

        return x


def train_model(dataloader: DataLoader, model: torch.nn.Module,
                loss_fn: torch.nn.Module, optimizer: torch.nn.Module, device):
    """
    Method to train model using pytorch.
    """
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.train()
    avg_train_loss = 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        # Compute
        y_hat = model(X.float())
        loss = loss_fn(y_hat, y)
        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        avg_train_loss += loss.item()
        if batch % 100 == 0:
            loss, current = loss.item(), (batch+1)*len(X)
            print(f'loss: {loss:.7f} [{current:d}/{size:d}]')
    avg_train_loss /= num_batches

    return avg_train_loss


def test_model(dataloader: DataLoader, model: torch.nn.Module,
               loss_fn:torch.nn.Module, device):
    """
    Method to test model using pytorch.
    """
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0
    accuracy = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            y_hat = model(X.float())
            test_loss += loss_fn(y_hat, y).item()
            accuracy += (y_hat.argmax(1) == y.argmax(1)).type(torch.float).sum().item()
    test_loss /= num_batches
    accuracy /= (size*64)
    msg = f'Test Error: \n Accuracy: {(100*accuracy):.2f}%, ' \
        + f'Avg Loss: {test_loss:.7f} \n'
    print(msg)
    
    return test_loss


def run_model(dataloader: DataLoader, model: torch.nn.Module, device):
    """Method to run model using pytorch"""
    size = len(dataloader.dataset)
    accuracy = 0
    ser = 0
    model.eval()
    with torch.no_grad():
        for X, y in tqdm(dataloader):
            X, y = X.to(device), y.to(device)
            y_hat = model(X.float())
            accuracy += (y_hat.argmax(1) == y.argmax(1)).type(torch.float).sum().item()
            ser += (y_hat.argmax(1) != y.argmax(1)).type(torch.float).sum().item()
    accuracy /= (size*64)
    ser /= (size*64)

    return ser


def run_model_quantized(dataloader: DataLoader, model: torch.nn.Module,
                        wl: int, fl: int, device):
    """Method to run quantized model.
    
    Args
    ----
    dataloader : DataLoader
        dataset loader
    model : torch.nn.Module
        model we want to run
    wl : int
        wordlength for our quantization
    fl : int
        fractional length
    device : str
        device where we wanto to compute
    """
    size = len(dataloader.dataset)
    accuracy = 0
    ser = 0
    model.eval()
    with torch.no_grad():
        for X, y in tqdm(dataloader):
            X = fixed_point_quantize(X, wl, fl).to(device)
            y = fixed_point_quantize(y, wl, fl).to(device)
            y_hat = model(X.float())
            accuracy += (y_hat.argmax(1) == y.argmax(1)).type(torch.float).sum().item()
            ser += (y_hat.argmax(1) != y.argmax(1)).type(torch.float).sum().item()
    accuracy /= (size*64)
    ser /= (size*64)

    return ser
