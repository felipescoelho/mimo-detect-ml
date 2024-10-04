"""model.py

Script to contain autoencoder.

luizfelipe.coelho@smt.ufrj.br
Sep 17, 2024
"""


import os
import torch
from math import log2
from torch.utils.data import Dataset, DataLoader
from .qam import qammod
from .ofdm import ofdmmod


class OFDMDatasetAutoEncoder(Dataset):
    def __init__(self, N: int, M: int, mod_size: int, n_frames: int,
                 seed: int | None = None, dataset_path: str | None = None):
        """Generates the dataset for the autoencoder.
        
        Args
        ----
        N : int
            length of the DFT
        M : int
            number of blocks in OFDM frame
        mod_size : int
            modulation size for QAM
        n_frames : int
            number of frames in dataset
        seed : int | None = None
            seed for RNG
        dataset_path : str | None = None
            path to dataset
        """

        super().__init__()
        if seed is not None:
            torch.random.seed(seed)
        if os.path.isfile(dataset_path):
            self.source_bits, self.modulated_bits, self.ofdm_frames = \
                torch.load(dataset_path)
        else:
            self.source_bits = torch.randint(
                2, size=(n_frames, int(N*M*log2(mod_size)))
            )
            self.modulated_bits = qammod(self.source_bits, mod_size)
            self.ofdm_frames = ofdmmod(self.modulated_bits, N)
            torch.save([self.source_bits, self.modulated_bits,
                        self.ofdm_frames], dataset_path)

    def __len__(self):
        return self.modulated_bits.shape[0]
    
    def __getitem__(self, index):
        x = self.modulated_bits[index, :, :]
        label = self.modulated_bits[index, :, :]
        return x, label


class MLP(torch.nn.Module):
    """Multi-layer perceptron."""

    def __init__(self):
        super.__init__()
        


class AutoEncoder(torch.nn.Module):
    """"""

    def __init__(self, input_size, hidden_layer, latent_layer):
        super().__init__()

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_layer),
            torch.nn.Sigmoid(),
            torch.nn.Linear(hidden_layer, latent_layer)
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(latent_layer, hidden_layer),
            torch.nn.Sigmoid(),
            torch.nn.Linear(hidden_layer, input_size)
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        return decoded


def train(dataloader: DataLoader, model: torch.nn.Module,
          loss_fn: torch.nn.Module, optimizer: torch.nn.Module, device):
    """
    Method to train model using pytorch.
    """
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        # Compute
        y_hat = model(X)
        loss = loss_fn(y_hat, y)
        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch+1)*len(X)
            print(f'loss: {loss:.4f} [{current:d}/{size:d}]')


def test(dataloader: DataLoader, model: torch.nn.Module,
         loss_fn:torch.nn.Module, device):
    """
    Method to test model using pytorch.
    """
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            y_hat = model(X)
            # test_loss += 
