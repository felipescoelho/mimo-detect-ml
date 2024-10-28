"""model.py

Script to contain autoencoder.

luizfelipe.coelho@smt.ufrj.br
Sep 17, 2024
"""


import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from .utils import onehot_demap, decide


class MLP_QPSK(torch.nn.Module):
    """Multilayer Perceptron"""

    def __init__(self, N, K):
        super().__init__()
        input_size = int(2*N)
        hidden_features = int(10*K)
        output_size = int(4*K)
        self.fc1 = torch.nn.Linear(input_size, hidden_features)
        self.fc2 = torch.nn.Linear(hidden_features, hidden_features)
        self.fc3 = torch.nn.Linear(hidden_features, hidden_features)
        self.fc4 = torch.nn.Linear(hidden_features, hidden_features)
        self.fc5 = torch.nn.Linear(hidden_features, hidden_features)
        self.fc6 = torch.nn.Linear(hidden_features, output_size)
        self.logit_shape = (2, int(2*K))

    def forward(self, x):
        
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = torch.nn.functional.relu(self.fc3(x))
        x = torch.nn.functional.relu(self.fc4(x))
        x = torch.nn.functional.relu(self.fc5(x))
        x = self.fc6(x)
        x = x.reshape(x.shape[0], *self.logit_shape)

        return x


class MLP_BPSK(torch.nn.Module):
    """Multilayer Perceptron"""

    def __init__(self, N, K):
        super().__init__()
        input_size = N
        hidden_features = int(10*K)
        output_size = int(2*K)
        self.fc1 = torch.nn.Linear(input_size, hidden_features)
        self.fc2 = torch.nn.Linear(hidden_features, hidden_features)
        self.fc3 = torch.nn.Linear(hidden_features, hidden_features)
        self.fc4 = torch.nn.Linear(hidden_features, hidden_features)
        self.fc5 = torch.nn.Linear(hidden_features, hidden_features)
        # self.fc6 = torch.nn.Linear(hidden_features, hidden_features)
        self.fc7 = torch.nn.Linear(hidden_features, output_size)
        self.logit_shape = (2, K)

    def forward(self, x):
        
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = torch.nn.functional.relu(self.fc3(x))
        x = torch.nn.functional.relu(self.fc4(x))
        x = torch.nn.functional.relu(self.fc5(x))
        # x = torch.nn.functional.relu(self.fc6(x))
        x = self.fc7(x)
        x = x.reshape(x.shape[0], *self.logit_shape)

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
    ber = 0
    model.eval()
    num_batches = len(dataloader)
    batch_size = dataloader.batch_size
    with torch.no_grad():
        for X, y in tqdm(dataloader):
            X, y = X.to(device), y.to(device)
            y_hat = model(X.float()).argmax(1)
            y_true = y.argmax(1)
            ber += (y_hat != y_true).sum().item()/(batch_size*y_true.shape[1])
    ber /= num_batches
    print(ber)

    return ber


def run_zf_qpsk(dataloader: DataLoader, H: np.ndarray, device):
    """Method to run model using pytorch"""
    ber = 0
    H = torch.from_numpy(H).to(device)
    H_inv = torch.linalg.inv(torch.conj(H).T@H)
    for X, y in tqdm(dataloader.dataset):
        X, y = X.to(device), y.to(device)
        X_complex = X[:H.shape[0]] + 1j*X[H.shape[0]:]
        y_hat = H_inv @ torch.conj(H).T @ X_complex
        y_pred = torch.concat((y_hat.real, y_hat.imag), axis=-1)
        y_pred[y_pred > 0] = 1
        y_pred[y_pred < 0] = 0
        y_true = y.argmax(0)
        ber += (y_pred != y_true).sum().item()/len(y_pred)
    ber /= len(dataloader.dataset)
    print(ber)

    return ber


def run_zf_bpsk(dataloader: DataLoader, H: np.ndarray):
    """Method to run model using pytorch"""
    ser = 0
    H = torch.from_numpy(H)
    H_inv = torch.linalg.inv(torch.conj(H).T@H)
    spam = torch.tensor([0., 1.])
    for X, y in tqdm(dataloader.dataset):
        y = y.float()
        y_hat = H_inv @ H.T @ X
        y_pred = torch.tensor([1 if val.item() > 0 else -1 for val in y_hat])
        y_true = torch.tensor([1 if torch.equal(y[:, idx], spam)
                               else -1 for idx in range(y.shape[1])])
        ser += (y_pred != y_true).sum().item()/len(y_pred)
    ser /= len(dataloader.dataset)
    print(ser)

    return ser


def run_model_bpsk(dataloader: DataLoader, model: torch.nn.Module, device: str):
    """Method to run model using pytorch"""
    ser = 0
    model.eval()
    num_batches = len(dataloader)
    batch_size = dataloader.batch_size
    with torch.no_grad():
        for X, y in tqdm(dataloader):
            X, y = X.to(device), y.to(device)
            y_hat = model(X.float()).argmax(1)
            y_true = y.argmax(1)
            ser += (y_hat != y_true).sum().item()/(batch_size*y_true.shape[1])
    ser /= num_batches
    print(ser)

    return ser


def run_model_bpsk2(dataloader: DataLoader, model: torch.nn.Module, device: str):
    """Method to run model using pytorch"""
    ser = 0
    model.eval()
    num_batches = len(dataloader)
    batch_size = dataloader.batch_size
    with torch.no_grad():
        for X, y in tqdm(dataloader):
            X, y = X.to(device), y.to(device)
            y_hat = model(X.float())
            y_hat[y_hat > 0] = 1
            y_hat[y_hat <= 0] = -1
            ser += (y_hat != y).sum().item()/(batch_size*y.shape[1])
    ser /= num_batches
    print(ser)

    return ser
