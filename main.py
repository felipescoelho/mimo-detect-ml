"""main.py

Script to run mimo signal detection.

luizfelipe.coelho@smt.ufrj.br
Out 7, 2024
"""


import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from src.model import (DatasetQAM, MLP, MLPQuantized, train_model, test_model,
                       run_model)
from src.utils import matrix_55_toeplitz, quantize_coefficients


plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'mathptmx',
    'font.size': 8
})


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', default='train', help='Operation mode',
                        type=str, choices=['train', 'run', 'test'])
    parser.add_argument('--K', type=int, default=32)
    parser.add_argument('--N', type=int, default=64)
    parser.add_argument('--mod_size', type=int, choices=[2, 4, 16], default=16)
    parser.add_argument('--ensemble', type=int, default=1250000)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=1000)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--n_run', type=int, default=250000)
    parser.add_argument('--snr_limits', type=str, default='7,14')
    parser.add_argument('--seed', type=int, default=None, help='Seed for RNG')

    return parser.parse_args()


if __name__ == '__main__':
    args = arg_parser()
    K = args.K
    N = args.N
    ensemble = args.ensemble
    mod_size = args.mod_size
    learning_rate = args.learning_rate
    n_epochs = args.epochs
    batch_size = args.batch_size
    snr_limits = [float(val) for val in args.snr_limits.split(',')]
    dataset_folder = 'datasets'
    model_folder = 'models'
    figures_folder = 'figures'
    golden_ratio = .5*(1+5**.5)
    width = 3.5
    height = width/golden_ratio
    if not os.path.isdir(dataset_folder):
        os.makedirs(dataset_folder)
    if not os.path.isdir(model_folder):
        os.makedirs(model_folder)
    if not os.path.isdir(figures_folder):
        os.makedirs(figures_folder)
    device = ('cuda' if torch.cuda.is_available() else
              'mps' if torch.backends.mps.is_available() else
              'cpu')
    match args.mode:
        case 'test':
            from src.vector_quatizer import mpgbp
            from math import floor
            v = torch.randn((N,))
            v_hat = mpgbp(v, 128, floor(N**.5), (-64, 64))
            print(v)
            print(v_hat)
        case 'train':
            train_path = os.path.join(dataset_folder, 'train_dataset.pt')
            test_path = os.path.join(dataset_folder, 'test_dataset.pt')
            channel_path = os.path.join(dataset_folder, 'channel_matrix.npy')
            model_path = os.path.join(model_folder, 'model_weights.pth')
            train_results_path = os.path.join(model_folder,
                                              'train_results.npz')
            if not os.path.isfile(channel_path):
                H = matrix_55_toeplitz(N, K)
                np.save(channel_path, H)
            else:
                H = np.load(channel_path)
            snr_values = np.arange(*snr_limits, 1)
            print('Train dataset:')
            train_dataset = DatasetQAM(K, N, mod_size, int(.8*ensemble), H,
                                       snr_values, dataset_path=train_path)
            print('Done!')
            print('Test dataset:')
            test_dataset = DatasetQAM(K, N, mod_size, int(.2*ensemble), H,
                                      snr_values, dataset_path=test_path)
            print('Done!')
            # Prepare data with loader:
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                                          shuffle=True)
            test_dataloader = DataLoader(test_dataset, batch_size=batch_size,
                                         shuffle=True)
            # Our train model:
            model = MLP(N, K).to(device)
            # Training Loop:
            loss_fn = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            train_loss = np.zeros((n_epochs,))
            test_loss = np.zeros((n_epochs,))
            for it in range(n_epochs):
                print(f'Epoch {it+1}\n---------------------------------------')
                train_loss[it] = train_model(train_dataloader, model, loss_fn,
                                             optimizer, device)
                test_loss[it] = test_model(test_dataloader, model, loss_fn,
                                           device)
            np.savez(train_results_path, train_loss=train_loss,
                     test_loss=test_loss)
            torch.save(model.state_dict(), model_path)

            fig = plt.figure(figsize=(width, height))
            ax = fig.add_subplot(111)
            ax.plot(range(n_epochs), train_loss, label='Train')
            ax.plot(range(n_epochs), test_loss, label='Validation')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Avg. Loss')
            ax.legend()
            fig.tight_layout()

            fig.savefig(os.path.join(figures_folder, 'train_loss.eps'),
                        format='eps', bbox_inches='tight')
            fig.savefig(os.path.join(figures_folder, 'train_loss.png'),
                        format='png', bbox_inches='tight')
        case 'run':
            channel_path = os.path.join(dataset_folder, 'channel_matrix.npy')
            model_path = os.path.join(model_folder, 'model_weights.pth')
            H = np.load(channel_path)
            state_dict = torch.load(model_path)
            model = MLP(N, K).to(device)
            model.load_state_dict(state_dict)
            snr_values = np.arange(*snr_limits, 1)
            MN_ratio_values = np.array((1., 3., 5., 8))
            ser = np.zeros((len(MN_ratio_values)+1, len(snr_values)))
            for idx_MN, MN_ratio in np.ndenumerate(MN_ratio_values):
                idx0 = idx_MN[0]
                quantized_model_path = os.path.join(
                    model_folder, f'model_weights_{MN_ratio}.pth'
                )
                model_quatized = MLP(N, K).to(device)
                if os.path.isfile(quantized_model_path):
                    state_dict_quantized = torch.load(model_path)
                    model_quatized.load_state_dict(state_dict_quantized)
                else:
                    state_dict_quantized = quantize_coefficients(
                        state_dict, MN_ratio, device
                    )
                    model_quatized.load_state_dict(state_dict_quantized)
                    torch.save(model_quatized.state_dict(),
                               quantized_model_path)
                for idx_snr, snr_dB in np.ndenumerate(snr_values):
                    idx1 = idx_snr[0]
                    dataset_path = os.path.join(dataset_folder,
                                                f'run_SNR_{snr_dB}.pt')
                    dataset = DatasetQAM(K, N, mod_size, args.n_run, H,
                                         np.array((snr_dB,)),
                                         dataset_path=dataset_path)
                    dataloader = DataLoader(dataset, batch_size=batch_size,
                                            shuffle=True)
                    print(f'Run quantized model M/N {MN_ratio} | SNR {snr_dB}')
                    ser[idx0+1, idx1] = run_model(dataloader, model_quatized,
                                                  device)
                    if idx_MN == 0:
                        ser[0, idx1] = run_model(dataloader, model, device)
            
            fig = plt.figure(figsize=(width, height))
            ax = fig.add_subplot(111)
            ax.plot(snr_values, ser[0, :], label='Full Precision')
            for idx, MN_ratio in np.ndenumerate(MN_ratio_values):
                ax.plot(snr_values, ser[idx+1, :], label=f'M/N = {MN_ratio}')
            ax.legend()
            ax.set_xlabel('SNR, dB')
            ax.set_ylabel('SER')
            fig.tight_layout()

            fig.savefig(os.path.join(figures_folder, 'ser.eps'),
                        format='eps', bbox_inches='tight')
            fig.savefig(os.path.join(figures_folder, 'ser.png'),
                        format='png', bbox_inches='tight')
