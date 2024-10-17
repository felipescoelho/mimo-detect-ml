"""main.py

Script to run mimo signal detection.

luizfelipe.coelho@smt.ufrj.br
Out 7, 2024
"""


import os
import argparse
import torch
import math
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from tqdm import tqdm
from torch.utils.data import DataLoader
from src.model import (DatasetQAM, MLP, MLPQuantized, train_model, test_model,
                       run_model, run_model_zf)
from src.utils import (matrix_55_toeplitz, quantize_coefficients_mpgbp,
                       quantize_coefficients_naive_mpgbp)


plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'mathptmx',
    'font.size': 8
})


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', default='train', help='Operation mode',
                        type=str, choices=['train', 'run', 'test', 'run2',
                                           'run3'])
    parser.add_argument('--K', type=int, default=32)
    parser.add_argument('--N', type=int, default=64)
    parser.add_argument('--mod_size', type=int, choices=[2, 4, 16], default=16)
    parser.add_argument('--ensemble', type=int, default=1250000)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=1000)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--n_run', type=int, default=1000)
    parser.add_argument('--snr_limits', type=str, default='7,14')
    parser.add_argument('--seed', type=int, default=None, help='Seed for RNG')

    return parser.parse_args()


def polygon_under_graph(x, y):
    """
    Construct the vertex list which defines the polygon filling the
    space under the (x, y) line graph. This assumes x is in ascending
    order.
    """

    return [(x[0], 0.), *zip(x, y), (x[-1], 0.)]


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
    results_folder = 'results'
    golden_ratio = .5*(1+5**.5)
    width = 3.5
    height = width/golden_ratio
    if not os.path.isdir(dataset_folder):
        os.makedirs(dataset_folder)
    if not os.path.isdir(model_folder):
        os.makedirs(model_folder)
    if not os.path.isdir(figures_folder):
        os.makedirs(figures_folder)
    if not os.path.isdir(results_folder):
        os.makedirs(results_folder)
    device = ('cuda' if torch.cuda.is_available() else
              'mps' if torch.backends.mps.is_available() else
              'cpu')
    match args.mode:
        case 'test':
            from src.utils import twos_complement
            from src.vector_quatizer import mpgbp
            from math import sqrt, floor
            L = 16
            mse_mpgbp_list = []
            mse_naive_list = []
            min_wl, max_wl = 2, 33
            ensemble = 100
            for wl in tqdm(range(min_wl, max_wl)):
                mse_mpgbp = []
                mse_naive = []
                for it in range(ensemble):
                    a = torch.randn((L,), device='cuda')
                    approx_naive = torch.zeros((L,), device='cuda')
                    spt_total = 0
                    for idx in range(L):
                        normalizer = 2**torch.ceil(
                            torch.log2(a[idx].abs())
                        ).item()
                        approx, spt_count = twos_complement(
                            a[idx].item()/normalizer, wl, 'cuda'
                        )
                        spt_total += spt_count
                        approx_naive[idx] = approx*normalizer
                    normalizer = 2**torch.ceil(
                        torch.log2(a.abs().max())
                    ).item()
                    approx_mpgbp = mpgbp(
                        a/normalizer, spt_total, floor(sqrt(L)), 'cuda'
                    )*normalizer
                    mse_mpgbp.append(torch.norm(a-approx_mpgbp, p=2).item()/L)
                    mse_naive.append(torch.norm(a-approx_naive, p=2).item()/L)
                mse_mpgbp_list.append(sum(mse_mpgbp)/ensemble)
                mse_naive_list.append(sum(mse_naive)/ensemble)
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(range(min_wl, max_wl), mse_mpgbp_list, label='mpgbp')
            ax.plot(range(min_wl, max_wl), mse_naive_list, label='naive')
            ax.set_xlabel('wordlength')
            ax.set_ylabel('MSE')
            ax.legend()
            fig.tight_layout()
            fig.savefig(os.path.join(figures_folder, 'mse.png'))
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
            # Experiment Quantization:
            channel_path = os.path.join(dataset_folder, 'channel_matrix.npy')
            model_path = os.path.join(model_folder, 'model_weights.pth')
            H = np.load(channel_path)
            state_dict = torch.load(model_path)
            model = MLP(N, K).to(device)
            model.load_state_dict(state_dict)
            snr_values = np.arange(*snr_limits, 1)
            MN_ratio_values = np.array((1., 8, 16.))
            ser_zf = np.zeros((len(snr_values),))
            ser = np.zeros((len(MN_ratio_values)+1, len(snr_values)))
            print('Load Datasets ...')
            dataset_list = [
                DatasetQAM(K, N, mod_size, args.n_run, H,
                           np.array((snr_values[idx],)),
                           dataset_path=os.path.join(
                               dataset_folder,
                               f'run_SNR_{snr_values[idx]}.pt'
                           )) for idx in tqdm(range(len(snr_values)))
            ]
            print('Done!')
            for idx_MN, MN_ratio in np.ndenumerate(MN_ratio_values):
                idx0 = idx_MN[0]
                quantized_model_path = os.path.join(
                    model_folder, f'model_weights_SPT_{int(MN_ratio)}.pth'
                )
                model_quatized = MLP(N, K).to(device)
                if os.path.isfile(quantized_model_path):
                    state_dict_quantized = torch.load(quantized_model_path)
                    model_quatized.load_state_dict(state_dict_quantized)
                else:
                    state_dict_quantized = quantize_coefficients_mpgbp(
                        state_dict, MN_ratio, device
                    )
                    model_quatized.load_state_dict(state_dict_quantized)
                    torch.save(model_quatized.state_dict(),
                               quantized_model_path)
                for idx_snr, snr_dB in np.ndenumerate(snr_values):
                    results_path = os.path.join(
                        results_folder,
                        f'results_SPT_{int(MN_ratio)}_SNR_{snr_dB}.npy'
                    )
                    results_zf_path = os.path.join(
                        results_folder, f'results_zf_SNR_{snr_dB}.npy'
                    )
                    results_fp_path = os.path.join(
                        results_folder, f'results_fp_SNR_{snr_dB}.npy'
                    )
                    idx1 = idx_snr[0]
                    dataloader = DataLoader(
                        dataset_list[idx1], batch_size=batch_size, shuffle=True
                    )
                    if idx0 == 0:
                        if os.path.isfile(results_zf_path):
                            ser_zf_idx = np.load(results_zf_path)
                        else:
                            print(f'Run Zero forcing model | SNR {snr_dB}')
                            ser_zf_idx = run_model_zf(dataloader, H, device)
                            np.save(results_zf_path, ser_zf_idx)
                        ser_zf[idx1] = ser_zf_idx
                        if os.path.isfile(results_fp_path):
                            ser_fp = np.load(results_fp_path)
                        else:
                            print(f'Run full precision model | SNR {snr_dB}')
                            ser_fp = run_model(dataloader, model, device)
                            np.save(results_fp_path, ser_fp)
                        ser[0, idx1] = ser_fp
                    if os.path.isfile(results_path):
                        ser_idx = np.load(results_path)
                    else:
                        print(f'Run quantized model M_max/L {int(MN_ratio)}'
                              + f'| SNR {snr_dB}')
                        ser_idx = run_model(dataloader, model_quatized, device)
                        np.save(results_path, ser_idx)
                    ser[idx0+1, idx1] = ser_idx

            fig = plt.figure(figsize=(width, height))
            ax = fig.add_subplot(111)
            ax.semilogy(snr_values, ser_zf,
                        label='$\\textrm{ZF}_{\\textrm{genie}}$')
            ax.semilogy(snr_values, ser[0, :], label='Full Precision')
            for idx, MN_ratio in np.ndenumerate(MN_ratio_values):
                ax.semilogy(snr_values, ser[idx[0]+1, :],
                            label='$M_{\\textrm{max}}/L$ = ' \
                                + f'{int(MN_ratio)}')
            ax.legend(ncols=1)
            ax.grid()
            ax.set_xlabel('SNR, dB')
            ax.set_ylabel('SER')
            fig.tight_layout()

            fig.savefig(os.path.join(figures_folder, 'ser.eps'),
                        format='eps', bbox_inches='tight')
            fig.savefig(os.path.join(figures_folder, 'ser.png'),
                        format='png', bbox_inches='tight')
        case 'run2':
            # Compare with naive method
            wl = 8
            channel_path = os.path.join(dataset_folder, 'channel_matrix.npy')
            model_path = os.path.join(model_folder, 'model_weights.pth')
            H = np.load(channel_path)
            state_dict = torch.load(model_path)
            model_naive = MLP(N, K).to(device)
            model_mpgbp = MLP(N, K).to(device)
            state_dict_naive, state_dict_mpgbp = \
                quantize_coefficients_naive_mpgbp(state_dict, wl, device)
            model_naive.load_state_dict(state_dict_naive)
            model_mpgbp.load_state_dict(state_dict_mpgbp)
            snr_values = np.arange(*snr_limits, 1)
            ser = np.zeros((2, len(snr_values)))
            print('Load Datasets ...')
            dataset_list = [
                DatasetQAM(K, N, mod_size, args.n_run, H,
                           np.array((snr_values[idx],)),
                           dataset_path=os.path.join(
                               dataset_folder,
                               f'run_SNR_{snr_values[idx]}.pt'
                           )) for idx in tqdm(range(len(snr_values)))
            ]
            print('Done!')
            for idx_snr, snr_dB in np.ndenumerate(snr_values):
                idx0 = idx_snr[0]
                dataloader = DataLoader(dataset_list[idx0],
                                        batch_size=batch_size, shuffle=True)
                print(f'Run MPGBP model SNR {snr_dB}.')
                ser[0, idx0] = run_model(dataloader, model_mpgbp, device)
                print(f'Run naive model SNR {snr_dB}.')
                ser[1, idx0] = run_model(dataloader, model_naive, device)
            
            fig = plt.figure(figsize=(width, height))
            ax = fig.add_subplot(111)
            ax.semilogy(snr_values, ser[0, :], label='MPGBP')
            ax.semilogy(snr_values, ser[1, :], label='Naive')
            ax.legend(ncols=1)
            ax.grid()
            ax.set_xlabel('SNR, dB')
            ax.set_ylabel('SER')
            fig.tight_layout()

            fig.savefig(os.path.join(figures_folder, 'ser2.eps'),
                        format='eps', bbox_inches='tight')
            fig.savefig(os.path.join(figures_folder, 'ser2.png'),
                        format='png', bbox_inches='tight')
        case 'run3':
            # Layer sensibility test
            channel_path = os.path.join(dataset_folder, 'channel_matrix.npy')
            model_path = os.path.join(model_folder, 'model_weights.pth')
            model_quantized_path = os.path.join(model_folder,
                                                'model_weights.pth')
            H = np.load(channel_path)
            state_dict = torch.load(model_path)
            state_dict_quantized = torch.load(model_quantized_path)
            model_list = []
            layer_list = []
            for layer_name in set([key.split('.')[0] for key in state_dict.keys()]):
                key_list = [key for key in state_dict.keys() if
                            key.startswith(layer_name)]
                spam = state_dict.copy()
                for key in key_list:
                    spam[key] = state_dict_quantized[key]
                model = MLP(N, K)
                model.load_state_dict(spam)
                model.to(device)
                model_list.append(model)
                layer_list.append(layer_name)
            sorting_index = sorted(range(len(layer_list)),
                                   key=layer_list.__getitem__)
            snr_values = np.arange(*snr_limits, 1)
            print('Load Datasets ...')
            dataset_list = [
                DatasetQAM(K, N, mod_size, args.n_run, H,
                           np.array((snr_values[idx],)),
                           dataset_path=os.path.join(
                               dataset_folder,
                               f'run_SNR_{snr_values[idx]}.pt'
                           )) for idx in tqdm(range(len(snr_values)))
            ]
            print('Done!')
            ser = np.zeros((len(layer_list), len(snr_values)))
            for idx_snr, snr_dB in np.ndenumerate(snr_values):
                idx0 = idx_snr[0]
                dataloader = DataLoader(dataset_list[idx0],
                                        batch_size=batch_size, shuffle=True)
                for idx1, model in enumerate(model_list):
                    layer_num = sorting_index[idx1]+1
                    print(f'Run model quantized layer {layer_num}')
                    ser[layer_num-1, idx0] = run_model(dataloader, model,
                                                       device)
            
            fig = plt.figure(figsize=(width, height))
            ax = fig.add_subplot(projection='3d')
            # verts[i] is a list of (x, y) pairs defining polygon i.
            verts = [polygon_under_graph(
                snr_values, ser[l, :])
                for l in range(len(layer_list))
            ]
            facecolors = plt.colormaps['viridis_r'](np.linspace(0, 1,
                                                                len(verts)))
            poly = PolyCollection(verts, facecolors=facecolors, alpha=.7)
            ax.add_collection3d(poly, zs=range(1, len(layer_list)+1),
                                zdir='y')
            ax.set_zlabel('log')
            ax.set(xlabel='SNR, dB', ylabel='Layer', zlabel='SER')
            fig.tight_layout()
            fig_eps_path = os.path.join(figures_folder, 'ser_run3.eps')
            fig_png_path = os.path.join(figures_folder, 'ser_run3.png')
            fig.savefig(fig_eps_path, format='eps', bbox_inches='tight')
                    
                    