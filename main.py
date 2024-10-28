"""main.py

Script to run mimo signal detection.

luizfelipe.coelho@smt.ufrj.br
Out 7, 2024
"""


import os
import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from tqdm import tqdm
from torch.utils.data import DataLoader
from src.dataset import DatasetQAM, DatasetBPSK, DatasetQPSK
from src.model import (MLP_QPSK, run_model, run_zf_qpsk,
                       run_zf_bpsk, MLP_BPSK, run_model_bpsk)
from src.utils import (matrix_55_toeplitz, quantize_coefficients_mpgbp,
                       quantize_coefficients_naive_mpgbp, matrix_55_toeplitz_real)


plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'mathptmx',
    'font.size': 8
})


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', default='train', help='Operation mode',
                        type=str, choices=['train', 'train_bpsk', 'run_bpsk',
                                           'run2_bpsk', 'run3_bpsk',
                                           'run', 'test', 'run2', 'run3'])
    parser.add_argument('--K', type=int, default=16)
    parser.add_argument('--N', type=int, default=32)
    parser.add_argument('--mod_size', type=int, choices=[2, 4, 16], default=16)
    parser.add_argument('--train_epochs', type=int, default=200)
    parser.add_argument('--train_iter', type=int, default=5000)
    parser.add_argument('--train_batchsize', type=int, default=1000)
    parser.add_argument('--learning_rate', type=float, default=0.0008)
    parser.add_argument('--decay_lr_factor', type=float, default=.97)
    parser.add_argument('--decay_lr_iter', type=int, default=1500)
    parser.add_argument('--test_iter', type=int, default=200)
    parser.add_argument('--test_batchsize', type=int, default=1000)
    parser.add_argument('--validation_tol', type=float, default=0.0001)
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
    rng = np.random.default_rng(seed=args.seed)
    test_ensemble = int(args.test_iter * args.test_batchsize)
    # ensemble = args.ensemble
    # mod_size = args.mod_size
    # n_epochs = args.epochs
    # batch_size = args.batch_size
    snr_limits = [float(val) for val in args.snr_limits.split(',')]
    dataset_folder = 'datasets'
    model_folder = 'models'
    figures_folder = 'figures'
    results_folder = 'results'
    checkpoints_folder = os.path.join(model_folder, 'checkpoints')
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
    if not os.path.isdir(checkpoints_folder):
        os.makedirs(checkpoints_folder)
    device = ('cuda' if torch.cuda.is_available() else
              'mps' if torch.backends.mps.is_available() else
              'cpu')
    match args.mode:
        case 'test':
            N = 32
            K = 16
            from src.utils import matrix_55_toeplitz
            snr_arr = np.arange(8, 13+1, 1)
            ensemble = 10000
            H = matrix_55_toeplitz(N, K)
            H_inv = np.linalg.inv(np.conj(H.T) @ H) @ np.conj(H.T)
            ber = np.zeros((len(snr_arr)))
            for snr_idx in range(len(snr_arr)):
                dataset = DatasetQPSK(K, N, ensemble, H, (snr_arr[snr_idx],))
                dataloader = DataLoader(dataset, batch_size=1000, shuffle=True)
                for X, y in dataloader.dataset:
                    y_true = y.argmax(0)
                    X = X[:N] + 1j*X[N:]
                    yy = torch.from_numpy(H_inv) @ X
                    y_pred = torch.concat((yy.real, yy.imag), axis=-1)
                    y_pred[y_pred > 0] = 1
                    y_pred[y_pred < 0] = 0
                    # import pdb;pdb.set_trace()
                    ber[snr_idx] += (y_pred != y_true).sum().item()/(2*K)
                # ser[snr_idx] = run_zf_bpsk(dataloader, H, device)
                ber[snr_idx] /= len(dataloader.dataset)
                print(ber)
            fig = plt.figure(figsize=(width, height))
            ax = fig.add_subplot(111)
            ax.semilogy(snr_arr, ber, label='ZF')
            ax.set_xlabel('SNR, dB')
            ax.set_ylabel('SER')
            ax.grid()
            ax.legend()
            fig.tight_layout()
            fig.savefig(os.path.join(figures_folder, 'test.png'), bbox_inches='tight')
        case 'train_bpsk':
            train_batch_size = args.train_batchsize
            test_batch_size = args.test_batchsize
            train_ensemble = int(args.train_iter * args.train_batchsize)
            test_ensemble = int(args.test_iter * args.test_batchsize)
            learning_rate = args.learning_rate
            channel_path = os.path.join(dataset_folder,
                                        'channel_matrix_bpsk.npy')
            model_path = os.path.join(model_folder, 'model_weights_bpsk.pth')
            if not os.path.isfile(channel_path):
                H = matrix_55_toeplitz_real(N, K)
                np.save(channel_path, H)
            else:
                H = np.load(channel_path)
            snr_values = np.arange(snr_limits[0]+1, snr_limits[1], 1)
            test_datasets = [
                DatasetBPSK(K, N, test_ensemble, H, (snr_values[idx],),
                            dataset_path=os.path.join(
                                dataset_folder,
                                f'test_bpsk_SNR_{snr_values[idx]}.pt'
                            )) for idx in range(len(snr_values))
            ]
            model = MLP_BPSK(N, K).to(device)
            loss_fn = torch.nn.CrossEntropyLoss()
            # loss_fn = torch.nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=args.decay_lr_iter, 
                gamma=args.decay_lr_factor
            )
            test_results = np.zeros((args.train_epochs, len(snr_values)))
            patience = 0
            for epoch in range(args.train_epochs):
                print(f'Run Epoch {epoch+1}.')
                dataset_path = os.path.join(
                    dataset_folder,
                    f'train_dataset_bpsk_epoch_{epoch}_triangular.pt'
                )
                dataset = DatasetBPSK(K, N, train_ensemble, H, snr_limits,
                                      dataset_path=dataset_path)
                dataloader = DataLoader(dataset, train_batch_size, True)
                model.train()
                for it, (X, y) in enumerate(dataloader):
                    X, y = X.to(device), y.to(device)
                    y_hat = model(X.float())
                    loss = loss_fn(y_hat, y.float())
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    if it % 100 == 0:
                        print(f'Loss: {loss.item():.7f} [{it+1:d}/'
                            + f'{args.train_iter:d}]')
                    scheduler.step()
                test_result = np.zeros((len(snr_values),))
                model.eval()
                for idx in range(len(snr_values)):
                    test_dataloader = DataLoader(test_datasets[idx],
                                                 test_batch_size, True)
                    ber = 0
                    with torch.no_grad():
                        for X, y in test_dataloader:
                            X, y = X.to(device), y.to(device)
                            y_hat = model(X.float()).argmax(1)
                            y_true = y.argmax(1)
                            ber += (y_hat != y_true).sum().item()/K
                    test_result[idx] = ber/test_ensemble
                test_results[epoch, :] = test_result
                checkpoint_path = os.path.join(checkpoints_folder,
                                               f'model_bpsk_epoch_{epoch}.pth')
                torch.save(model.state_dict(), checkpoint_path)
                print(test_results[:epoch+1])
                if epoch > 0:
                    val_crit = np.mean(
                        (test_results[epoch-1, :] - test_results[epoch, :])
                        / test_results[epoch-1, :]
                    )
                    if val_crit <= 0:
                        patience += 1
                        if patience == 5:
                            print('No advance!')
                            state_dict = torch.load(
                                os.path.join(checkpoints_folder,
                                            f'model_bpsk_epoch_{epoch-patience}.pth')
                            )
                            model.load_state_dict(state_dict)
                            break
                    else:
                        patience = 0
                
            torch.save(model.state_dict(), model_path)
        case 'run_bpsk':
            # Experiment Quantization:
            ensemble = int(args.test_iter * args.test_batchsize)
            batch_size = args.test_batchsize
            channel_path = os.path.join(dataset_folder,
                                        'channel_matrix_bpsk.npy')
            model_path = os.path.join(model_folder, 'model_weights_bpsk.pth')
            H = np.load(channel_path)
            state_dict = torch.load(model_path)
            model = MLP_BPSK(N, K).to(device)
            model.load_state_dict(state_dict)
            snr_values = np.arange(snr_limits[0]+1, snr_limits[1], 1)
            MN_ratio_values = np.array((1., 1.5, 2.))
            ser_zf = np.zeros((len(snr_values),))
            ser_fp = np.zeros((len(snr_values),))
            ser_qt = np.zeros((len(MN_ratio_values), len(snr_values)))
            print('Load Datasets ...')
            dataset_list = [
                DatasetBPSK(K, N, ensemble, H, (snr_values[idx],),
                            dataset_path=os.path.join(
                               dataset_folder,
                               f'test_bpsk_SNR_{snr_values[idx]}.pt'
                           )) for idx in tqdm(range(len(snr_values)))
            ]
            print('Done!')
            for idx_MN, MN_ratio in np.ndenumerate(MN_ratio_values):
                idx0 = idx_MN[0]
                quantized_model_path = os.path.join(
                    model_folder,
                    f'model_weights_bpsk_SPT_ratio_{MN_ratio}.pth'
                )
                model_quatized = MLP_BPSK(N, K).to(device)
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
                        f'results_SPT_{MN_ratio}_SNR_{snr_dB}_bpsk.npy'
                    )
                    results_zf_path = os.path.join(
                        results_folder, f'results_zf_SNR_{snr_dB}_bpsk.npy'
                    )
                    results_fp_path = os.path.join(
                        results_folder, f'results_fp_SNR_{snr_dB}_bpsk.npy'
                    )
                    idx1 = idx_snr[0]
                    dataloader = DataLoader(dataset_list[idx1],
                                            batch_size=batch_size, shuffle=True)
                    if idx0 == 0:
                        if os.path.isfile(results_zf_path):
                            ser_zf_spam = np.load(results_zf_path)
                        else:
                            print(f'Run Zero force model | SNR {snr_dB}')
                            ser_zf_spam = run_zf_bpsk(dataloader, H)
                            np.save(results_zf_path, ser_zf_spam)
                        ser_zf[idx1] = ser_zf_spam
                        if os.path.isfile(results_fp_path):
                            ser_fp_spam = np.load(results_fp_path)
                        else:
                            print(f'Run full precision model | SNR {snr_dB}')
                            ser_fp_spam = run_model_bpsk(dataloader, model,
                                                         device)
                            np.save(results_fp_path, ser_fp_spam)
                        ser_fp_spam = run_model_bpsk(dataloader, model,
                                                         device)
                        ser_fp[idx1] = ser_fp_spam
                    if os.path.isfile(results_path):
                        ser_qt_spam = np.load(results_path)
                    else:
                        print(f'Run quantized model M_max/L {MN_ratio}'
                              + f'| SNR {snr_dB}')
                        ser_qt_spam = run_model_bpsk(dataloader,
                                                     model_quatized, device)
                        np.save(results_path, ser_qt_spam)
                    ser_qt[idx0, idx1] = ser_qt_spam
            ser_sd = np.array([0.0147166, 0.0062166, 0.00218, 5.0333e-04,
                               6.333e-05, 1.4e-05])
            ser_zf_didier=np.array([0.0330367, 0.0203466, 0.0105633,
                                    0.004863, 0.00176, 5.633e-04])
            fig = plt.figure(figsize=(width, height))
            ax = fig.add_subplot(111)
            ax.semilogy(snr_values, ser_zf, marker='s', label='ZF')
            ax.semilogy(snr_values, ser_zf_didier, label='ZF Didier')
            ax.semilogy(snr_values, ser_fp, marker='<', label='Full Precision')
            ax.semilogy(snr_values, ser_sd, marker='d', label='SD')
            marker_list = ['p', '*', 'X']
            for idx, MN_ratio in np.ndenumerate(MN_ratio_values):
                ax.semilogy(snr_values, ser_qt[idx[0], :],
                            marker=marker_list[idx[0]],
                            label='$M_{\scriptsize\\textrm{max}}/L$ = ' \
                                + f'{MN_ratio}')
            ax.legend(ncols=1)
            ax.grid()
            ax.set_xlabel('SNR, dB')
            ax.set_ylabel('BER')
            fig.tight_layout()

            fig.savefig(os.path.join(figures_folder, 'ber.eps'),
                        format='eps', bbox_inches='tight')
            fig.savefig(os.path.join(figures_folder, 'ber.png'),
                        format='png', bbox_inches='tight')
        case 'run2_bpsk':
            # Compare with naive method
            wl = 6
            ensemble = int(args.test_iter*args.test_batchsize)
            batch_size = args.test_batchsize
            channel_path = os.path.join(dataset_folder, 'channel_matrix_bpsk.npy')
            model_path = os.path.join(model_folder, 'model_weights_bpsk.pth')
            H = np.load(channel_path)
            state_dict = torch.load(model_path)
            model_full = MLP_BPSK(N, K).to(device)
            model_naive = MLP_BPSK(N, K).to(device)
            model_mpgbp = MLP_BPSK(N, K).to(device)
            state_dict_naive, state_dict_mpgbp = \
                quantize_coefficients_naive_mpgbp(state_dict, wl, device)
            model_naive.load_state_dict(state_dict_naive)
            model_mpgbp.load_state_dict(state_dict_mpgbp)
            model_full.load_state_dict(state_dict)
            snr_values = np.arange(snr_limits[0]+1, snr_limits[1], 1)
            ser = np.zeros((4, len(snr_values)))
            print('Load Datasets ...')
            dataset_list = [
                DatasetBPSK(K, N, ensemble, H,
                           np.array((snr_values[idx],)),
                           dataset_path=os.path.join(
                               dataset_folder,
                               f'test_bpsk_SNR_{snr_values[idx]}.pt'
                           )) for idx in tqdm(range(len(snr_values)))
            ]
            print('Done!')
            for idx_snr, snr_dB in np.ndenumerate(snr_values):
                idx0 = idx_snr[0]
                dataloader = DataLoader(dataset_list[idx0],
                                        batch_size=batch_size, shuffle=True)
                print(f'Run MPGBP model SNR {snr_dB}.')
                ser[0, idx0] = run_model_bpsk(dataloader, model_mpgbp, device)
                print(f'Run naive model SNR {snr_dB}.')
                ser[1, idx0] = run_model_bpsk(dataloader, model_naive, device)
                print(f'Run full precision SNR {snr_dB}.')
                ser[2, idx0] = run_model_bpsk(dataloader, model_full, device)
                print(f'Run zero forcing SNR {snr_dB}.')
                ser[3, idx0] = run_zf_bpsk(dataloader, H)
            
            fig = plt.figure(figsize=(width, height))
            ax = fig.add_subplot(111)
            ax.semilogy(snr_values, ser[3, :], marker='s', label='ZF')
            ax.semilogy(snr_values, ser[2, :], marker='<', label='Full Precision')
            ax.semilogy(snr_values, ser[0, :], marker='p', label='MPGBP')
            ax.semilogy(snr_values, ser[1, :], marker='*', label='Naïve (CSD)')
            ax.legend(ncols=1)
            ax.grid()
            ax.set_xlabel('SNR, dB')
            ax.set_ylabel('BER')
            fig.tight_layout()

            fig.savefig(os.path.join(figures_folder, 'ber2.eps'),
                        format='eps', bbox_inches='tight')
            fig.savefig(os.path.join(figures_folder, 'ber2.png'),
                        format='png', bbox_inches='tight')
        case 'run3_bpsk':
            # Layer sensibility test
            ensemble = int(args.test_iter*args.test_batchsize)
            batch_size = args.test_batchsize
            channel_path = os.path.join(dataset_folder, 'channel_matrix_bpsk.npy')
            model_path = os.path.join(model_folder, 'model_weights_bpsk.pth')
            model_quantized1_path = os.path.join(
                model_folder, 'model_weights_bpsk_SPT_ratio_1.0.pth'
            )
            model_quantized2_path = os.path.join(
                model_folder, 'model_weights_bpsk_SPT_ratio_1.5.pth'
            )
            model_quantized3_path = os.path.join(
                model_folder, 'model_weights_bpsk_SPT_ratio_2.0.pth'
            )
            H = np.load(channel_path)
            state_dict = torch.load(model_path)
            state_dict_quantized1 = torch.load(model_quantized1_path)
            state_dict_quantized2 = torch.load(model_quantized2_path)
            state_dict_quantized3 = torch.load(model_quantized3_path)
            model_list1 = []
            model_list2 = []
            model_list3 = []
            layer_list = []
            for layer_name in set([key.split('.')[0] for key in state_dict.keys()]):
                layer_list.append(layer_name)
                key_list = [key for key in state_dict.keys() if
                            key.startswith(layer_name)]
                spam1 = state_dict.copy()
                spam2 = state_dict.copy()
                spam3 = state_dict.copy()
                for key in key_list:
                    spam1[key] = state_dict_quantized1[key]
                    spam2[key] = state_dict_quantized2[key]
                    spam3[key] = state_dict_quantized3[key]
                model1 = MLP_BPSK(N, K).to(device)
                model2 = MLP_BPSK(N, K).to(device)
                model3 = MLP_BPSK(N, K).to(device)
                model1.load_state_dict(spam1)
                model2.load_state_dict(spam2)
                model3.load_state_dict(spam3)
                model_list1.append(model1)
                model_list2.append(model2)
                model_list3.append(model3)
            ser = np.zeros((3, len(layer_list),))
            dataset = DatasetBPSK(K, N, ensemble, H, (11.,),
                                  dataset_path=os.path.join(
                                      dataset_folder, 'test_bpsk_SNR_11.0.pt'
                                  ))
            dataloader = DataLoader(dataset, batch_size=batch_size,
                                    shuffle=True)
            translate_layers = {'fc1': 1, 'fc2': 2, 'fc3': 3, 'fc4': 4,
                                'fc5': 5, 'fc7': 6}
            for idx, model1 in enumerate(model_list1):
                layer_num = translate_layers[layer_list[idx]]
                ser[0, layer_num-1] = run_model_bpsk(dataloader, model1, device)
                ser[1, layer_num-1] = run_model_bpsk(dataloader,
                                                     model_list2[idx], device)
                ser[2, layer_num-1] = run_model_bpsk(dataloader,
                                                     model_list3[idx], device)
            full_model = MLP_BPSK(N, K).to(device)
            full_model.load_state_dict(state_dict)
            ser_full = run_model_bpsk(dataloader, full_model, device)
            fig = plt.figure(figsize=(width, height))
            ax = fig.add_subplot(111)
            ax.semilogy(range(1, len(model_list1)+1), ser[0, :],
                        label='$M_{\scriptsize\\textrm{max}}/L = 1$')
            ax.semilogy(range(1, len(model_list1)+1), ser[1, :],
                        label='$M_{\scriptsize\\textrm{max}}/L = 1.5$')
            ax.semilogy(range(1, len(model_list1)+1), ser[2, :],
                        label='$M_{\scriptsize\\textrm{max}}/L = 2$')
            ax.hlines(ser_full, 1, len(model_list1), colors='k', ls='--',
                      label='Full Precision')
            ax.grid()
            ax.legend()
            ax.set_ylim((ser_full*.9, ax.get_ylim()[1]))
            ax.set_xlabel('Quantized Layer')
            ax.set_ylabel('BER')
            fig.tight_layout()
            fig.savefig(os.path.join(figures_folder, 'ber3.eps'), format='eps',
                        bbox_inches='tight')
            fig.savefig(os.path.join(figures_folder, 'ber3.png'), format='png',
                        bbox_inches='tight')
        case 'train':
            train_batch_size = args.train_batchsize
            test_batch_size = args.test_batchsize
            train_ensemble = int(args.train_iter * args.train_batchsize)
            test_ensemble = int(args.test_iter * args.test_batchsize)
            learning_rate = args.learning_rate
            channel_path = os.path.join(dataset_folder, 'channel_matrix.npy')
            model_path = os.path.join(model_folder, 'model_weights.pth')
            if not os.path.isfile(channel_path):
                H = matrix_55_toeplitz(N, K)
                np.save(channel_path, H)
            else:
                H = np.load(channel_path)
            snr_values = np.arange(snr_limits[0]+1, snr_limits[1], 1)
            test_datasets = [
                DatasetQPSK(K, N, test_ensemble, H, (snr_values[idx],),
                           dataset_path=os.path.join(
                               dataset_folder,
                               f'test_SNR_{snr_values[idx]}.pt'
                           )) for idx in range(len(snr_values))
            ]
            model = MLP_QPSK(N, K).to(device)
            loss_fn = torch.nn.CrossEntropyLoss()
            # loss_fn = torch.nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=args.decay_lr_iter,
                gamma=args.decay_lr_factor
            )
            test_results = np.zeros((args.train_epochs, len(snr_values)))
            patience = 1
            for epoch in range(args.train_epochs):
                print(f'Epoch {epoch+1}\n------------------------------------')
                dataset_path = os.path.join(
                    dataset_folder, f'train_dataset_epoch_{epoch}.pt'
                )
                dataset = DatasetQPSK(K, N, train_ensemble, H, snr_limits,
                                     dataset_path=dataset_path)
                dataloader = DataLoader(dataset, train_batch_size, True)
                model.train()
                for it, (X, y) in enumerate(dataloader):
                    X, y = X.to(device), y.to(device)
                    y_hat = model(X.float())
                    loss = loss_fn(y_hat, y.float())
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    if it % 100 == 0:
                        print(f'Loss: {loss.item():.7f} [{it+1:d}/'
                              + f'{args.train_iter:d}]')
                    scheduler.step()
                test_result = np.zeros((len(snr_values),))
                model.eval()
                for idx in range(len(snr_values)):
                    test_dataloader = DataLoader(test_datasets[idx],
                                                test_batch_size, True)
                    ber = 0
                    with torch.no_grad():
                        for X, y in test_dataloader:
                            X, y = X.to(device), y.to(device)
                            y_hat = model(X.float()).argmax(1)
                            y_true = y.argmax(1)
                            ber += (y_hat != y_true).sum().item()/(2*K)
                    test_result[idx] = ber/test_ensemble
                test_results[epoch, :] = test_result
                checkpoint_path = os.path.join(checkpoints_folder,
                                            f'model_epoch_{epoch}.pth')
                torch.save(model.state_dict(), checkpoint_path)
                print(test_results[:epoch+1])
                if epoch > 0:
                    val_crit = np.mean(
                        (test_results[epoch-patience, :] - test_results[epoch, :])
                        / test_results[epoch-patience, :]
                    )
                    print(val_crit)
                    if val_crit <= 0:
                        patience += 1
                        if patience == 7:
                            print('No advance!')
                            state_dict = torch.load(
                                os.path.join(checkpoints_folder,
                                            f'model_epoch_{epoch-patience}.pth')
                            )
                            model.load_state_dict(state_dict)
                            break
                    else:
                        patience = 1
            torch.save(model.state_dict(), model_path)
        case 'run':
            # Experiment Quantization:
            ensemble = int(args.test_iter * args.test_batchsize)
            batch_size = args.test_batchsize
            channel_path = os.path.join(dataset_folder, 'channel_matrix.npy')
            model_path = os.path.join(model_folder, 'model_weights.pth')
            H = np.load(channel_path)
            state_dict = torch.load(model_path)
            model = MLP_QPSK(N, K).to(device)
            model.load_state_dict(state_dict)
            snr_values = np.arange(*snr_limits, 1)
            MN_ratio_values = np.array((1., 1.5, 2.))
            ser_zf = np.zeros((len(snr_values),))
            ser = np.zeros((len(MN_ratio_values)+1, len(snr_values)))
            print('Load Datasets ...')
            dataset_list = [
                DatasetQPSK(K, N, ensemble, H, (snr_values[idx],),
                           dataset_path=os.path.join(
                               dataset_folder,
                               f'run_SNR_{snr_values[idx]}.pt'
                           )) for idx in tqdm(range(len(snr_values)))
            ]
            print('Done!')
            for idx_MN, MN_ratio in np.ndenumerate(MN_ratio_values):
                idx0 = idx_MN[0]
                quantized_model_path = os.path.join(
                    model_folder, f'model_weights_SPT_ratio_{MN_ratio}.pth'
                )
                model_quatized = MLP_QPSK(N, K).to(device)
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
                        f'results_SPT_{MN_ratio}_SNR_{snr_dB}.npy'
                    )
                    results_zf_path = os.path.join(
                        results_folder, f'results_zf_SNR_{snr_dB}.npy'
                    )
                    results_fp_path = os.path.join(
                        results_folder, f'results_fp_SNR_{snr_dB}.npy'
                    )
                    idx1 = idx_snr[0]
                    dataloader = DataLoader(
                        dataset_list[idx1], batch_size=ensemble, shuffle=True
                    )
                    if idx0 == 0:
                        if os.path.isfile(results_zf_path):
                            ser_zf_idx = np.load(results_zf_path)
                        else:
                            print(f'Run Zero force model | SNR {snr_dB}')
                            ser_zf_idx = run_zf_qpsk(dataloader, H, device)
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
                        print(f'Run quantized model M_max/L {MN_ratio}'
                              + f'| SNR {snr_dB}')
                        ser_idx = run_model(dataloader, model_quatized, device)
                        np.save(results_path, ser_idx)
                    ser[idx0+1, idx1] = ser_idx

            fig = plt.figure(figsize=(width, height))
            ax = fig.add_subplot(111)
            ax.semilogy(snr_values, ser_zf, marker='s', label='ZF')
            ax.semilogy(snr_values, ser[0, :], marker='<',
                        label='Full Precision')
            marker_list = ['p', '*', 'X']
            for idx, MN_ratio in np.ndenumerate(MN_ratio_values):
                ax.semilogy(snr_values, ser[idx[0]+1, :],
                            marker=marker_list[idx[0]],
                            label='$M_{\scriptsize\\textrm{max}}/L$ = ' \
                                + f'{MN_ratio}')
            ax.legend(ncols=1)
            ax.grid()
            ax.set_xlabel('SNR, dB')
            ax.set_ylabel('BER')
            fig.tight_layout()

            fig.savefig(os.path.join(figures_folder, 'ber.eps'),
                        format='eps', bbox_inches='tight')
            fig.savefig(os.path.join(figures_folder, 'ber.png'),
                        format='png', bbox_inches='tight')
        case 'run2':
            # Compare with naive method
            wl = 6
            ensemble = int(args.test_iter * args.test_batchsize)
            batch_size = args.test_batchsize
            channel_path = os.path.join(dataset_folder, 'channel_matrix.npy')
            model_path = os.path.join(model_folder, 'model_weights.pth')
            H = np.load(channel_path)
            state_dict = torch.load(model_path)
            model_full = MLP_QPSK(N, K).to(device)
            model_naive = MLP_QPSK(N, K).to(device)
            model_mpgbp = MLP_QPSK(N, K).to(device)
            state_dict_naive, state_dict_mpgbp = \
                quantize_coefficients_naive_mpgbp(state_dict, wl, device)
            model_naive.load_state_dict(state_dict_naive)
            model_mpgbp.load_state_dict(state_dict_mpgbp)
            model_full.load_state_dict(state_dict)
            snr_values = np.arange(snr_limits[0]+1, snr_limits[1], 1)
            ser = np.zeros((4, len(snr_values)))
            print('Load Datasets ...')
            dataset_list = [
                DatasetQPSK(K, N, ensemble, H,
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
                print(f'Run full precision SNR {snr_dB}.')
                ser[2, idx0] = run_model(dataloader, model_full, device)
                print(f'Run zero forcing SNR {snr_dB}.')
                ser[3, idx0] = run_zf_qpsk(dataloader, H, device)
            
            fig = plt.figure(figsize=(width, height))
            ax = fig.add_subplot(111)
            ax.semilogy(snr_values, ser[3, :], marker='s', label='ZF')
            ax.semilogy(snr_values, ser[2, :], marker='<', label='Full Precision')
            ax.semilogy(snr_values, ser[0, :], marker='p', label='MPGBP')
            ax.semilogy(snr_values, ser[1, :], marker='*', label='Naïve (CSD)')
            ax.legend(ncols=1)
            ax.grid()
            ax.set_xlabel('SNR, dB')
            ax.set_ylabel('BER')
            fig.tight_layout()

            fig.savefig(os.path.join(figures_folder, 'ber2.eps'),
                        format='eps', bbox_inches='tight')
            fig.savefig(os.path.join(figures_folder, 'ber2.png'),
                        format='png', bbox_inches='tight')
        case 'run3':
            # Layer sensibility test
            snr_value = 11.
            ensemble = int(args.test_iter * args.test_batchsize)
            batch_size = args.test_batchsize
            channel_path = os.path.join(dataset_folder, 'channel_matrix.npy')
            model_path = os.path.join(model_folder, 'model_weights.pth')
            model_quantized_path = os.path.join(model_folder,
                                                'model_weights_SPT_ratio_1.0.pth')
            H = np.load(channel_path)
            state_dict = torch.load(model_path)
            state_dict_quantized = torch.load(model_quantized_path)
            model_list = []
            layer_list = []
            for layer_name in set([key.split('.')[0] for key in state_dict.keys()]):
                layer_list.append(layer_name)
                key_list = [key for key in state_dict.keys() if
                            key.startswith(layer_name)]
                spam = state_dict.copy()
                for key in key_list:
                    spam[key] = state_dict_quantized[key]
                model = MLP_QPSK(N, K)
                model.load_state_dict(spam)
                model.to(device)
                model_list.append(model)
            ber = np.zeros((len(layer_list),))
            dataset = DatasetQPSK(K, N, ensemble, H, (snr_value,),
                                  dataset_path=os.path.join(
                                      dataset_folder, f'test_SNR_{snr_value}.pt'
                                  ))
            dataloader = DataLoader(dataset, batch_size=batch_size,
                                    shuffle=True)
            translate_layers = {'fc1': 1, 'fc2': 2, 'fc3': 3, 'fc4': 4,
                                'fc5': 5, 'fc6': 6}
            for idx, model in enumerate(model_list):
                layer_num = translate_layers[layer_list[idx]]
                ber[layer_num-1] = run_model(dataloader, model,device)
            full_model = MLP_QPSK(N, K).to(device)
            full_model.load_state_dict(state_dict)
            ber_full = run_model(dataloader, full_model, device)
            ber_zf = run_zf_qpsk(dataloader, H, device)
            fig = plt.figure(figsize=(width, height))
            ax = fig.add_subplot(111)
            ax.semilogy(range(1, len(layer_list)+1), ber,
                        label='$M_{\scriptsize\\textrm{max}}/L = 1$')
            ax.hlines(ber_full, 1, len(layer_list), colors='k', ls='--',
                      label='Full Precision')
            ax.hlines(ber_zf, 1, len(layer_list), colors='r', ls='--',
                      label='ZF')
            ax.grid()
            ax.legend()
            ax.set_ylim((ber_full*.95, ber_zf*1.05))
            ax.set_xlabel('Quantized Layer')
            ax.set_ylabel('BER')
            fig.tight_layout()
            fig_eps_path = os.path.join(figures_folder, 'ber3.eps')
            fig_png_path = os.path.join(figures_folder, 'ber3.png')
            fig.savefig(fig_eps_path, format='eps', bbox_inches='tight')
            fig.savefig(fig_png_path, format='png', bbox_inches='tight')


SER = [0.0147166,0.0062166,0.00218,5.0333e-04,6.333e-05,1.4e-05]
SER_ZF=[0.0330367,0.0203466,0.0105633,0.004863,0.00176,5.633e-04]