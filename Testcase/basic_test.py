# -*- coding: utf-8 -*- 
# @Time : 2022/10/18 21:05 
# @Author : lepold
# @File : basic_test.py

import unittest

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Data.Lorenz import Lorenz
from model.Esn import Esn


class TestCase(unittest.TestCase):
    """
    basic testcase for ESNDA and ESN
    """

    @staticmethod
    def rmse(outputs, targets):
        if outputs.size != targets.size:
            raise ValueError(u"Ouputs and targets ndarray don have the same number of elements")
        return np.sqrt(np.mean(np.sum((targets - outputs) ** 2, axis=0)))

    @staticmethod
    def distance(x, y):
        loss = np.linalg.norm(x - y, ord=2, axis=0)
        loss = loss / np.linalg.norm(x, ord=2, axis=0)
        return loss

    @staticmethod
    def prediction_power(x, y, dt=0.01, unit=0.91):
        loss = np.linalg.norm(x - y, ord=2, axis=0) / np.linalg.norm(x, ord=2, axis=0)
        if (loss[:20] < 0.1).all():
            try:
                out = np.where(loss > 0.22)[0][0]
            except:
                out = x.shape[1]
        else:
            out = 0
        out = out * dt * unit
        return out

    @staticmethod
    def initialize_data(train_time=80, test_time=10, noise=None):
        dt = 0.02
        lyapunov_exp = 0.91
        train_length = int(train_time / lyapunov_exp / dt)
        test_length = int(test_time / lyapunov_exp / dt)
        np.random.seed(10000)
        coords = np.random.rand(3) * 10
        model = Lorenz(10., 28., 8 / 3, dt, *coords)
        states = model.propagate(train_length + test_length + 2000, 1000, noise=noise)
        train_data = states[:, :train_length]
        test_data = states[:, train_length:train_length + test_length]
        return train_data, test_data

    def single_plot(self, target, prediction):
        lyapunov_exp = 0.91
        dt = 0.02
        test_length = target.shape[1]
        max_time_unit = int(test_length * dt * lyapunov_exp) + 1
        print("max_time_unit", max_time_unit)
        time_ticks = [l / lyapunov_exp / dt for l in range(max_time_unit)]
        pred_power = self.prediction_power(target, prediction)
        loss = self.distance(target, prediction)
        fig, ax = plt.subplots(4, 1, figsize=(5, 4), dpi=200)
        ax = ax.flatten()
        coords = ["x coordinate", "y coordinate", "z coordinate"]

        for i in range(3):
            ax[i].plot(range(test_length), target[i, :], 'k', lw=0.5, label="target system")
            ax[i].plot(range(test_length), prediction[i, :], 'r', lw=0.5, label="free running RFDA")
            ax[i].set_xticks([])
            ax[i].text(0.15, 0.8, coords[i], fontsize=8, ha='center', va='center', color='b',
                       transform=ax[i].transAxes)
        ax[3].plot(range(test_length), loss, 'k', lw=0.5, label="loss")
        ax[3].legend(loc=(0.05, 0.7), fontsize=8)
        ax[0].legend(loc=(0.75, 1.1), fontsize='x-small')
        ax[3].set_xticks(time_ticks)
        ax[3].set_xticklabels(np.arange(max_time_unit))
        ax[3].set_xlabel('$\lambda_{max}t$')
        ax[0].set_title(f"Predict Power {pred_power:.2f}")
        return fig

    def _test_basic_esn_prediction(self):
        train_data, test_data = self.initialize_data(100, 20)
        esn = Esn(n_inputs=3,
                  n_outputs=3,
                  n_reservoir=100,
                  leaky_rate=0.9,
                  spectral_radius=1.,
                  random_state=1,
                  sparsity=0.4,
                  silent=False,
                  ridge_param=1e-5,
                  washout=400,
                  readout_training='ridge_regression')

        test_inputs, test_targets = test_data[:, :], test_data[:, 1:]
        test_length = test_targets.shape[1]

        # eta = 1.
        # noise_train_data = train_data + np.random.multivariate_normal(np.zeros(3), np.eye(3) * eta, size=train_data.shape[1]).T
        esn.fit(train_data)
        prediction = esn.auto_evolve(test_inputs[:, 0], n_iteration=test_length)
        fig = self.single_plot(test_targets, prediction)
        # fig.savefig(os.path.join("../Result", 'Esn_prediction.png'))
        plt.show()

    def test_esnda_prediction(self):
        train_data, test_data = self.initialize_data(100, 20)
        esn = Esn(n_inputs=3,
                  n_outputs=3,
                  n_reservoir=92,
                  leaky_rate=0.987,
                  spectral_radius=0.148,
                  random_state=100,
                  sparsity=0.4,
                  silent=False,
                  ridge_param=1e-3,
                  washout=400)

        test_inputs, test_targets = test_data[:, :], test_data[:, 1:]
        test_length = test_targets.shape[1]

        eta = 1.
        dt = 0.01
        noise_train_data = train_data + np.random.multivariate_normal(np.zeros(3), np.eye(3) * eta,
                                                                      size=train_data.shape[1]).T
        esn.fit_da(noise_train_data, ensembles=500, observation_noise=eta, model_noise=None, initial_w_dist=1000.)
        prediction = esn.auto_evolve(test_inputs[:, 0], n_iteration=test_length)
        pred_power = self.prediction_power(test_targets, prediction)
        loss = self.distance(test_targets, prediction)

        print(f'noise level {eta:.2f}')
        # print(f'maxium prediction time {pred_power:.2f}')

        lyapunov_exp = 0.91
        max_time_unit = int(test_length * dt * lyapunov_exp) + 1
        time_ticks = [l / lyapunov_exp / dt for l in range(max_time_unit)]
        fig, ax = plt.subplots(4, 2, figsize=(8, 5))
        coords = ["x coordinate", "y coordinate", "z coordinate"]

        for i in range(3):
            ax[i, 0].plot(range(test_length), test_targets[i, :], 'k', lw=0.5, label="target system")
            ax[i, 0].plot(range(test_length), prediction[i, :], 'r', lw=0.5, label="free running RFDA")
            ax[i, 0].set_xticks([])
            ax[i, 0].text(0.1, 0.8, coords[i], fontsize=10, ha='center', va='center', color='b',
                          transform=ax[i, 0].transAxes)
        ax[3, 0].plot(range(test_length), loss, 'k', lw=0.5, label="loss")
        ax[3, 0].legend(loc=(0.05, 0.7), fontsize=8)
        ax[0, 0].legend(loc=(0.81, 1.1), fontsize='x-small')
        ax[3, 0].set_xticks(time_ticks)
        ax[3, 0].set_xticklabels(np.arange(max_time_unit))
        ax[3, 0].set_xlabel('$ \lambda_{max}t $')
        ax[0, 0].set_title(f"ESN with DA|Power {pred_power:.2f}", fontsize=10)

        esn = Esn(n_inputs=3,
                  n_outputs=3,
                  n_reservoir=92,
                  leaky_rate=0.987,
                  spectral_radius=0.148,
                  random_state=100,
                  sparsity=0.4,
                  silent=False,
                  ridge_param=1e-3,
                  washout=400)
        _ = esn.fit(noise_train_data)
        prediction = esn.auto_evolve(test_inputs[:, 0], n_iteration=test_length)
        pred_power = self.prediction_power(test_targets, prediction)
        loss = self.distance(test_targets, prediction)

        for i in range(3):
            ax[i, 1].plot(range(test_length), test_targets[i, :], 'k', lw=0.5, label="target system")
            ax[i, 1].plot(range(test_length), prediction[i, :], 'r', lw=0.5, label="free running RFDA")
            ax[i, 1].set_xticks([])
            ax[i, 1].text(0.1, 0.8, coords[i], fontsize=10, ha='center', va='center', color='b',
                          transform=ax[i, 1].transAxes)
        ax[3, 1].plot(range(test_length), loss, 'k', lw=0.5, label="loss")
        ax[3, 1].legend(loc=(0.05, 0.7), fontsize=8)
        ax[0, 1].legend(loc=(0.81, 1.1), fontsize='x-small')
        ax[3, 1].set_xticks(time_ticks)
        ax[3, 1].set_xticklabels(np.arange(max_time_unit))
        ax[3, 1].set_xlabel('$ \lambda_{max}t $')
        ax[0, 1].set_title(f"ESN with LR| Power {pred_power:.2f}", fontsize=10)
        # fig.savefig(os.path.join('results/DAvsLR.png'), dpi=100)
        plt.show()

    def _test_plot_noise_compare(self):
        # After run compare_analysis.py anc compose.py
        df_lr = pd.read_csv("../Result/noise_dependence_esnlr_new.csv")
        df_da = pd.read_csv("../Result/noise_dependence_esnda_new.csv")
        res_lr = df_lr.values[:, 1:]
        res_da = df_da.values[:, 1:]
        print(res_lr.shape)

        etas = np.linspace(-10, 10, 100)
        fig = plt.figure(figsize=(10, 4))
        ax = fig.add_subplot(1, 2, 1)
        ax.plot(etas, res_lr.mean(axis=0), label="ESN_LR")
        ax.plot(etas, res_da.mean(axis=0), label="ESN_DA")
        ax.set_xlabel(r"$ln \,\eta$")
        ax.set_ylabel(r"$\lambda \, t$")
        ax.legend(loc="best")
        ax = fig.add_subplot(1, 2, 2)
        ax.errorbar(etas[2::4], res_lr.mean(axis=0)[2::4], yerr=res_lr.std(axis=0)[2::4], fmt='bo-', elinewidth=1.5,
                    label="ESN_LR")
        ax.errorbar(etas[2::4], res_da.mean(axis=0)[2::4], yerr=res_da.std(axis=0)[2::4], fmt='ro-', elinewidth=1.5,
                    label="ESN_DA")
        ax.set_xlabel(r"$ln \,\eta$")
        ax.set_ylabel(r"$\lambda \, t$")
        ax.legend(loc="best")
        plt.tight_layout()
        # fig.savefig(os.path.join("../Result/noise_analysis.png"))
        plt.show()

    def _test_noise2d_analysis(self):
        file = np.load("../Result/noise2d_lr.npz")
        power = file['res']
        etas = file['etas']
        noise = file["noise"]
        x, y = power.shape[0], power.shape[1]
        fig = plt.figure(figsize=(10, 4))

        ax = fig.add_subplot(1, 2, 1)
        data1 = ax.imshow(power.mean(axis=-1))
        plt.colorbar(data1, ax=ax, shrink=0.8)
        xticks = np.linspace(0, x, 4, endpoint=False, dtype=np.int)
        ax.set_yticks(xticks)
        ax.set_yticklabels([f'{data:.2f}' for data in noise[xticks]], rotation=60)
        yticks = np.linspace(0, y, 4, endpoint=False, dtype=np.int)
        ax.invert_yaxis()
        ax.set_xticks(yticks)
        ax.set_xticklabels([f'{data:.2f}' for data in etas[xticks]], )
        ax.set_xlabel('etas')
        ax.set_ylabel("model_noise")
        ax.set_title("lr_power_mean")

        ax = fig.add_subplot(1, 2, 2)
        data1 = ax.imshow(power.std(axis=-1))
        plt.colorbar(data1, ax=ax, shrink=0.8)
        xticks = np.linspace(0, x, 4, endpoint=False, dtype=np.int)
        ax.set_yticks(xticks)
        ax.set_yticklabels([f'{data:.2f}' for data in noise[xticks]], rotation=60)
        yticks = np.linspace(0, y, 4, endpoint=False, dtype=np.int)
        ax.invert_yaxis()
        ax.set_xticks(yticks)
        ax.set_xticklabels([f'{data:.2f}' for data in etas[xticks]], )
        ax.set_xlabel('etas')
        ax.set_ylabel("model_noise")
        ax.set_title("lr_power_std")
        fig.savefig("../Result/noise_2dim_of_lr.png")
        plt.show()

    def _test_par_analysis(self):
        file = pd.read_csv('../Result/da_noise_to_eta_1.csv')
        data = file.values[:, 1:]
        da_noise = np.linspace(0, 5, 100)
        plt.plot(da_noise, data.mean(axis=0))
        plt.show()


if __name__ == '__main__':
    unittest.main()
