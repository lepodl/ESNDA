# -*- coding: utf-8 -*- 
# @Time : 2023/5/3 13:42 
# @Author : lepold
# @File : grid_sim.py
import os
from multiprocessing.pool import Pool

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from mpi4py import MPI
from mpl_toolkits.axes_grid1 import make_axes_locatable

from Data.Lorenz import Lorenz
from model.Esn import Esn

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def initialize_data(train_time=80, test_time=10, noise=None):
    dt = 0.02
    lyapunov_exp = 0.91
    train_length = int(train_time / lyapunov_exp / dt)
    test_length = int(test_time / lyapunov_exp / dt)
    np.random.seed(rank)
    coords = np.random.rand(3) * 10
    model = Lorenz(10., 28., 8 / 3, dt, *coords)
    states = model.propagate(train_length + test_length + 2000, 1000, noise=noise)
    train_data = states[:, :train_length]
    test_data = states[:, train_length:train_length + test_length]
    return train_data, test_data


def prediction_power(x, y, dt=0.02, unit=0.91):
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


train_data, test_data = initialize_data(train_time=100, test_time=20)
test_inputs, test_targets = test_data[:, :], test_data[:, 1:]
test_length = test_targets.shape[1]


class Esn_new(Esn):
    def __init__(self, n_inputs, n_outputs, first_order, second_order, **kwargs):
        super().__init__(n_inputs, n_outputs, **kwargs)
        self.first_order = first_order
        self.second_order = second_order
        self.W_square = np.matmul(self.W, self.W)

    def func(self, state, input_pattern):
        temp = state * (1 - self.leaky_rate) + self.leaky_rate * \
               np.tanh(
                   self.first_order * np.dot(self.W, state) + self.second_order * np.dot(self.W_square, state) + np.dot(
                       self.W_in, input_pattern) + self.W_bias)
        return temp

    def func_forward(self, state):
        """
        automatically forward using only the current state
        """
        input_pattern = np.dot(self.W_out, state)
        temp = state * (1 - self.leaky_rate) + self.leaky_rate * \
               np.tanh(
                   self.first_order * np.dot(self.W, state) + self.second_order * np.dot(self.W_square, state) + np.dot(
                       self.W_in, input_pattern) + self.W_bias)
        return temp


def run(model):
    global train_data
    global test_length
    global test_targets
    model.fit(train_data)
    prediction = model.auto_evolve(test_inputs[:, 0], n_iteration=test_length)
    out = prediction_power(test_targets, prediction)
    # print(f"done | power {out:.2f}")
    return out


first_orders = np.linspace(0.2, 1.8, 34, endpoint=True)
second_orders = np.linspace(-0.8, 0.8, 34, endpoint=True)
models = []
for i in range(len(first_orders)):
    for j in range(len(second_orders)):
        models.append(Esn_new(n_inputs=3,
                              n_outputs=3,
                              first_order=first_orders[i],
                              second_order=second_orders[j],
                              n_reservoir=100,
                              leaky_rate=0.9,
                              spectral_radius=1.,
                              random_state=rank,
                              sparsity=0.4,
                              silent=True,
                              ridge_param=1e-5,
                              washout=400,
                              readout_training='ridge_regression'
                              ))
N = len(models)
test_inputs, test_targets = test_data[:, :], test_data[:, 1:]
test_length = test_targets.shape[1]

total_experiments = 1
assert size == total_experiments + 1
for i in range(rank, total_experiments, size):
    with Pool(processes=os.cpu_count()) as p:
        powers = p.map(run, models)
    print("done in rank", i)
    comm.send(powers, dest=size - 1)
if rank == size - 1:
    total_powers = []
    for id in range(size - 1):
        powers = comm.recv(source=id)
        assert len(powers) == N
        total_powers.append(powers)
    total_powers = np.array(total_powers).reshape((total_experiments, len(first_orders), len(second_orders)))
    powers_mean = np.mean(total_powers, axis=0)
    max_coord = np.argmax(powers_mean)
    max_coords = np.argsort(powers_mean.flatten())[-5:]
    min_coords = np.argsort(powers_mean.flatten())[:5]
    powers_std = np.std(total_powers, axis=0)
    x = len(first_orders)
    y = len(second_orders)
    coord0, coord1 = max_coord // x, max_coord % x

    fig = plt.figure(figsize=(8, 8))
    axes = dict()
    gs = gridspec.GridSpec(2, 2)
    gs.update(left=0.1, right=0.94, top=0.93, bottom=0.1, wspace=0.2, hspace=0.2)
    axes['A'] = plt.subplot(gs[0, 0])
    axes['B'] = plt.subplot(gs[0, 1])
    axes['C'] = plt.subplot(gs[1, 0])
    axes['D'] = plt.subplot(gs[1, 1])

    im = axes['A'].imshow(powers_mean, cmap='RdBu_r', interpolation='gaussian')
    divider = make_axes_locatable(axes['A'])
    cax = divider.new_vertical(size='10%', pad=0.2)
    fig.add_axes(cax)
    cb = fig.colorbar(im, cax=cax, orientation='horizontal')  # shrink=0.6,
    cb.ax.set_title("prediction_mean")

    im = axes['B'].imshow(powers_std, cmap='RdBu_r', interpolation='gaussian')
    divider = make_axes_locatable(axes['B'])
    cax = divider.new_vertical(size='10%', pad=0.2)
    fig.add_axes(cax)
    cb = fig.colorbar(im, cax=cax, orientation='horizontal')  # shrink=0.6,
    cb.ax.set_title("prediction_std")

    # plot spectral
    n_reservoir = 100
    sparsity = 0.4
    spectral_radius = 1.
    np.random.rand(0)
    W = (np.random.rand(n_reservoir, n_reservoir) - 0.5) * 2  # [-1, 1]
    W[np.random.rand(*W.shape) < sparsity] = 0
    radius = np.max(np.abs(np.linalg.eigvals(W)))
    W = W * (spectral_radius / radius)

    radius_new = []
    for a in first_orders:
        for b in second_orders:
            W_new = a * W + b * np.matmul(W, W)
            radius = np.max(np.abs(np.linalg.eigvals(W_new)))
            radius_new.append(radius)
    radius_new = np.array(radius_new).reshape((len(first_orders), len(second_orders)))

    im = axes['C'].imshow(radius_new, cmap='RdBu_r', interpolation='gaussian')
    divider = make_axes_locatable(axes["C"])
    cax = divider.new_vertical(size='10%', pad=0.2)
    fig.add_axes(cax)
    cb = fig.colorbar(im, cax=cax, orientation='horizontal')  # shrink=0.6,
    cb.ax.set_title("radius_new")

    W_max_prediction = first_orders[coord0] * W + second_orders[coord1] * np.matmul(W, W)
    data = np.linalg.eigvals(W_max_prediction)
    try:
        axes['D'].scatter(np.real(data), np.imag(data), marker="o")
        axes["D"].spines["top"].set_visible(False)
        axes["D"].spines["right"].set_visible(False)
        axes["D"].spines["bottom"].set_position(('data', 0))
        axes["D"].spines["left"].set_position(('data', 0))
        plt.text(0.8, 0.7, f"alpha: {first_orders[coord0]:.2f}\nbeta: {second_orders[coord1]:.2f}",
                 fontdict={'fontsize': 11, 'horizontalalignment': 'left', 'verticalalignment':
                     'bottom'}, transform=axes["D"].transAxes)
    except:
        pass

    for name in ["A", "B", "C"]:
        yticks = np.linspace(0, x - 1, 4, endpoint=True, dtype=np.int8)
        axes[name].set_yticks(yticks)
        axes[name].set_yticklabels([f'{data:.3f}' for data in first_orders[yticks]], rotation=60)
        xticks = np.linspace(0, y - 1, 4, endpoint=True, dtype=np.int8)
        axes[name].invert_yaxis()
        axes[name].set_xticks(xticks)
        axes[name].set_xticklabels([f'{data:.3f}' for data in second_orders[xticks]], )
        axes[name].set_ylabel(r"$\alpha$")
        axes[name].set_xlabel(r"$\beta$")

    np.savez("powers_single_trial.npz", total_powers=total_powers, W=W)
    fig.savefig("./powers_single_trial.png", dpi=100)
    plt.close(fig)

    fig = plt.figure(figsize=(10, 5))
    axes = dict()
    gs = gridspec.GridSpec(2, 5)
    gs.update(left=0.1, right=0.94, top=0.93, bottom=0.1, wspace=0.2, hspace=0.3)
    axes["A0"] = plt.subplot(gs[0, 0])
    axes["A1"] = plt.subplot(gs[0, 1])
    axes['A2'] = plt.subplot(gs[0, 2])
    axes['A3'] = plt.subplot(gs[0, 3])
    axes['A4'] = plt.subplot(gs[0, 4])
    axes["B0"] = plt.subplot(gs[1, 0])
    axes["B1"] = plt.subplot(gs[1, 1])
    axes['B2'] = plt.subplot(gs[1, 2])
    axes['B3'] = plt.subplot(gs[1, 3])
    axes['B4'] = plt.subplot(gs[1, 4])

    for i, item in enumerate(max_coords):
        coord0, coord1 = item // x, item % x
        W_max_prediction = first_orders[coord0] * W + second_orders[coord1] * np.matmul(W, W)
        data = np.linalg.eigvals(W_max_prediction)
        axes["A%d"%i].scatter(np.real(data), np.imag(data), marker="o")
        axes["A%d"%i].spines["top"].set_visible(False)
        axes["A%d"%i].spines["right"].set_visible(False)
        axes["A%d"%i].spines["bottom"].set_position(('data', 0))
        axes["A%d"%i].spines["left"].set_position(('data', 0))
        plt.text(0.8, 0.7, f"alpha: {first_orders[coord0]:.2f}\nbeta: {second_orders[coord1]:.2f}",
                 fontdict={'fontsize': 11, 'horizontalalignment': 'left', 'verticalalignment':
                     'bottom'}, transform=axes["A%d"%i].transAxes)
    for i, item in enumerate(min_coords):
        coord0, coord1 = item // x, item % x
        W_max_prediction = first_orders[coord0] * W + second_orders[coord1] * np.matmul(W, W)
        data = np.linalg.eigvals(W_max_prediction)
        axes["B%d"%i].scatter(np.real(data), np.imag(data), marker="o")
        axes["B%d"%i].spines["top"].set_visible(False)
        axes["B%d"%i].spines["right"].set_visible(False)
        axes["B%d"%i].spines["bottom"].set_position(('data', 0))
        axes["B%d"%i].spines["left"].set_position(('data', 0))
        plt.text(0.8, 0.7, f"alpha: {first_orders[coord0]:.2f}\nbeta: {second_orders[coord1]:.2f}",
                 fontdict={'fontsize': 11, 'horizontalalignment': 'left', 'verticalalignment':
                     'bottom'}, transform=axes["B%d"%i].transAxes)
    for name in ["A0", "B0"]:
        plt.text(-0.08, 1.05, name.split("0")[0],
                 fontdict={'fontsize': 13, 'horizontalalignment': 'left', 'verticalalignment':
                     'bottom', 'weight': 'bold'}, transform=axes[name].transAxes)
    fig.savefig("./eigenvalues_single_trail.png", dpi=100)
    print("ALL DONE!")
