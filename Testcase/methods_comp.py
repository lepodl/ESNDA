# -*- coding: utf-8 -*- 
# @Time : 2023/6/4 15:38 
# @Author : lepold
# @File : methods_comp.py

import os
from multiprocessing.pool import Pool

import matplotlib.pyplot as plt
import numpy as np
from mpi4py import MPI
from matplotlib import gridspec

from Data.Lorenz import Lorenz
from model.Esn import Esn

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def initialize_data(train_time=80, test_time=10, noise=None, random_seed=None):
    dt = 0.02
    lyapunov_exp = 0.91
    train_length = int(train_time / lyapunov_exp / dt)
    test_length = int(test_time / lyapunov_exp / dt)
    if random_seed is not None:
        np.random.seed(random_seed)
    coords = np.random.rand(3) * 10
    model = Lorenz(10., 28., 8 / 3, dt, *coords)
    states = model.propagate(train_length, 1000, noise=noise)
    assert states.shape[1] == train_length
    train_data = states
    coords_new = states[:, -1]
    model = Lorenz(10., 28., 8 / 3, dt, *coords_new)
    test_data = model.propagate(test_length, 1, noise=None)
    return train_data, test_data


def prediction_power(x, y, dt=0.02, unit=0.91):
    loss = np.linalg.norm(x - y, ord=2, axis=0) / np.linalg.norm(x, ord=2, axis=0)
    if (loss[:30] < 0.5).all():
        try:
            out = np.where(loss > 0.5)[0][0]
        except:
            out = x.shape[1]
    else:
        out = 0
    out = out * dt * unit
    return out


def run(num_trial):
    model = Esn(n_inputs=3,
              n_outputs=3,
              n_reservoir=100,
              leaky_rate=0.9,
              spectral_radius=0.2,
              random_state=num_trial,
              sparsity=0.4,
              silent=True,
              ridge_param=1e-3,
              washout=400)
    ind = rank // 10
    idx = rank % 10
    dyn_noise_here = dyn_noise[ind]
    train_data, test_data = initialize_data(train_time=40, test_time=10, noise=dyn_noise_here, random_seed=num_trial)
    test_inputs, test_targets = test_data[:, :], test_data[:, 1:]
    test_length = test_targets.shape[1]
    noise_train_data = train_data + np.random.multivariate_normal(np.zeros(3), np.eye(3) * obs_noise[idx],
                                                                  size=train_data.shape[1]).T

    model.fit(noise_train_data)
    prediction = model.auto_evolve(test_inputs[:, 0], n_iteration=test_length)
    out = prediction_power(test_targets, prediction)

    model.clear_mode()
    model.reset_wout()  # Annotate this in reproducing the Testcase/results/methods_comp_final.png
    model.fit_da(noise_train_data, ensembles=500, observation_noise=obs_noise[idx], model_noise=None, initial_w_dist=10000.)
    prediction = model.auto_evolve(test_inputs[:, 0], n_iteration=test_length)
    pred_power = prediction_power(test_targets, prediction)
    print(f"done | lr vs da|power {out:.2f}")
    return [out, pred_power]


trials = 200
# apply 31 nodes, the last node is host node, setting as
# obs_noise: linspace(0., 10, 10) with inner noise 0, 0.2, 1.
dyn_noise = [None, 5.]
obs_noise = np.logspace(-2, 1, 10)
if rank < 20:
    with Pool(processes=os.cpu_count()) as p:
        powers = p.map(run, range(trials))
    print("done in rank", rank)
    comm.send(powers, dest=size - 1)

if rank == size - 1:
    total_powers = []
    for id in range(size - 1):
        powers = comm.recv(source=id)
        assert len(powers) == trials
        total_powers.append(powers)
    total_powers = np.array(total_powers).reshape((2, 10, trials, 2))
    fig = plt.figure(figsize=(8, 4))
    axes = dict()
    gs = gridspec.GridSpec(1, 2)
    gs.update(left=0.1, right=0.94, top=0.93, bottom=0.15, wspace=0.2, hspace=0.2)
    axes['A'] = plt.subplot(gs[0, 0])
    axes['B'] = plt.subplot(gs[0, 1])
    names = ["LR", "DA"]
    for i in range(total_powers.shape[-1]):
        powers = total_powers[0,]
        powers_mean = powers[:, :, i].mean(1)
        powers_std = powers[:, :, i].std(1)
        axes['A'].errorbar(np.linspace(-2, 1, 10, endpoint=True), powers_mean, yerr=powers_std, fmt='o:', ecolor='hotpink', capsize=3., capthick=1., errorevery=2, label=names[i])
    axes['A'].set_xlabel("$log \eta$")
    axes['A'].set_ylabel("Power")
    axes['A'].legend(loc="best")
    for i in range(total_powers.shape[-1]):
        powers = total_powers[1,]
        powers_mean = powers[:, :, i].mean(1)
        powers_std = powers[:, :, i].std(1)
        axes['B'].errorbar(np.linspace(-2, 1, 10, endpoint=True), powers_mean, yerr=powers_std, fmt='o:', ecolor='hotpink', capsize=3., capthick=1., errorevery=2, label=names[i])
    axes['B'].set_xlabel("$log \eta$")
    axes['B'].set_ylabel("Power")
    axes['B'].legend(loc="best")
    axes['A'].text(0.7, 0.8, "dyn noise:0", fontsize=10, ha='center', va='center', color='b',
                  transform=axes['A'].transAxes)
    axes['B'].text(0.7, 0.8, "dyn noise:2", fontsize=10, ha='center', va='center', color='b',
                  transform=axes['B'].transAxes)
    np.savez("./results/methods_comp.npz", dyn_noise=dyn_noise, obs_noise=obs_noise, powers=total_powers)
    fig.savefig("./results/methods_comp.png", dpi=100)
    print("Done")

