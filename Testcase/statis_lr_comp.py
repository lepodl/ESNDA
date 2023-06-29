# -*- coding: utf-8 -*- 
# @Time : 2023/6/4 11:20 
# @Author : lepold
# @File : statis_compare.py
import os
from multiprocessing.pool import Pool

import matplotlib.pyplot as plt
import numpy as np
from mpi4py import MPI

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
    # np.random.seed(100)
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
    if (loss[:20] < 1).all():
        try:
            out = np.where(loss > 1)[0][0]
        except:
            out = x.shape[1]
    else:
        out = 0
    out = out * dt * unit
    return out


def run(model):
    ind = rank // 10
    idx = rank % 10
    dyn_noise_here = dyn_noise[ind]
    train_data, test_data = initialize_data(train_time=30, test_time=10, noise=dyn_noise_here)
    test_inputs, test_targets = test_data[:, :], test_data[:, 1:]
    test_length = test_targets.shape[1]
    noise_train_data = train_data + np.random.multivariate_normal(np.zeros(3), np.eye(3) * obs_noise[idx],
                                                                  size=train_data.shape[1]).T

    model.fit(noise_train_data)
    prediction = model.auto_evolve(test_inputs[:, 0], n_iteration=test_length)
    out = prediction_power(test_targets, prediction)
    # print(f"done | power {out:.2f}")
    return out


trials = 200
models = []
for i in range(trials):
    esn = Esn(n_inputs=3,
              n_outputs=3,
              n_reservoir=100,
              leaky_rate=0.5,
              spectral_radius=0.2,
              random_state=100,
              sparsity=0.4,
              silent=True,
              ridge_param=1e-5,
              washout=400)
    models.append(esn)

# apply 31 nodes, the last node is host node, setting as
# obs_noise: linspace(0., 10, 10) with inner noise 0, 0.2, 1.
dyn_noise = [None, 1., 10.]
obs_noise = np.logspace(-1, 2, 10)
if rank < 30:
    with Pool(processes=os.cpu_count()) as p:
        powers = p.map(run, models)
    print("done in rank", rank)
    comm.send(powers, dest=size - 1)

if rank == size - 1:
    total_powers = []
    for id in range(size - 1):
        powers = comm.recv(source=id)
        assert len(powers) == trials
        total_powers.append(powers)
    total_powers = np.array(total_powers).reshape((3, 10, trials))
    fig = plt.figure(figsize=(5, 5))
    ax = fig.gca()
    names = ["dyn_noise 0.0", "dyn_noise 1", "dyn_noise 10",]
    for i in range(total_powers.shape[0]):
        powers = total_powers[i,]
        powers_mean = powers.mean(-1)
        powers_std = powers.std(-1)
        print("std", powers_std)
        ax.errorbar(np.linspace(-2, 2, 10, endpoint=True), powers_mean, yerr=powers_std, fmt='o:', ecolor='hotpink', label=f"{names[i]}")
        ax.set_xlabel("$log \eta$")
        ax.set_ylabel("Power")
        ax.legend(loc="best")
        fig.tight_layout()
    np.savez("./results/statis_lr.npz", dyn_noise=dyn_noise, obs_noise=obs_noise, powers=total_powers)
    fig.savefig("./results/statis_lr.png", dpi=100)
    print("Done")

