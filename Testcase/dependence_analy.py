# -*- coding: utf-8 -*- 
# @Time : 2023/7/1 11:17 
# @Author : lepold
# @File : dependence_analy.py


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


# experiment 1: only with observation noise, consider this simple case and analyse the dependence with \Gamma.
def run(num_trial):
    model = Esn(n_inputs=3,
              n_outputs=3,
              n_reservoir=100,
              leaky_rate=0.9,
              spectral_radius=0.2,
              random_state=num_trial,
              sparsity=0.4,
              silent=True,
              ridge_param=1e-5,
              washout=400)
    idx = rank
    train_data, test_data = initialize_data(train_time=40, test_time=10, noise=None, random_seed=num_trial)
    test_inputs, test_targets = test_data[:, :], test_data[:, 1:]
    test_length = test_targets.shape[1]
    noise_train_data = train_data + np.random.multivariate_normal(np.zeros(3), np.eye(3) * specified_obs_noise,
                                                                  size=train_data.shape[1]).T

    model.fit(noise_train_data)
    prediction = model.auto_evolve(test_inputs[:, 0], n_iteration=test_length)
    out = prediction_power(test_targets, prediction)

    model.clear_mode()
    model.fit_da(noise_train_data, ensembles=500, observation_noise=specified_obs_noise, model_noise=None, initial_w_dist=Gamma_range[idx])
    prediction = model.auto_evolve(test_inputs[:, 0], n_iteration=test_length)
    pred_power_0 = prediction_power(test_targets, prediction)

    model.clear_mode()
    model.reset_wout()
    model.fit_da(noise_train_data, ensembles=500, observation_noise=specified_obs_noise, model_noise=None,
                 initial_w_dist=Gamma_range[idx])
    prediction = model.auto_evolve(test_inputs[:, 0], n_iteration=test_length)
    pred_power_1 = prediction_power(test_targets, prediction)

    # [lr, da_from_0, da_from_lr]
    return [out, pred_power_0, pred_power_1]

# experiment 2: only with observation noise, consider this simple case and analyse the dependence with \gamma in da procedure.
def run2(num_trial):
    model = Esn(n_inputs=3,
              n_outputs=3,
              n_reservoir=100,
              leaky_rate=0.9,
              spectral_radius=0.2,
              random_state=num_trial,
              sparsity=0.4,
              silent=True,
              ridge_param=1e-5,
              washout=400)
    idx = rank
    train_data, test_data = initialize_data(train_time=40, test_time=10, noise=None, random_seed=num_trial)
    test_inputs, test_targets = test_data[:, :], test_data[:, 1:]
    test_length = test_targets.shape[1]
    noise_train_data = train_data + np.random.multivariate_normal(np.zeros(3), np.eye(3) * specified_obs_noise,
                                                                  size=train_data.shape[1]).T

    model.fit(noise_train_data)
    prediction = model.auto_evolve(test_inputs[:, 0], n_iteration=test_length)
    out = prediction_power(test_targets, prediction)

    model.clear_mode()
    model.fit_da(noise_train_data, ensembles=500, observation_noise=gamma[idx], model_noise=None, initial_w_dist=5000.)
    prediction = model.auto_evolve(test_inputs[:, 0], n_iteration=test_length)
    pred_power_0 = prediction_power(test_targets, prediction)

    model.clear_mode()
    model.reset_wout()
    model.fit_da(noise_train_data, ensembles=500, observation_noise=gamma[idx], model_noise=None,
                 initial_w_dist=5000.)
    prediction = model.auto_evolve(test_inputs[:, 0], n_iteration=test_length)
    pred_power_1 = prediction_power(test_targets, prediction)

    # [lr, da_from_0, da_from_lr]
    return [out, pred_power_0, pred_power_1]



trials = 200
dyn_noise = [None, 5.]
obs_noise = np.logspace(-2, 1, 10)
specified_obs_noise = 1.
gamma = np.logspace(-2, 1, 20)
Gamma_range = np.logspace(-1, 5, 20)
if rank < 20:
    with Pool(processes=os.cpu_count()) as p:
        powers = p.map(run2, range(trials))
    print("done in rank", rank)
    comm.send(powers, dest=size - 1)

# experiment 1
if rank == size - 1:
    print(f"{rank} is waiting for receiving information")
    total_powers = []
    for id in range(size - 1):
        powers = comm.recv(source=id)
        assert len(powers) == trials
        total_powers.append(powers)
    total_powers = np.array(total_powers).reshape((20, trials, 3))
    fig = plt.figure(figsize=(5, 5))
    axes = dict()
    gs = gridspec.GridSpec(1, 1)
    gs.update(left=0.1, right=0.94, top=0.93, bottom=0.15, wspace=0.2, hspace=0.2)
    axes['A'] = plt.subplot(gs[0, 0])
    names = ["LR", "$W^{out}\sim N(0, \Gamma I)$", "$W^{out}\sim N(W^{LR}, \Gamma I)$"]
    for i in range(total_powers.shape[-1]):
        powers_mean = total_powers[:, :, i].mean(1)
        powers_std = total_powers[:, :, i].std(1)
        if i ==0 :
            axes["A"].plot(np.linspace(-2, 1, 20, endpoint=True), powers_mean, color="purple", label=names[i])
        else:
            axes['A'].errorbar(np.linspace(-2, 1, 20, endpoint=True), powers_mean, yerr=powers_std, fmt='o:', ecolor='hotpink', capsize=3., capthick=1., errorevery=2, label=names[i])
    # axes['A'].set_xlabel("$log \Gamma$")
    axes['A'].set_xlabel("$log \gamma$")
    axes['A'].set_ylabel("Power")
    axes['A'].legend(loc="best")
    np.savez("./results/dependence_gamma.npz", Gamma=Gamma_range, powers=total_powers)
    fig.savefig("./results/dependence_gamma.png", dpi=100)
    print("Done")

