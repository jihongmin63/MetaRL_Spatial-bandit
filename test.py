import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from numpy.random import multivariate_normal
from sklearn import preprocessing
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import RBF

if 'Generating MAB' :
    # Basic setup
    grid_size = 11
    x1 = np.linspace(0, grid_size-1, grid_size)
    x2 = np.linspace(0, grid_size-1, grid_size)
    X1, X2 = np.meshgrid(x1, x2)
    X = np.vstack([X1.ravel(), X2.ravel()]).T

    def rbf_kernel(X1, X2, lamb):
        dists = cdist(X1, X2, 'cityblock')
        K = np.exp(-0.5 * (dists / lamb)**2)
        return K

    # Generate hard, and smooth distribution w/ multrivaraite_normal
    hard = []
    smooth = []
    num_samples = 20

    l = 1
    K = rbf_kernel(X, X, l)
    mean = np.zeros(len(X))

    for i in range(num_samples) :
        f_prior = multivariate_normal(mean, K)
        min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 85))
        f_prior = min_max_scaler.fit_transform(f_prior.reshape(-1, 1)).flatten()
        hard.append(f_prior.reshape(grid_size, grid_size))

    l = 2
    K = rbf_kernel(X, X, l)
    mean = np.zeros(len(X))

    for j in range(num_samples):
        f_prior = multivariate_normal(mean, K)
        min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 85))
        f_prior = min_max_scaler.fit_transform(f_prior.reshape(-1, 1)).flatten()
        smooth.append(f_prior.reshape(grid_size, grid_size))

    # Smooth
    plt.figure(figsize=(6, 5))
    plt.title("Spatially Correlated Smooth MAB, prior")
    plt.imshow(smooth[0], origin='lower', cmap='OrRd')
    for i in range(grid_size):
        for j in range(grid_size):
            value = smooth[0][i, j]
            plt.text(j, i, f"{value:.2f}", ha='center', va='center', color='black', fontsize=7)
    plt.colorbar(label="Function Value")
    plt.xticks([i for i in range(grid_size)])
    plt.yticks([i for i in range(grid_size)])
    plt.show()
    plt.savefig('smooth')

    # Hard
    plt.figure(figsize=(6, 5))
    plt.title("Spatially Correlated Hard MAB, prior")
    plt.imshow(hard[0], origin='lower', cmap='OrRd')
    for i in range(grid_size):
        for j in range(grid_size):
            value = hard[0][i, j]
            plt.text(j, i, f"{value:.2f}", ha='center', va='center', color='black', fontsize=7)
    plt.colorbar(label="Function Value")
    plt.xticks([i for i in range(grid_size)])
    plt.yticks([i for i in range(grid_size)])
    plt.show()
    plt.savefig('hard')

if False :
    def get(ground, x, y) :
        return ground[x, y] + np.random.normal(0, 1)

    def select(gpr, tau, typ, prev_x, prev_y, beta = 20, debug = 0) :
        if typ == 'MG' :
            results = gpr.predict(X_every)
        elif typ == 'VG' :
            _, results = gpr.predict(X_every, return_std = True)
        elif typ ==  'UCB' :
            mean, std = gpr.predict(X_every, return_std = True)
            results = mean + beta * std
    
        if local and (prev_x > -1 and prev_y > -1) :
            y = np.arange(grid_size).reshape(-1, 1)
            x = np.arange(grid_size).reshape(1, -1)
            manhatten = np.abs(x - prev_x) + np.abs(y - prev_y)
            manhatten[prev_x, prev_y] = 1
            np.reciprocal(manhatten)
            results = np.multiply(results, manhatten)

        results = results / tau
        results = results - np.max(results)
        results = np.exp(results)
        results = results  / np.max(results)
        ans = np.argmax(results)
        if(debug) :
            print("Select w/ prob %.2f" % (results[ans]))

        return int(ans / grid_size), ans % grid_size

    def learn(ground_truth, lamb, typ, iteration = 40, tau = 0.1, debug = 0) :
        init_x, init_y = np.random.randint(0, grid_size, 2)
        prev_x, prev_y = -1, -1
        X = np.array([[init_x, init_y]])
        Y = np.array([get(ground_truth, init_x, init_y)])
        kernel = RBF(lamb)

        for i in range(iteration) :
            gpr = GPR(kernel=kernel, random_state=0).fit(X, Y)
            x, y = select(gpr, tau, typ, prev_x, prev_y, debug = debug)
            res = get(ground_truth, x, y)
            X = np.vstack([X, [x, y]])
            Y = np.append(Y, res)
            if (debug) :
                print("Search (%d, %d) -> %.2f" % (x, y, res))
        return Y
    X_every = np.vstack([X2.ravel(), X1.ravel()]).T
    local = True

    # data generation
    if smooth :
        res_smooth_MG = np.array([learn(smooth[0], 2, 'MG')])
        res_smooth_VG = np.array([learn(smooth[0], 2, 'VG')])
        res_smooth_UCB = np.array([learn(smooth[0], 2, 'UCB')])
        for i in range(1, 20) :
            res_smooth_MG = np.vstack([res_smooth_MG, learn(smooth[i], 2, 'MG')])
            res_smooth_VG = np.vstack([res_smooth_VG, learn(smooth[i], 2, 'VG')])
            res_smooth_UCB = np.vstack([res_smooth_UCB, learn(smooth[i], 2, 'UCB')])

    if hard :
        res_hard_MG = np.array([learn(hard[0], 1, 'MG')])
        res_hard_VG = np.array([learn(hard[0], 1, 'VG')])
        res_hard_UCB = np.array([learn(hard[0], 1, 'UCB')])
        for i in range(1, 20) :
            res_hard_MG = np.vstack([res_hard_MG, learn(hard[i], 1, 'MG')])
            res_hard_VG = np.vstack([res_hard_VG, learn(hard[i], 1, 'VG')])
            res_hard_UCB = np.vstack([res_hard_UCB, learn(hard[i], 1, 'UCB')])

    # plot average
    mg = res_smooth_MG[:, 1:].mean(axis = 0)
    vg = res_smooth_VG[:, 1:].mean(axis = 0)
    ucb = res_smooth_UCB[:, 1:].mean(axis = 0)

    plt.plot(mg, label='MG')
    plt.plot(vg, label='VG')
    plt.plot(ucb, label='UCB')
    plt.legend(loc = 'upper left')

    for i in range(40) :
        plt.text(i, mg[i], f'{mg[i]:.1f}', fontsize = 5)
        plt.text(i, vg[i], f'{vg[i]:.1f}', fontsize = 5)
        plt.text(i, ucb[i], f'{ucb[i]:.1f}', fontsize = 5)

    plt.show()