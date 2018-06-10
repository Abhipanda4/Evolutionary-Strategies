import os

import numpy as np
import matplotlib.pyplot as plt

DNA_SIZE = 1
N_GEN = 100
DNA_BOUNDS = [0, 5]
T = 5
TAU = 0.9

def F(x):
    return np.sin(10*x)*x + np.cos(2*x)*x

def mutate(x_parent, sigma):
    x_kid = x_parent + sigma * np.random.randn(DNA_SIZE)
    x_kid = np.clip(x_kid, DNA_BOUNDS[0], DNA_BOUNDS[1])

    # evalute fitness of both parent and kid
    f_parent = F(x_parent)
    f_kid = F(x_kid)

    # select the better individual
    if f_parent <= f_kid:
        return x_kid, True
    else:
        return x_parent, False

def adjust_step_size(sigma, num_success):
    success_ratio = num_success / T
    target = 0.2
    x = 0
    if success_ratio > target:
        x = 1
    return sigma * np.exp(1/np.sqrt(DNA_SIZE+1) * (x - target)/(1 - target))

def main():
    # plotting
    plt.ion()
    x_plot = np.linspace(DNA_BOUNDS[0], DNA_BOUNDS[1], 100)
    y_plot = np.array([F(x) for x in x_plot])

    # initialize root parent randomly
    X = DNA_BOUNDS[1] * np.random.rand(DNA_SIZE)

    # initial mutation step size
    sigma = 5

    # counter for number of successful generations
    # in T evolution loops
    num_success = 0

    for g in range(1, N_GEN + 1):
        if g % T == 0:
            sigma = adjust_step_size(sigma, num_success)
            num_success = 0

        X_prime, success = mutate(X, sigma)

        # increment num_success if kid is better than parent
        if success:
            num_success += 1

        # plot child and kid
        plt.cla()
        plt.scatter(X, F(X), color='b', lw=5, alpha=0.3)
        plt.scatter(X_prime, F(X_prime), color='g', lw=5, alpha=0.3)
        plt.plot(x_plot, y_plot)
        plt.pause(0.05)

        # current child is the next genration's parent
        X = X_prime

    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()
