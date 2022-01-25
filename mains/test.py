import numpy as np
import matplotlib.pyplot as plt


def plot_reward(num):
    a = np.load(f"./cql_runs/{num}/reward_tran.npy")
    plt.plot(a)
    plt.title(f"{num}")
    plt.show()

# def

if __name__ == '__main__':
    plot_reward("hopper_311691")
    plot_reward("halfcheetah_88209")
    plot_reward("walker2d_873929")
    plot_reward("ant_713035")