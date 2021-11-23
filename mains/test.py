import numpy as np
import matplotlib.pyplot as plt


def plot_reward(num):
    a = np.load(f"./cql_runs/{num}/reward_tran.npy")
    plt.plot(a)
    # plt.title(f"{algo}_{env_name}_{aseed}")
    plt.show()


if __name__ == '__main__':
    plot_reward(861225)