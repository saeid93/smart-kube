import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import matplotlib
matplotlib.use("Agg")

def plot_workload(timesteps, workload, plot_smoothing, i):
    """
    plot the workloads per each resource
    resources:
        1. ram
        2. cpu
    return:
        fig object
        ready to be plotted
    """

    ram = workload[0, :]
    cpu = workload[1, :]

    timesteps = np.arange(0, timesteps)
    fig, axs = plt.subplots(2, 1)

    # plot ram
    axs[0].plot(timesteps, ram)
    # axs[0].plot(timesteps, savgol_filter(ram, plot_smoothing, 3), color='red')
    axs[0].set_xlabel('ram')
    axs[0].set_ylabel('usage fraction')
    axs[0].grid(True)

    # plot cpu
    axs[1].plot(timesteps, cpu)
    # axs[1].plot(timesteps, savgol_filter(cpu, plot_smoothing, 3), color='red')
    axs[1].set_xlabel('cpu')
    axs[1].set_ylabel('usage fraction')
    axs[1].grid(True)

    fig.tight_layout()

    return fig
