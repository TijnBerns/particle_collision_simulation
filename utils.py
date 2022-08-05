import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu, gaussian_kde
from pathlib import Path


def gamma(v: float):
    """Gamma compononent of energy-mass relation

    Args:
        v: total velocity

    Returns:
        gamma value
    """
    return 1 / np.sqrt(1 - (v ** 2))


def energy(p, m):
    """Computes the energy given total-momentum and invariant mass

    Args:
        p (float): total momentum
        m (float): invariant mass

    Returns:
        _type_: _description_
    """
    return np.sqrt(p ** 2 + m ** 2)


def to_cartesian(r, theta, phi):
    """Converts polar coordinates to cartesian coordinates.

    Args:
        r (float): The radius of the polar-coordinate system
        theta (float): The theta angle of the polar-coordinate system
        phi (float): The phi angle of the polar-coordinate system

    Returns:
        _type_: _description_
    """
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z


def to_polar(x, y, z):
    """Converts cartesian coordinates to polar coordinates.

    Args:
        x (float): The x-coordinate in the cartisian system 
        y (float): The x-coordinate in the cartisian system 
        z (float): The x-coordinate in the cartisian system 

    Returns:
        The corresponding polar coordinates
    """
    r = np.linalg.norm([x, y, z])
    phi = np.arccos(x / np.sqrt(x**2 + y**2)) * (-1 if y < 0 else 1)
    theta = np.arccos(z / r)
    return r, theta, phi


def create_hist(mc_data, vae_data, xlabel, save, n_bins=100):
    """Creates a single histogram given a two vectors

    Args:
        mc_data: column from the MC-simulation dataset
        vae_data : column from the VAE-generated dataset
        xlabel : x-label of the plot
        save : Information used to save the plot 
        n_bins (int,): the number of bins in the histogram. Defaults to 100.
    """
    minimum = min(mc_data.min(), vae_data.min())
    maximum = max(mc_data.max(), vae_data.max())
    bins = [minimum + i * ((maximum - minimum) / (n_bins - 1))
            for i in range(n_bins)]

    plt.figure(figsize=(5, 5))
    plt.hist(vae_data, histtype='step', label="VAE", bins=bins)
    plt.hist(mc_data, histtype='step', label="MC", bins=bins)

    plt.ylabel('count')
    plt.xlabel(xlabel)
    plt.legend()
    plt.savefig(save)


# def get_colormap(x, y):

#     xy = np.vstack([x, y])
#     z = gaussian_kde(xy)(xy)
#     return z


def create_scatter(data, save=''):
    """Creates scatter plots of the energy momentum relation
    Args:
        data: the dataset that is used to create a scatter plot o 
    """

    p1_vec = np.array(data[['p_x1', 'p_y1', 'p_z1']])
    p2_vec = np.array(data[['p_x2', 'p_y2', 'p_z2']])
    p1 = np.linalg.norm(p1_vec, axis=1)
    p2 = np.linalg.norm(p2_vec, axis=1)
    m1 = np.array(data['M1'])
    m2 = np.array(data['M1'])

    E1 = np.array(data['E1'])
    E1_hat = np.array(list(map(lambda pm: energy(*pm), zip(p1, m1))))
    E2 = np.array(data['E2'])
    E2_hat = np.array(list(map(lambda pm: energy(*pm), zip(p2, m2))))

    plt.figure(figsize=(7, 7))
    plt.scatter(E1, E1_hat)
    plt.xlabel("Generated energy")
    plt.ylabel("Computed energy")
    plt.savefig(f'results/{save}_ep1.png')

    plt.figure(figsize=(7, 7))
    plt.scatter(E2, E2_hat)
    plt.xlabel("Generated energy")
    plt.ylabel("Computed energy")
    plt.savefig(f'results/{save}_ep2.png')


def plot_all(mc_data, vae_data):
    """Creates histograms for all columns of the datasets

    Args:
        mc_data: Monte-Carlo dataset
        vae_data: Variational-auto-encoder dataset
    """

    # Get polar coordinates
    x1_polar = np.array(
        list(map(lambda x: to_polar(*x), np.array(vae_data[['p_x1', 'p_y1', 'p_z1']]))))
    x2_polar = np.array(
        list(map(lambda x: to_polar(*x), np.array(vae_data[['p_x2', 'p_y2', 'p_z2']]))))

    # Plots for x1
    create_hist(mc_data['E1'], vae_data['E1'],
                "energy x1", "results/energy_x1.png")
    create_hist(mc_data['M1'], vae_data['M1'],
                "mass x1", "results/mass_x1.png")
    create_hist(mc_data['p_x1'], vae_data['p_x1'],
                "px x1", "results/px_x1.png")
    create_hist(mc_data['p_y1'], vae_data['p_y1'],
                "py x1", "results/py_x1.png")
    create_hist(mc_data['p_z1'], vae_data['p_z1'],
                "pz x1", "results/pz_x1.png")
    create_hist(mc_data['r1'], x1_polar[:, 0], "radius x1", "results/r_x1.png")
    create_hist(mc_data['theta1'], x1_polar[:, 1],
                "theta x1", "results/theta_x1.png")
    create_hist(mc_data['phi1'], x1_polar[:, 2],
                "phi x1", "results/phi_x1.png")

    # Plots for x2
    create_hist(mc_data['E2'], vae_data['E2'],
                "energy x2", "results/energy_x2.png")
    create_hist(mc_data['M2'], vae_data['M2'],
                "mass x2", "results/mass_x2.png")
    create_hist(mc_data['p_x2'], vae_data['p_x2'],
                "px x2", "results/px_x2.png")
    create_hist(mc_data['p_y2'], vae_data['p_y2'],
                "py x2", "results/py_x2.png")
    create_hist(mc_data['p_z2'], vae_data['p_z2'],
                "pz x2", "results/pz_x2.png")
    create_hist(mc_data['r2'], x2_polar[:, 0], "radius x1", "results/r_x2.png")
    create_hist(mc_data['theta2'], x2_polar[:, 1],
                "theta x2", "results/theta_x2.png")
    create_hist(mc_data['phi2'], x2_polar[:, 2],
                "phi x2", "results/phi_x2.png")


def compare_all(mc_data, vae_data):
    """Compares two distributions using Mann-Whitney test

    Args:
        mc_data: Monte-Carlo dataset
        vae_data: Variational-auto-encoder dataset
    """
    # Get polar coordinates
    x1_polar = np.array(
        list(map(lambda x: to_polar(*x), np.array(vae_data[['p_x1', 'p_y1', 'p_z1']]))))
    x2_polar = np.array(
        list(map(lambda x: to_polar(*x), np.array(vae_data[['p_x2', 'p_y2', 'p_z2']]))))

    # Compare the two distributions
    print(f"E1: {mannwhitneyu(mc_data['E1'], vae_data['E1'])}")
    print(f"M1: {mannwhitneyu(mc_data['M1'], vae_data['M1'])}")
    print(f"p_x1: {mannwhitneyu(mc_data['p_x1'], vae_data['p_x1'])}")
    print(f"p_y1: {mannwhitneyu(mc_data['p_y1'], vae_data['p_y1'])}")
    print(f"p_z1: {mannwhitneyu(mc_data['p_z1'], vae_data['p_z1'])}")
    print(f"r1: {mannwhitneyu(mc_data['r1'], x1_polar[:,0])}")
    print(f"theta1: {mannwhitneyu(mc_data['theta1'], x1_polar[:,1])}")
    print(f"phi1: {mannwhitneyu(mc_data['phi1'], x1_polar[:,1])}")

    print(f"E2: {mannwhitneyu(mc_data['E2'], vae_data['E2'])}")
    print(f"M2: {mannwhitneyu(mc_data['M2'], vae_data['M2'])}")
    print(f"p_x2: {mannwhitneyu(mc_data['p_x2'], vae_data['p_x2'])}")
    print(f"p_y2: {mannwhitneyu(mc_data['p_y2'], vae_data['p_y2'])}")
    print(f"p_z2: {mannwhitneyu(mc_data['p_z2'], vae_data['p_z2'])}")
    print(f"r2: {mannwhitneyu(mc_data['r2'], x2_polar[:,0])}")
    print(f"theta2: {mannwhitneyu(mc_data['theta2'], x2_polar[:,2])}")
    print(f"phi2: {mannwhitneyu(mc_data['phi2'], x2_polar[:,1])}")


def get_loss():
    path = Path('results/')
    for f in path.glob('loss*.csv'):
        data = pd.read_csv(f)
        m = data['test_loss'].min()
        print(f"{str(f):<20}{m:<20}")


if __name__ == "__main__":
    mc_data = pd.read_csv('data/mc_sim_full.csv', dtype=float, header=0)
    vae_data = pd.read_csv('data/vae.csv', dtype=float, header=0)
    plot_all(mc_data, vae_data)
    compare_all(mc_data, vae_data)
    create_scatter(mc_data, save='mc')
    create_scatter(vae_data, save='vae')
