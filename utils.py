import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def to_cartesian(r, theta, phi):
    """_summary_

    Args:
        r (_type_): _description_
        theta (_type_): _description_
        phi (_type_): _description_

    Returns:
        _type_: _description_
    """
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z

def to_polar(x, y, z):
    """_summary_

    Args:
        x (_type_): _description_
        y (_type_): _description_
        z (_type_): _description_

    Returns:
        _type_: _description_
    """
    r = np.linalg.norm([x, y, z])
    phi = np.arccos(x / np.sqrt(x**2 + y**2)) * (-1 if y < 0 else 1)
    theta = np.arccos(z / r)
    return r, theta, phi

   
def create_hist(mc_data, vae_data, xlabel, save, n_bins=100): 
    minimum = min(mc_data.min(), vae_data.min())
    maximum = max(mc_data.max(), vae_data.max())
    bins = [minimum + i * ((maximum - minimum) / (n_bins - 1)) for i in range(n_bins)]
    
    plt.figure(figsize=(7, 7))
    plt.hist(vae_data, histtype='step', label="VAE", bins=bins)
    plt.hist(mc_data, histtype='step', label="MC", bins=bins)
    
    plt.ylabel('count')
    plt.xlabel(xlabel)
    plt.legend()
    plt.savefig(save)
    
    
def plot_all(mc_data, vae_data):
    # Get polar coordinates
    x1_polar = np.array(list(map(lambda x : to_polar(*x[1:4]), np.array(vae_data))))
    x2_polar = np.array(list(map(lambda x : to_polar(*x[4:7]), np.array(vae_data))))

    # Plots for x1
    create_hist(mc_data['E1'], vae_data['E1'] , "energy x1", "results/energy_x1.png")
    create_hist(mc_data['p_x1'], vae_data['p_x1'] , "px x1", "results/px_x1.png")
    create_hist(mc_data['p_y1'], vae_data['p_y1'] , "py x1", "results/py_x1.png")
    create_hist(mc_data['p_z1'], vae_data['p_z1'] , "pz x1", "results/pz_x1.png")
    create_hist(mc_data['theta1'], x1_polar[:,1], "theta x1", "results/theta_x1.png")
    create_hist(mc_data['phi1'], x1_polar[:,2], "phi x1", "results/phi_x1.png")
    
    # Plots for x2
    create_hist(mc_data['E2'], vae_data['E2'] , "energy x2", "results/energy_x2.png")
    create_hist(mc_data['p_x2'], vae_data['p_x2'] , "px x2", "results/px_x2.png")
    create_hist(mc_data['p_y2'], vae_data['p_y2'] , "py x2", "results/py_x2.png")
    create_hist(mc_data['p_z2'], vae_data['p_z2'] , "pz x2", "results/pz_x2.png")
    create_hist(mc_data['theta2'], x2_polar[:,1], "theta x2", "results/theta_x2.png")
    create_hist(mc_data['phi2'], x2_polar[:,2], "phi x2", "results/phi_x2.png")
    
    
if __name__ == "__main__":
    mc_data = pd.read_csv('data/mc_sim_full.csv', dtype=float, header=0)
    vae_data = pd.read_csv('data/vae.csv', dtype=float, header=0)
    plot_all(mc_data, vae_data)