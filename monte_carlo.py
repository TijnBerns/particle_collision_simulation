import numpy as np
import pandas as pd
from tqdm import tqdm
import utils


def monte_carlo_sim(total_energy=400, energy_res=0.10, angle_res=0.05, n=100_000, out="mc_sim"):
    """_summary_

    Args:
        total_energy (int, optional): _description_. Defaults to 400.
        energy_res (float, optional): _description_. Defaults to 0.10.
        angle_res (float, optional): _description_. Defaults to 0.05.
        n (int, optional): _description_. Defaults to 1000.
        out (str, optional): _description_. Defaults to "monte_carlo_sim.csv".

    Returns:
        _type_: _description_
    """
    res = []
    for _ in tqdm(range(n)):
        rest_mass = np.random.normal(loc=100, scale=10)
        e1 = total_energy / 2
        e2 = total_energy / 2
        radius = np.sqrt(e1 ** 2 - rest_mass ** 2)

        # Generate 3-momentum vectors
        p1 = generate_momentum(radius)
        p2 = -p1

        # Smear the angles of the 3-momentum vectors and energy
        p1 = smear_angles(p1, angle_res)
        p2 = smear_angles(p2, angle_res)
        e1 = e1 * np.random.normal(loc=1.0, scale=energy_res)
        e2 = e2 * np.random.normal(loc=1.0, scale=energy_res)

        # Create final 4-vectors
        x1 = np.concatenate(([e1, rest_mass], p1))
        x2 = np.concatenate(([e2, rest_mass], p2))

        polar1 = utils.to_polar(*p1)
        polar2 = utils.to_polar(*p2)
        res.append(np.concatenate((x1, list(polar1), x2, list(polar2))))

    res = pd.DataFrame(np.array(res), columns=[
                       "E1", "M1", "p_x1", "p_y1", "p_z1", "r1", "theta1", "phi1", "E2", "M2", "p_x2", "p_y2", "p_z2", "r2", "theta2", "phi2"])
    res.to_csv(f"data/{out}_full.csv")
    res[["E1", "M1", "p_x1", "p_y1", "p_z1", "E2", "M2", "p_x2",
        "p_y2", "p_z2"]].to_csv(f"data/{out}.csv")
    return res


def generate_momentum(radius: float):
    """_summary_

    Args:
        radius (_type_): _description_

    Returns:
        _type_: _description_
    """
    v_vec = np.random.uniform(-1, 1, size=(3))
    return radius * v_vec / np.linalg.norm(v_vec)


def smear_angles(p, resolution):
    """_summary_

    Args:
        p (_type_): _description_
        resolution (_type_): _description_

    Returns:
        _type_: _description_
    """
    r, theta, phi = utils.to_polar(*p)
    theta = theta * np.random.normal(loc=1, scale=resolution)
    phi = phi * np.random.normal(loc=1, scale=resolution)
    return np.array(utils.to_cartesian(r, theta, phi))


if __name__ == "__main__":
    monte_carlo_sim(n=100_000)
