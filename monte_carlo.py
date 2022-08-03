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
        # # # Draw random numbers for the energy of x1 and compute energy of x2
        # # e1 = random.randint(0, total_energy)
        # # e2 = total_energy - e1

        # # Draw random numbers for velocities
        # v1 = random.random()
        # v2 = random.random()

        # # Compute the mass
        # m = compute_mass(v1, v2, total_energy)

        # # Compute the energies
        # e1 = gamma(v1) * m
        # e2 = gamma(v2) * m

        # # Compute the 3-velocity vectors
        # vvec1, vvec2 = generate_vvec(v1, v2, e1, e2)
        # # assert(np.linalg.norm(vvec1) == v1 and np.linalg.norm(vvec2) == v2)

        # # Compute the 3-momentum vectors
        # p1 = e1 * vvec1
        # p2 = e2 * vvec2
        # # assert(p1 + p2 == np.zeros(3))

        # # Create final 4-vectors
        # breakpoint()
        # res.append(np.concatenate(([m, e1, v1], vvec1, p1, [e2, v2], vvec2, p2)))

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
        x1 = np.concatenate(([e1], p1))
        x2 = np.concatenate(([e2], p2))

        _, theta1, phi1 = utils.to_polar(*x1[1:])
        _, theta2, phi2 = utils.to_polar(*x2[1:])
        res.append(np.concatenate((x1, [theta1, phi1], x2, [theta2, phi2])))

    res = pd.DataFrame(np.array(res), columns=[
                       "E1", "p_x1", "p_y1", "p_z1", "theta1", "phi1", "E2", "p_x2", "p_y2", "p_z2", "theta2", "phi2"])
    res.to_csv(f"data/{out}_full.csv")
    res[["E1", "p_x1", "p_y1", "p_z1", "E2", "p_x2",
        "p_y2", "p_z2"]].to_csv(f"data/{out}.csv")
    return res


def gamma(v: float):
    """_summary_

    Args:
        v (_type_): _description_

    Returns:
        _type_: _description_
    """
    return 1 / np.sqrt(1 - (v ** 2))


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
    data = monte_carlo_sim(n=100_000)
    utils.create_plots(data, info="mc")


# def compute_mass(v1, v2, total_energy):
#     return total_energy / (gamma(v1) + gamma(v2))
# def generate_vvec(v):

#     vvec = np.random.uniform(size=(3))
#     vvec = v * vvec / np.linalg.norm(vvec)
#     return vvec

# def generate_vvec(v1, v2, e1, e2):
#     #     # # Genrate random angles for x1
#     #     # theta1_deg = random.random() * 360
#     #     # theta1_rad = np.radians(theta1_deg)
#     #     # phi1_deg = random.random() * 360
#     #     # phi1_rad = np.radians(phi1_deg)

#     #     # # Compute velocity vectors for x1 and x2
#     #     # vvec1 = to_cartesian(v1, theta1_rad, phi1_rad)
#     #     # vvec2 = -e1 * vvec1 / e2

#     #     # # Compute angles for x2
#     #     # r2, theta2_rad, phi2_rad = to_polar(v2, *vvec2)
#     #     # theta2_deg = np.degrees(theta2_rad)
#     #     # phi2_deg = np.degrees(phi2_rad)
#     #     # breakpoint()
#     #     # return vvec1, theta1_deg, phi1_deg, vvec2, theta2_deg, phi2_deg
#     #     vvec1 = np.random.uniform(size=(3))
#     #     vvec1 = v1 * vvec1 / np.linalg.norm(vvec1)
#     #     vvec2 = -e1 * vvec1 / e2
#     #     return vvec1, vvec2


#     # v11 = random.random()
#     # v12 = random.random()

#     # x = e2 ** 2 * (v1 ** 2 - v2 ** 2)
#     # y = x + e1 ** 2 * (v11 ** 2 + v12 ** 2) - e2 ** 2 * (v11 ** 2 + v12 ** 2)

#     # v13 = y / (e2 ** 2 - e1 ** 2)
#     # vvec1 = np.array([v11, v12, v13])
#     # vvec2 = -e1 * vvec1 / e2
#     # return vvec1, vvec2
#     vvec1 = np.random.uniform(size=(3))
#     vvec1 = v1 * vvec1 / np.linalg.norm(vvec1)
#     vvec2 = -e1 * vvec1 / e2
#     return vvec1, vvec2


# def compute_mass(energy: int, velocity: float):
#     return energy * np.sqrt(1 - velocity ** 2)

# def compute_velocity(energy: int, mass: float):
#     return np.sqrt(1 - (mass ** 2 / energy **2))
