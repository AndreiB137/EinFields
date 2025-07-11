""" MIT License
# 
# Copyright (c) 2025 Andrei Bodnar (Dept of Physics and Astronomy, University of Manchester,United Kingdom), Sandeep S. Cranganore (Ellis Unit, LIT AI Lab, JKU Linz, Austria) and Arturs Berzins (Ellis Unit, LIT AI Lab, JKU Linz, Austria)
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#           Geodesic solvers run on different spacetime metrics
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

import jax
import jax.numpy as jnp
from typing import Tuple, Callable

import diffrax
from differential_geometry import diffgeo
from general_relativity.geodesics.geodesics_solver import solver

from general_relativity.metrics.kerr import kerr_metric_boyer_lindquist
from general_relativity.coordinate_transformations.coord_transform import oblate_spheroid_to_cartesian

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

def create_solver_params(t1 : int,
                         init_step : float = 1e-5, 
                         rtol : float = 1e-8, 
                         atol : float = 1e-9, 
                         max_steps : int = 10000) -> dict:
    t0 = 0
    solver_params = {
        "t0": t0,
        "t1": t1,
        "dt": init_step,
        "saveat": diffrax.SaveAt(steps=True),
        "solver": diffrax.Kvaerno5(),
        "stepsize_controller": diffrax.PIDController(rtol, atol),
        "throw": False,
        "max_steps": max_steps,
        "progress_meter": diffrax.TextProgressMeter(),
    }
    return solver_params

def kerr_init_condition(r0 : float, theta0 : float, phi0 : float, M : float, a : float, L_z : float, v0 : float, vr0: float, v_theta0 : float, v_phi0 : float = None) -> Tuple[jax.Array, jax.Array]:
    """
        Creates the initial conditions for the Kerr metric in Boyer-Lindquist coordinates.
    """
    
    L_z = L_z

    r0 = r0
    theta0 = theta0
    phi0 = phi0
    x0 = jnp.array([0., r0, theta0, phi0])

    sigma = r0**2 + a**2*jnp.cos(theta0)**2 

    kerr_metric = kerr_metric_boyer_lindquist(x0, M=M, a=a)
    kerr_inv = jnp.linalg.inv(kerr_metric)
    csi = jnp.sqrt(-kerr_inv[0, 0])

    lorentz_factor = 1 / jnp.sqrt(1 - v0**2)
    # print(lorentz_factor)
    # print(csi * lorentz_factor)
    v_t0 = csi * lorentz_factor

    v_theta0 = v_theta0 * lorentz_factor / jnp.sqrt(sigma)
    if v_phi0 is None:
        v_phi0 = (L_z  - kerr_metric[0, 3] * v_t0) / kerr_metric[3, 3]

    v0 = jnp.array([v_t0, vr0, v_theta0, v_phi0])

    return x0, v0

def schwarzschild_init_condition(r0 : float, theta0 : float, phi0 : float, M : float, v0 : float, vr0: float, v_theta0 : float, v_phi0 : float = 0.) -> Tuple[jax.Array, jax.Array]:
    """
        Creates the initial conditions for the Schwarzschild metric in spherical coordinates.
    """
    x0 = jnp.array([0., r0, theta0, phi0])

    lorentz_factor = 1 / jnp.sqrt(1 - v0**2)

    v_t0 = lorentz_factor / jnp.sqrt(1 - 2 * M / r0)

    v_phi0 = v0 * lorentz_factor * jnp.cos(phi0) / r0

    v0 = jnp.array([v_t0, vr0, v_theta0, v_phi0])

    return x0, v0

def run_geodesic(model : diffgeo, init_condition : jax.Array, solver_params : dict, coord_transform : Callable[[jax.Array], jax.Array] = lambda coords : coords) -> jax.Array:
    """
    Runs the geodesic solver for a given metric object defined through the tensor calculus package and with the specified initial conditions.
    The function uses the diffrax library to solve the geodesic equation. If using a neural network based metric, don't forget to reshape the
    output to a 4x4 matrix.

    Parameters
    ----------
    model : DifferentialGeometry
        The differential geometry object initialized with the metric. Can be either analytic or neural network based.
    
    init_condition : jax.Array
        The initial condition for the geodesic equation. It can be a batch of initial conditions.

    solver_params : dict
        The parameters for the solver, including t0, t1, dt, saveat, solver, stepsize_controller, throw, max_steps, and progress_meter.

    coord_transform : Callable, optional
        A function to transform the coordinates where the Christoffel symbols are evaluated inside the solver.
        This is useful to fix coordinates which are out of the training domain of the neural field. For example,
        if the neural field is trained only for t=0 or for phi in the range(0, 2*pi).
    
    Returns
    -------

    trajectories : jax.Array 
        The trajectories of the geodesic in the form of a 4D array with shape (batch_size, num_timesteps, 4).
        Num_timesteps is the number of steps taken by the solver.

    """

    solution = solver(model, coord_transform=coord_transform, **solver_params)(init_condition)
    trajectories = solution.ys[:, :, :4]

    return trajectories

def create_plot(xlim : int, 
                ylim : int, 
                config : dict, 
                orbit_name : str, 
                x_gt : jax.Array, 
                y_gt : jax.Array,
                x_nef : jax.Array = None,
                y_nef : jax.Array = None):
    
    """
    Creates the geodesic plot with extra visualization of the horizons.

    Parameters
    ----------
    xlim : int
        The x-axis limit in the range [-xlim, xlim].
    ylim : int
        The y-axis limit in the range [-ylim, ylim].
    config : dict
        The configuration dictionary containing the metric parameters.
        This should generally include the name of the metric, the mass (M), the spin parameter (a) and optionally the charge (Q).
    orbit_name : str
        The name of the orbit or simply the name of the figure.
    x_gt : jax.Array
        The x-coordinates of the geodesic or batch of geodesics.
    y_gt : jax.Array
        The y-coordinates of the geodesic or batch of geodesics.

    """

    fig, ax = plt.subplots(figsize=(6, 6))
    # Set plot limits and aspect ratio
    ax.set_xlim(-xlim, xlim)
    ax.set_ylim(-ylim, ylim)
    ax.set_aspect('equal')
    center_x, center_y = 0, 0
    r_s = 2 * config["M"]  # Schwarzschild radius
    if config["name"] == "Schwarzschild":
        event_horizon = r_s
        circle_event_horizon = plt.Circle((center_x, center_y), event_horizon, color='black', label='Event horizon')
        ax.add_artist(circle_event_horizon)
    elif config["name"] == "Kerr":
        inner_event_horizon = jnp.sqrt(((r_s - jnp.sqrt(r_s**2 - 4*config["a"]**2)) / 2)**2 + config["a"]**2)
        outer_event_horizon = jnp.sqrt(((r_s + jnp.sqrt(r_s**2 - 4*config["a"]**2)) / 2)**2 + config["a"]**2)
        outer_ergosphere = jnp.sqrt(r_s**2+config["a"]**2)
        ring_singularity = config["a"]

        circle_inner_event = plt.Circle((center_x, center_y), inner_event_horizon, color='maroon', alpha=0.6, label='Inner horizon')
        circle_outer_event = plt.Circle((center_x, center_y), outer_event_horizon, color='blue', alpha=0.6, label='Outer horizon')
        circle_outer_ergosphere = plt.Circle((center_x, center_y), outer_ergosphere, color='cornflowerblue', alpha=0.3, label='Outer ergosphere')
        circle_ring_singularity = plt.Circle((center_x, center_y), ring_singularity, color='red', alpha=0.5, label='Ring singularity')
        ax.add_artist(circle_outer_event)
        ax.add_artist(circle_inner_event)
        ax.add_artist(circle_outer_ergosphere)
        ax.add_artist(circle_ring_singularity)

    ax.set_xlabel('$x$ ($GM/c^2$)')
    ax.set_ylabel('$y$ ($GM/c^2$)')

    ax.set_title(orbit_name, fontsize=10)

    lc_gt = LineCollection(
            [jnp.column_stack([x_gt[i], y_gt[i]]) for i in range(len(x_gt))],
            color='green',
            alpha=1.0,
            label='Geodesic',
            linestyles='solid',
            linewidths=0.8,
    )
    if x_nef is not None and y_nef is not None:
        lc_nef = LineCollection(
            [jnp.column_stack([x_nef[i], y_nef[i]]) for i in range(len(x_nef))],
            color='red',
            alpha=1.,
            label='NEF Geodesic',
            linestyles='solid',
            linewidths=0.8,
        )
        ax.add_collection(lc_nef)

    ax.add_collection(lc_gt)
    ax.legend(prop={'size': 6})
    plt.show()

if __name__ == "__main__":

    jax.config.update("jax_enable_x64", True)
    # jax.config.update("jax_platform_name", "cpu")

    ### Example with the Kerr metric Zackiger orbit
    gt_metric = lambda coords: kerr_metric_boyer_lindquist(coords, M=1.0, a=0.95, Q=0.0)
    gt_metric_tensor = diffgeo(gt_metric)

    solver_params = create_solver_params(
        t1=1641,
        rtol=1e-8,
        atol=1e-9,
        max_steps=10000
    )

    x0, v0 = kerr_init_condition(r0=6.5, theta0=jnp.pi / 2, phi0=0., M=1.0, a=0.95, L_z=-0.830327, v0=0.5, vr0=0., v_theta0=-jnp.cos(11 / 50) / 2)

    init_condition = jnp.concatenate([x0.reshape(1, -1), v0.reshape(1, -1)], axis=1)

    boyer_lindquist_trajectories = run_geodesic(gt_metric_tensor, init_condition, solver_params)

    print(jnp.min(boyer_lindquist_trajectories[:, :, 1]), jnp.max(boyer_lindquist_trajectories[:, 500, 1]))

    cartesian_coords = jax.vmap(jax.vmap(oblate_spheroid_to_cartesian, in_axes=(0, None)), in_axes=(0, None))(boyer_lindquist_trajectories, 0.95)

    create_plot(
        xlim=16,
        ylim=16,
        config={"name": "kerr", "M": 1.0, "a": 0.95},
        orbit_name="Zackiger orbit",
        x_gt=cartesian_coords[:, :, 1],
        y_gt=cartesian_coords[:, :, 2],
    )

