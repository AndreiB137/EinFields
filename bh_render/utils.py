import jax
import jax.numpy as jnp
import numpy as np
from typing import Callable, Tuple
from general_relativity.geodesics import solver
from general_relativity.geodesics.utils_geodesics import create_solver_params
from differential_geometry import diffgeo
from scipy.interpolate import interp1d
from PIL import Image
import time

def tetrad(coords: jax.Array) -> jax.Array:
    """Returns the Schwarzschild tetrad in spherical coordinates.
    
    Parameters
    ----------
    coords : jax.Array
        Array of shape (4,) of four scalars `[t, r, θ, φ]` representing the
        spacetime point in spherical coordinates.

    Returns
    -------
    jax.Array
        A `(4, 4)` JAX array representing the Schwarzschild tetrad `e_{a}^{μ}` in spherical coordinates.
        The tetrad is defined such that `η_{ab} = e_{a}^{μ} e_{b}^{ν} g_{μν}`,
        where `η_{ab}` is the Minkowski metric tensor in spherical coordinates.
    
    """
    t, r, theta, phi = coords

    tet = jnp.zeros((4, 4))
    tet = tet.at[0, 0].set(1 / jnp.sqrt(1 - 2 / r))  # t component
    tet = tet.at[1, 1].set(jnp.sqrt(1 - 2 / r))  # r component
    tet = tet.at[2, 2].set(1 / r)  # theta component
    tet = tet.at[3, 3].set(1 / (r * jnp.sin(theta)))  # phi component

    return tet

def spherical_to_cartesian(theta: np.array, phi: np.array) -> Tuple[np.array, np.array, np.array]:
    """
        Convert spherical coordinates to cartesian.
        This is a special version with the radius fixed to 1.
        So, the output vector is just the an orientation vector.
    """
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return x, y, z


def cartesian_to_spherical(x: np.array, y: np.array, z: np.array) -> Tuple[np.array, np.array]:
    """Convert cartesian coordinates to spherical."""
    # doesn't compute r because chosen equal to 1
    with np.errstate(all='ignore'):
        theta = np.arccos(z)
    phi = np.arctan2(y, x)

    return theta, phi

def rotation_matrix(beta: np.array) -> np.array:
    """
    Return the rotation matrix associated with counterclockwise rotation
    about the x axis by beta degree.
    """
    beta = np.array(beta)
    aa_bb, ab2neg = np.cos(beta), np.sin(beta)
    zero, one = np.zeros(beta.shape), np.ones(beta.shape)

    return np.array([[one, zero, zero],
                     [zero, aa_bb, -ab2neg],
                     [zero, ab2neg, aa_bb]])

def generate_phi_samples(sparse_samples: int, 
                         sparse_angle: float = 0.7, 
                         critical_angle: float = 0.40620804, 
                         density_factor: int = 4) -> jax.Array:
    """
        Generate phi samples for the observer's initial conditions. These samples
        represent the viewing angles of the observer in the equatorial plane.
        
        Note, this is only meant for Schwarzschild black hole and 
        observer's initial conditions on the equatorial plane.
        
        Parameters
        ----------
        sparse_samples : int
            Number of samples to generate in the sparse region, usually meaning rays further from the black hole.
        sparse_angle : float
            Angle in radians to define the boundary between the sparse/dense region around the observer's viewing direction.
        critical_angle : float
            Angle in radians defining the minimum angle after which all rays fall in the black hole.
        density_factor : int
            Factor by which to increase the number of samples in the dense region.
        
        Returns
        -------
        jax.Array
            A JAX array containing the phi samples.
    """
    if sparse_angle <= critical_angle:
        raise ValueError("sparse_angle must be greater than critical_angle.")
    if sparse_angle >= 2 * jnp.pi:
        raise ValueError("sparse_angle must be less than 2 * pi.")
    fov_phi = 2 * jnp.pi 
    phi_sparse = jnp.concatenate([
        jnp.linspace(fov_phi / 2, sparse_angle, sparse_samples, endpoint=False),
    ])
    phi_dense = jnp.concatenate([
        jnp.linspace(sparse_angle, critical_angle, sparse_samples * density_factor, endpoint=False),
    ])

    return jnp.sort(jnp.concatenate([phi_sparse, phi_dense]))  # Combine and sort angles

def create_initial_condition_equatorial_observer(phi_samples: jax.Array, observer_coords: jax.Array) -> jax.Array:
    
    """
        Create initial conditions for ray tracing from an observer located on the equatorial plane.
        If you see the local coordinates of the observer are modified by `π - phi_samples`, this is
        because the observer has a coordinate system centered at him.

        Note, this is only meant for Schwarzschild black hole and
        observer's initial conditions on the equatorial plane.

        Parameters
        ----------
        phi_samples : jax.Array
            Array of phi samples representing the viewing angles of the observer in the equatorial plane.
        observer_coords : jax.Array
            Coordinates of the observer in the form `[t, r, θ, φ]`, where
            `θ` is set to `π/2` for the equatorial plane.

        Returns
        -------
        jax.Array
            A JAX array containing the initial conditions for the observer.
    """

    if observer_coords[2] != jnp.pi / 2:
        raise ValueError("Observer coordinates must have theta set to pi/2 for equatorial plane.")
    
    vr_local = jnp.cos(np.pi - phi_samples)
    vtheta_local = jnp.zeros_like(phi_samples)  # Set theta to zero for equatorial plane
    vphi_local = jnp.sin(np.pi - phi_samples)  # Set z to zero for equatorial plane
    vt0_local = jnp.ones_like(phi_samples) * -1.0  # Set time component to 1.0

    v0_local = jnp.stack([vt0_local, vr_local, vtheta_local, vphi_local], axis=-1)  # [width, 4] (t, r, theta, phi)

    pos = jnp.tile(observer_coords.reshape(1, 4), (len(phi_samples), 1))  # [width, 4]

    tetrad_matrix = jax.vmap(tetrad)(pos)

    v0_global = jnp.einsum('kij, ki -> kj', tetrad_matrix, v0_local)  # Transform to global coordinates

    return jnp.concatenate([pos, v0_global], axis=-1)


def ray_trace_init_cond(init_cond: jax.Array, metric_fn: Callable, batch_size: int) -> np.array:

    """
        Ray trace the initial conditions using the geodesic solver.

        Parameters
        ----------
        init_cond : jax.Array
            Initial conditions for the geodesic equation, of shape `(width * height, 8)`, where the last dimension
            contains, for example in spherical coordinates: `[t, r, θ, φ, v_t, v_r, v_θ, v_φ]`.
        metric_fn : Callable
            Function to compute the metric at given coordinates.
        batch_size : int
            Number of trajectories to process at a time.

        Returns
        -------
        np.array
            Array of shape `(num_trajectories, num_steps, 4)` representing the traced trajectories.

    """

    init_cond = jnp.reshape(init_cond, (-1, 8))  # Reshape to (width * height, 8)

    solver_params = create_solver_params(
        t1=jnp.inf,  # Use inf to allow for indefinite tracing
        rtol=1e-7,
        atol=1e-7,
        max_steps=400
    )

    geodesic_solver = solver(
        diffgeo(metric_fn),
        **solver_params,
    )

    start_time = time.time()
    all_trajectories = []
    for i in range(0, init_cond.shape[0], batch_size):
        batch = init_cond[i:i + batch_size]
        trajectories = geodesic_solver(batch).ys[:, :, :4]
        all_trajectories.append(np.array(trajectories))

    end_time = time.time()
    print(f"Ray tracing completed in {end_time - start_time:.4f} seconds")
    return np.concatenate(all_trajectories, axis=0)  # Concatenate all trajectories

def generate_rendering_schwarzschild(trajectories: jax.Array, 
                                     render_shape: tuple, 
                                     background_img_path: str, 
                                     phi_samples, 
                                     r_exit: float, 
                                     eps: float = 1e-1) -> jax.Array:
    """
        Generate a rendering of the Schwarzschild black hole using the provided trajectories.

        Parameters
        ----------
        trajectories : jax.Array
            Array of shape `(num_trajectories, num_steps, 4)` representing the trajectories of the rays.
        render_shape : tuple
            Shape of the rendered image as `(height, width)`.
        background_img_path : str
            Path to the background image to be used in the rendering.
        phi_samples : jax.Array
            Array of phi samples representing the viewing angles of the observer in the equatorial plane.
        r_exit : float
            Radius at which the rays are coming from the background image, usually this is far away.
        eps : float, optional
            Trajectories with radius less than `r_exit + eps` are considered to have entered the black hole.
    
        Returns
        -------
        np.Array
            Numpy array representing the rendered image of the black hole.
    """
    image = Image.open(background_img_path, "r")
    image_width, image_height = image.size
    mask_enter = np.any(trajectories[:, :, 1] <= 2 + eps, axis=-1)
    trajectories = np.nan_to_num(trajectories, posinf=0.)
    mask_exit = np.any(trajectories[:, :, 1] >= r_exit, axis=-1) & np.logical_not(mask_enter)

    trajectories_enter = trajectories[mask_enter]
    trajectories_exit = trajectories[mask_exit]

    size_factor = image_width / render_shape[1]  # Adjust size factor based on image width and render shape

    X = int(image_width)
    Y = int(size_factor * image_height)

    image = image.resize((X, Y), Image.Resampling.LANCZOS)  # Resize image to match render shape

    img_res_x = X / (2 * np.pi)
    img_res_y = Y / np.pi  # Adjust resolution based on aspect ratio

    fov_phi = 2 * np.pi
    fov_theta = 2 * np.pi * Y / X  # Adjust vertical field of view based on aspect ratio

    pixel_space_x = np.arange(0, X)
    pixel_space_y = np.arange(0, Y)
    pixel_grid_x, pixel_grid_y = np.meshgrid(pixel_space_x, pixel_space_y)

    phi_space = pixel_grid_x * fov_phi / (2 * np.pi) / img_res_x
    theta_space = pixel_grid_y * fov_theta / np.pi / img_res_y  # Adjust theta based on aspect ratio

    phi_space = phi_space + (2 * np.pi - fov_phi) / 2 # Adjust phi to match spherical coordinates
    theta_space = theta_space + (np.pi - fov_theta) / 2  # Adjust theta to match spherical coordinates

    phi_exit_all_equator = np.zeros(len(trajectories_exit))

    print(f"Percentage of rays exiting the black hole: {trajectories_exit.shape[0] / (len(trajectories)) * 100}")
    print(f"Percentage of rays entering black hole: {trajectories_enter.shape[0] / (len(trajectories)) * 100}")

    index_closest_exit = np.argmin(jnp.abs(trajectories_exit[:, :, 1] - r_exit), axis=-1).reshape(-1, 1, 1)
    last_r, last_phi = np.take_along_axis(trajectories_exit, index_closest_exit, axis=1).squeeze()[:, 1], np.take_along_axis(trajectories_exit, index_closest_exit, axis=1).squeeze()[:, 3]

    seen_phi = np.pi - phi_samples[mask_exit]

    phi_exit_equator = last_phi + np.arcsin(12 * np.sin(last_phi) / last_r) # Get phi values at the exit points

    phi_exit_all_equator = phi_exit_equator

    seen_phi_interp = interp1d(
        seen_phi,
        phi_exit_all_equator,
        bounds_error=False,
        kind='linear'
    )

    x, y, z = spherical_to_cartesian(theta_space, phi_space)  # Convert spherical coordinates to cartesian

    with np.errstate(all='ignore'):
        beta = -np.arctan(z / y)

    matrix = rotation_matrix(beta)  # Create rotation matrix for each point
    x2 = matrix[0, 0] * x
    y2 = matrix[1, 1] * y + matrix[1, 2] * z
    z2 = matrix[2, 1] * y + matrix[2, 2] * z
    _, phi_enter_angle = cartesian_to_spherical(x2, y2, z2)  # Convert back to spherical coordinates
    phi_enter_angle = np.mod(phi_enter_angle, 2 * np.pi)

    phi_exit_all = np.zeros_like(phi_enter_angle)  # Initialize phi exit array
    phi_exit_all[phi_enter_angle <= np.pi] = seen_phi_interp(phi_enter_angle[phi_enter_angle <= np.pi])
    phi_exit_all[phi_enter_angle > np.pi] = 2 * np.pi - seen_phi_interp(2 * np.pi - phi_enter_angle[phi_enter_angle > np.pi])

    theta = np.ones_like(phi_exit_all) * np.pi / 2
    phi = phi_exit_all
    x3, y3, z3 = spherical_to_cartesian(theta, phi)  # Convert spherical coordinates to cartesian

    matrix = rotation_matrix(-beta)
    x4 = matrix[0, 0] * x3
    y4 = matrix[1, 1] * y3 + matrix[1, 2] * z3
    z4 = matrix[2, 1] * y3 + matrix[2, 2] * z3
    
    theta, phi = cartesian_to_spherical(x4, y4, z4)  # Convert back to spherical coordinates

    phi -= (2 * np.pi - fov_phi) / 2  # Adjust phi to be relative to the equatorial plane
    theta -= (np.pi - fov_theta) / 2  # Adjust theta to

    # print(np.pi - fov_theta)

    phi = np.mod(phi, 2 * np.pi)
    theta = np.mod(theta, np.pi)
    phi[phi == 2 * np.pi] = 0

    pixel_grid_x = phi * (2 * np.pi) / fov_phi * img_res_x 
    pixel_grid_y = theta * np.pi / fov_theta * img_res_y

    pixel_grid_x[np.isnan(pixel_grid_x)] = -1
    pixel_grid_y[np.isnan(pixel_grid_y)] = -1

    pixel_grid_x = pixel_grid_x.astype(int)
    pixel_grid_y = pixel_grid_y.astype(int)

    pixel_grid_x[pixel_grid_x >= X] = -2  # locate pixels outside of the image
    pixel_grid_y[pixel_grid_y >= Y] = -2

    image = np.array(image, dtype=np.uint8)
    render_image = np.array(image, dtype=np.uint8)  # Create a copy of the image for rendering
    render_image = image[pixel_grid_y, pixel_grid_x]  # Sample the image at the exit points
    
    render_image[pixel_grid_x == -1] = [0, 0, 0]

    render_image[pixel_grid_x == -2] = [255, 192, 203] 
    render_image[pixel_grid_y == -2] = [255, 192, 203] 

    return render_image