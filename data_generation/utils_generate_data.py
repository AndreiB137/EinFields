import os
import numpy as np
import itertools
import jax
import jax.numpy as jnp
import jaxlib 
import logging
import yaml
from typing import Callable, Union, Optional, Tuple, Literal, Sequence
from ml_collections import ConfigDict


from einstein_fields.utils.utils_symmetry import take_symmetric_metric, take_symmetric_jacobian, take_symmetric_hessian
from differential_geometry import vdiffgeo, diffgeo
from data_generation.data_lookup_tables import metric_dict, coord_transform_dict
from einstein_fields.utils.config_training import valid_dir, valid_file

def get_kwargs_for_func(args : list[str], args_dict_values : Union[dict, ConfigDict]) -> dict:
    values_list = [args_dict_values.get(arg) for arg in args]

    kwargs = dict(zip(args, values_list))

    return kwargs

def return_metric_fn(metric_name : Literal["Minkowski", "Schwarzschild", "Kerr", "GW"],
                     metric_type: Literal["full", "distortion"],
                     coordinate_system: Literal["cartesian", 
                                                "spherical", 
                                                "kerr_schild_cartesian", 
                                                "oblate_spheroid", 
                                                "boyer_lindquist", 
                                                "eddington_finkelstein",
                                                "ingoing_eddington_finkelstein_non_rotating",
                                                "ingoing_eddington_finkelstein_rotating"], 
                     extra_args_dict : Union[dict, ConfigDict],
                     ) -> Callable[[jax.Array], jax.Array]:
    
    if metric_name not in metric_dict:
        raise ValueError(f"{metric_name} metric is not available or not implemented yet.")
    if metric_type not in ["full", "distortion"]:
        raise ValueError(f"This type of metric is not allowed.")
    if coordinate_system not in metric_dict[metric_name]["coordinate_system"]:
        raise ValueError(f"Unknown coordinate system {coordinate_system} for {metric_name} metric.")
    if metric_name == 'Minkowski':
        if metric_type != "full":
            raise ValueError(f"Distortion metric is not available for {metric_name} metric.")
    
    kwargs = get_kwargs_for_func(metric_dict[metric_name]["coordinate_system"][coordinate_system]["extra_args"], extra_args_dict)

    metric_fn = lambda coords : metric_dict[metric_name]["coordinate_system"][coordinate_system][metric_type](coords, **kwargs)
    
    return metric_fn 

def validate_config(config : Union[dict, ConfigDict]) -> None:
    metric_name = config.get("metric")
    input_coord = config.get("coordinate_system")
    other_coord_systems = config.get("other_coordinate_systems")

    if metric_name not in metric_dict:
        raise ValueError(f"{metric_name} metric is not available or not implemented yet. Please choose one from {list(metric_dict.keys())}.")

    if config.get("data_dir") is None:
        raise ValueError("Please provide a valid data directory in the config file.")
    
    if valid_dir(config.get("data_dir")) is False:
        raise ValueError(f"Provided data directory {config.get('data_dir')} is not valid. Please provide a valid directory.")

    if input_coord not in metric_dict[metric_name]["coordinate_system"]:
        raise ValueError(
            f"'{input_coord}' coordinate system is not available for {metric_name} metric. "
            f"Allowed: {list(metric_dict[metric_name]['coordinate_system'].keys())}"
        )

    if input_coord in other_coord_systems:
        raise ValueError(
            f"'{input_coord}' input coordinate system cannot be in the list of 'other coordinate systems'."
        )

    if set(other_coord_systems).issubset(set(metric_dict[metric_name]['coordinate_system'].keys())) is False:
        raise ValueError(
            f"Some of the other coordinate systems are not available for {metric_name} metric. "
            f"Allowed: {list(metric_dict[metric_name]['coordinate_system'].keys())}"
        )
    
def extract_axes_from_coords(coords: jax.Array, grid_shape: list[int]) -> tuple[jax.Array, ...]:
    """
    Extract the original axes from flattened coordinates array.
    
    Args:
        coords: Flattened coordinates array of shape (N, 4)
        grid_shape: Original grid shape used to create the coordinates
        
    Returns:
        Tuple of arrays representing the original axes
    """
    # Reshape back to original grid shape + coordinate dimension
    if coords.shape[-1] != 4:
        raise NotImplementedError(f"Expected coordinates to have 4 components, got {coords.shape[-1]} instead.")
    coords_reshaped = coords.reshape(tuple(grid_shape) + (4,))
    
    # Extract each axis (take unique values along each dimension)
    
    axes = [
        coords_reshaped[:, 0, 0, 0, 0],
        coords_reshaped[0, :, 0, 0, 1],
        coords_reshaped[0, 0, :, 0, 2],
        coords_reshaped[0, 0, 0, :, 3]
    ]

    return axes


def compute_volume_from_axes(
    axes_grid: Tuple[jax.Array, ...],
    integrating_axes: list[int]) -> jax.Array:
    """
    Computes the coordinate volume element (e.g., d^3x or d^4x) from a spacetime or spatial grid.
    
    Args:
        axes_grid(tuple[jax.Array, jax.Array, jax.Array, jax.Array]): Tuple of coordinate intervals (x0_grid, x1_grid, x2_grid, x3_grid).
        integrating_axes (list[int]): List of axis indices to include in the volume element computation
    
    Returns:
        dV_grid (jax.Array): Flattened array of differential volume elements at each grid point.
    """
    if len(integrating_axes) == 0:
        raise ValueError(f"At least one axis must be provided for measure element computation, got {len(integrating_axes)} axes.")

    dx = []
    for ax in integrating_axes:
        dx.append(jnp.gradient(axes_grid[ax]))

    if len(integrating_axes) == 1:
        dV_grid = dx[0].reshape(-1)
    else:
        mesh_grids = jnp.meshgrid(*dx, indexing='ij')
        dV_grid = jnp.prod(jnp.array(mesh_grids), axis=0).reshape(-1)

    return dV_grid

def squareroot_negative_metric_determinant(metric_grid: jax.Array) -> jax.Array: 
    jac_g_grid = jnp.sqrt(jnp.abs(jnp.linalg.det(metric_grid)))
    return jac_g_grid.reshape(-1)

def compute_invariant_volume(metric_fn: Callable, coords: jax.Array, dV_grid: jax.Array, integrating_axes: list[int]): 
    pairs = list(itertools.product(integrating_axes, repeat=2))
    row_indices = [pair[0] for pair in pairs]
    col_indices = [pair[1] for pair in pairs]
    spacetime = vdiffgeo(diffgeo(metric_fn))
    metric_grid = spacetime.metric(coords)
    jac_g = squareroot_negative_metric_determinant(metric_grid[:, row_indices, col_indices].reshape(-1, len(integrating_axes), len(integrating_axes))) 
    invariant_dV = jac_g * dV_grid
    return invariant_dV.reshape(-1)

def create_coords_and_vol_el(grid_range : list[list[float]], 
                                   grid_shape : list[int],
                                   endpoint: list[bool],
                                   key: jax.random.PRNGKey = jax.random.PRNGKey(0),
                                   compute_volume_element: bool=True) -> tuple[jax.Array, jax.Array, Union[tuple[jax.Array, jax.Array], None], list[int]]:
    
    axes = [jnp.linspace(start, stop, steps, endpoint=endpoint) for (start, stop), steps, endpoint in zip(grid_range, grid_shape, endpoint)]
    X0, X1, X2, X3 = jnp.meshgrid(*axes, indexing='ij') 
    coords_train = jnp.stack((X0, X1, X2, X3), axis=-1).reshape(-1, 4)

    keys = jax.random.split(key, len(grid_range))
    random_axes = [
        jax.random.uniform(k, shape=(n,), minval=l, maxval=m)
        for k, (l, m), n in zip(keys, grid_range, grid_shape)
    ]

    # validation grid
    X = jnp.meshgrid(*random_axes, indexing='ij')
    coords_validation = jnp.stack(X, axis=-1).reshape(-1, len(random_axes))

    if compute_volume_element:
        integrating_axes = []
        for i in range(len(axes)):
            if len(axes[i]) > 1:
                integrating_axes.append(i)
        dV_grid_train = jnp.abs(compute_volume_from_axes(axes, integrating_axes))
        dV_grid_val = jnp.abs(compute_volume_from_axes(random_axes, integrating_axes))
        dV_grid = (dV_grid_train, dV_grid_val)
    else:
        dV_grid_train = None
        integrating_axes = None
        dV_grid_val = None
        dV_grid = None

    del X0, X1, X2, X3, axes
    del X, random_axes
    
    return coords_train, coords_validation, dV_grid, integrating_axes
    

def save_quantities(data_dict : dict, save_dir: str) -> None: 
    for k, v in data_dict.items():
        jnp.save(os.path.join(save_dir, k), v)

def save_coords_vol_el(coords_train: jax.Array, 
                           coords_validation: jax.Array, 
                           inv_volume_measure: tuple[jax.Array, jax.Array],
                           save_dir: str) -> None:
    """Saves the coordinates to the specified directory."""
    coords_train_path = os.path.join(save_dir, "coords_train.npy")
    coords_validation_path = os.path.join(save_dir, "coords_validation.npy")
    if inv_volume_measure is not None:
        inv_volume_measure_train_path = os.path.join(save_dir, "inv_volume_measure_train.npy")
        inv_volume_measure_validation_path = os.path.join(save_dir, "inv_volume_measure_validation.npy")
    
    jnp.save(coords_train_path, coords_train)
    jnp.save(coords_validation_path, coords_validation)
    if inv_volume_measure is not None:
        jnp.save(inv_volume_measure_train_path, inv_volume_measure[0])
        jnp.save(inv_volume_measure_validation_path, inv_volume_measure[1])
    
def create_data_tensors(metric_fn : Callable, 
                   coords_train : jax.Array, 
                   coords_validation: jax.Array,
                   store_GR_tensors : bool = False) -> Tuple[dict, dict]:

    spacetime = vdiffgeo(diffgeo(metric_fn))
    # logging.info(f"Computing the metric")
    metric      = spacetime.metric(coords_train)
    # logging.info(f"Computing the validation metric")
    metric_val      = spacetime.metric(coords_validation)
    # logging.info(f"Computing the Jacobian")
    metric_jacobian = spacetime.metric_jacobian(coords_train)
    # logging.info(f"Computing the validation jacobian")
    metric_jacobian_val   = spacetime.metric_jacobian(coords_validation)
    # logging.info(f"Computing the Hessian")
    metric_hessian  = spacetime.metric_hessian(coords_train)
    # logging.info(f"Computing the validation Hessian")
    metric_hessian_val   = spacetime.metric_hessian(coords_validation)

    if store_GR_tensors:
        # logging.info(f"Computing the Riemann tensor")
        riemann_tensor  = spacetime.riemann_tensor(coords_train)
        # logging.info(f"Computing the validation Riemann tensor")
        riemann_tensor_val  = spacetime.riemann_tensor(coords_validation)
        # logging.info(f"Computing the Kretschmann")
        kretschmann     = spacetime.kretschmann_invariant_static(metric.reshape(-1, 4, 4), riemann_tensor.reshape(-1, 4, 4, 4, 4))
        # logging.info(f"Computing the validation Kretschmann")
        kretschmann_val     = spacetime.kretschmann_invariant_static(metric_val.reshape(-1, 4, 4), riemann_tensor_val.reshape(-1, 4, 4, 4, 4))
        # logging.info(f"Computing the Ricci scalar")
        ricci_scalar    = spacetime.ricci_scalar_static(metric.reshape(-1, 4, 4), riemann_tensor.reshape(-1, 4, 4, 4, 4))
        # logging.info(f"Computing the validation Ricci scalar")
        ricci_scalar_val = spacetime.ricci_scalar_static(metric_val.reshape(-1, 4, 4), riemann_tensor_val.reshape(-1, 4, 4, 4, 4))
    

    train_data = {
            "metric": metric,
            "jacobian": metric_jacobian,
            "hessian": metric_hessian,
        }
    
    validation_data = {
        "metric": metric_val,
        "jacobian": metric_jacobian_val,
        "hessian": metric_hessian_val,
    }

    if store_GR_tensors is False:
        return train_data, validation_data

    train_data.update({
        "kretschmann": kretschmann,
        "riemann_tensor": riemann_tensor,
    })

    validation_data.update({
        "kretschmann": kretschmann_val,
        "riemann_tensor": riemann_tensor_val,
    })

    return train_data, validation_data

def store_train_and_val_data(
        metric_fn: Callable, 
        coords_train : jax.Array,
        coords_validation: jax.Array,
        save_dir: str, 
        store_GR_tensors: bool = False,
        store_symmetric: bool = False
        )-> None: 
        
    train_data, validation_data = create_data_tensors(metric_fn, 
                                                 coords_train, 
                                                 coords_validation,
                                                 store_GR_tensors=store_GR_tensors,
    )

    if store_symmetric:
        for k, _ in train_data.items():
            if k == 'metric':
                vmapped_take_symmetric_metric = jax.vmap(take_symmetric_metric)
                train_data[k] = vmapped_take_symmetric_metric(train_data[k])
                validation_data[k] = vmapped_take_symmetric_metric(validation_data[k])
            elif k == 'jacobian':
                vmapped_take_symmetric_jacobian = jax.vmap(take_symmetric_jacobian)
                train_data[k] = vmapped_take_symmetric_jacobian(train_data[k])
                validation_data[k] = vmapped_take_symmetric_jacobian(validation_data[k])
            elif k == 'hessian':
                vmapped_take_symmetric_hessian = jax.vmap(take_symmetric_hessian)
                train_data[k] = vmapped_take_symmetric_hessian(train_data[k])
                validation_data[k] = vmapped_take_symmetric_hessian(validation_data[k])
    else:
        for k, _ in train_data.items():
            if k != 'inv_volume_measure':
                train_data[k] = train_data[k].reshape(train_data[k].shape[0], -1)
                validation_data[k] = validation_data[k].reshape(validation_data[k].shape[0], -1)
    
    logging.info(f"Storing the training data.")
    train_dir = os.path.join(save_dir, "training")
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    save_quantities(train_data, train_dir)
    logging.info(f"Storing the validation data.")
    val_dir = os.path.join(save_dir, "validation")
    if not os.path.exists(val_dir):
        os.makedirs(val_dir)
    save_quantities(validation_data, val_dir)
    
    del train_data, validation_data
    
    return 


def _create_scale_directories(base_dir: str, transform_list: tuple) -> list[str]:
    """Create directories for scaled transformations and return the list of paths."""
    aux_dirs = []
    for i in range(len(transform_list)):
        dir_name = f"scale{i+1}"
        aux_dir = os.path.join(base_dir, dir_name)
        aux_dirs.append(aux_dir)
        if not os.path.exists(aux_dir):
            os.makedirs(aux_dir)
    return aux_dirs

def store_full_distortion_in_file(
    metric_fn: Callable,
    coords_train: jax.Array,
    coords_validation: jax.Array,
    save_dir: str,
    store_GR_tensors: bool = False,
    store_symmetric: bool = False,
    distortion_metric_fn: Optional[Callable[[jax.Array], jax.Array]] = None,
    dV_grid: Optional[tuple[jax.Array, jax.Array]] = None,
    integrating_axes: Optional[list[int]] = None
    ) -> None:

    full_sub_dir = os.path.join(save_dir, "full_flatten")
    if not os.path.exists(full_sub_dir):
        os.makedirs(full_sub_dir)  
    logging.info(f"Storing the full flatten tensors in {full_sub_dir}.")
    if dV_grid is not None and integrating_axes is not None:
        inv_volume_train = compute_invariant_volume(metric_fn, coords_train, dV_grid[0], integrating_axes)
        inv_volume_val = compute_invariant_volume(metric_fn, coords_validation, dV_grid[1], integrating_axes)
        inv_volume = (inv_volume_train, inv_volume_val)
    else:
        inv_volume = None
    save_coords_vol_el(coords_train, coords_validation, inv_volume, save_dir)
    store_train_and_val_data(metric_fn,
                            coords_train,
                            coords_validation,
                            save_dir=full_sub_dir,
                            store_GR_tensors=store_GR_tensors,
                            store_symmetric=False
    )

    if store_symmetric:

        sym_full_sub_dir = os.path.join(full_sub_dir, "symmetric")
        logging.info(f"Storing the symmetric part of the full flatten tensors in {sym_full_sub_dir}.")
        if not os.path.exists(sym_full_sub_dir):
            os.makedirs(sym_full_sub_dir)

        store_train_and_val_data(metric_fn,
                                coords_train,
                                coords_validation,
                                save_dir=sym_full_sub_dir,
                                store_GR_tensors=False,
                                store_symmetric=True
        )

    if distortion_metric_fn is not None:
        logging.info(f"Storing distortion metric data.")
        dist_dir = os.path.join(save_dir, "distortion")
        if not os.path.exists(dist_dir):
            os.makedirs(dist_dir)
        store_train_and_val_data(distortion_metric_fn,
                                coords_train, 
                                coords_validation, 
                                save_dir=dist_dir,
                                store_GR_tensors=False
        )

        if store_symmetric:
            sym_dist_dir = os.path.join(dist_dir, "symmetric")
            logging.info(f"Storing the symmetric part of the distortion metric data in {sym_dist_dir}.")
            if not os.path.exists(sym_dist_dir):
                os.makedirs(sym_dist_dir)
            store_train_and_val_data(distortion_metric_fn,
                                    coords_train, 
                                    coords_validation, 
                                    save_dir=sym_dist_dir,
                                    store_GR_tensors=False
            )
    

def loop_over_tensor_storing(
    metric_fn: Callable[[jax.Array], jax.Array], 
    coords_train : jax.Array,
    coords_validation: jax.Array,
    config: Union[dict, ConfigDict],
    save_dir: str,
    dV_grid: Optional[tuple[jax.Array, jax.Array]] = None,
    integrating_axes: Optional[list[int]] = None,
    transform_list: tuple[Callable[[jax.Array], jax.Array], ...] = None
    ) -> None: 

    sub_dir = os.path.join(save_dir, config.get('coordinate_system'))
    if not os.path.exists(sub_dir): 
        os.makedirs(sub_dir) 

    no_scale_dir = os.path.join(sub_dir, "no_scale")

    if config.get('store_quantities').get('store_distortion'):
        distortion_metric_fn = return_metric_fn(
            config.get('metric'),
            "distortion",
            config.get('coordinate_system'),
            config.get('metric_args')
        )
    else:
        distortion_metric_fn = None

    store_full_distortion_in_file(metric_fn=metric_fn,
                                  coords_train=coords_train,
                                  coords_validation=coords_validation,
                                  save_dir=no_scale_dir,
                                  store_GR_tensors=config.get('store_quantities').get('store_GR_tensors'),
                                  store_symmetric=config.get('store_quantities').get('store_symmetric'),  
                                  distortion_metric_fn=distortion_metric_fn,
                                  dV_grid=dV_grid,
                                  integrating_axes=integrating_axes)

    logging.info(f"Checking if the scaled data is to be stored.")
    if transform_list is not None:
        scaled_dir = sub_dir
        if not os.path.exists(scaled_dir):
            os.makedirs(scaled_dir)
        
        aux_dirs = _create_scale_directories(scaled_dir, transform_list)
                
        ### frame transformed tensors stored in corresponding aux dirs 
        for dir, pair_F in zip(aux_dirs, transform_list):
            logging.info(f"Storing the scaled tensors in {dir}.")
            params = pair_F[1]
            scaled_coords_fn, scaled_full_metric_fn = frame_transformed_quantities_functional(pair_F[0], metric_fn)
            if distortion_metric_fn is not None:
               scaled_distortion_metric_fn = frame_transformed_quantities_functional(pair_F[0], distortion_metric_fn)[1]
            else:
                scaled_distortion_metric_fn = None
            scaled_coords = jax.vmap(scaled_coords_fn)(coords_train)
            scaled_coords_val = jax.vmap(scaled_coords_fn)(coords_validation)
            store_full_distortion_in_file(metric_fn=scaled_full_metric_fn,
                                          coords_train=scaled_coords,
                                          coords_validation=scaled_coords_val,
                                          save_dir=dir,
                                          store_GR_tensors=False,
                                          store_symmetric=config.get('store_quantities').get('store_symmetric'),
                                          distortion_metric_fn=scaled_distortion_metric_fn,
                                          dV_grid=dV_grid,
                                          integrating_axes=integrating_axes
            )
            logging.info(f"Storing the parameters of the scaling transformation.")
            params_path = os.path.join(dir, "params_scale.yml")
            with open(params_path, 'w') as f:
                yaml.dump(params, f)
    else:
        logging.info(f"No scale transformations are provided, hence no scaled quantities are stored.")

    logging.info(f"All arrays have been stored on disk.")

def store_other_coord_systems_quantities(config: Union[dict, ConfigDict], 
                                         coords_train : jax.Array,
                                         coords_validation: jax.Array, 
                                         save_dir: str,
                                         recompute_volume_elements: bool = True) -> None: 

    logging.info(f"Starting the process for storing other coordinate systems: {config.get('other_coordinate_systems')}")
    transformed_metrics_fn = {} 
    for k in config["other_coordinate_systems"]:  
        try:
            full_metric_fn = return_metric_fn(config.get('metric'), 'full', k, config.get('metric_args'))
        except ValueError as e:
            logging.info(f"Coordinate transformation for {k} is not available or not implemented yet, skipping this coordinate system.")
            continue

        distortion_metric_fn = return_metric_fn(config.get('metric'), 'distortion', k, config.get('metric_args')) if config.get('store_quantities').get('store_distortion') else None

        kwargs = get_kwargs_for_func(coord_transform_dict[config.get("coordinate_system")][k]["extra_args"], config.get('metric_args'))
        coord_fns = lambda coords : coord_transform_dict[config.get('coordinate_system')][k]["transform"](coords, **kwargs)
        
        vmapped_coord_fns = jax.vmap(coord_fns)
        coords_transformed = vmapped_coord_fns(coords_train)
        coords_val_transformed = vmapped_coord_fns(coords_validation)

        other_coord_dir = os.path.join(save_dir, k)
        other_coord_dir = os.path.join(other_coord_dir, "no_scale")
        if not os.path.exists(other_coord_dir): 
            os.makedirs(other_coord_dir)

        logging.info(f"Storing the quantities for {k} coordinate system at {other_coord_dir}.")

        store_full_distortion_in_file(metric_fn=full_metric_fn,
                                        coords_train=coords_transformed,
                                        coords_validation=coords_val_transformed,
                                        save_dir=other_coord_dir,
                                        store_GR_tensors=config.get('store_quantities').get('store_GR_tensors'),
                                        store_symmetric=config.get('store_quantities').get('store_symmetric'),
                                        distortion_metric_fn=distortion_metric_fn,
                                        dV_grid=None,  # No volume element for transformed coordinates
                                        integrating_axes=None  # No volume element for transformed coordinates
        )
        
        print(f"Done with {k}.")

        del coords_transformed, coords_val_transformed, other_coord_dir

def scaled_func(k0 : float, k1 : float, k2 : float, k3 : float) -> Callable: 
    """
    Returns a list of functions that scale the input coordinates by k0, k1, k2, k3 respectively.
    """
    f0 = lambda x0: k0 * x0
    f1 = lambda x1: k1 * x1
    f2 = lambda x2: k2 * x2
    f3 = lambda x3: k3 * x3

    params_dict = {
        "k0" : k0,
        "k1" : k1,
        "k2" : k2,
        "k3" : k3,
    }

    return apply_scaling_transformations((f0, f1, f2, f3)), params_dict

def apply_scaling_transformations(
    F: Sequence[Callable[[jax.Array], jax.Array]]
) -> Optional[jaxlib.xla_extension.PjitFunction]:
    """
    Applies a tuple or list of component-wise transformations `F = (f₀, f₁, ..., fₙ₋₁)` 
    to the coordinates returned by a map `X(p)`.

    This returns a new function `X̃(p) = (f₀(X₀(p)), f₁(X₁(p)), ..., fₙ₋₁(Xₙ₋₁(p)))`.

    Parameters
    ----------
    F : Sequence[Callable[[jax.Array], jax.Array]]
        A sequence of `n` scalar functions. Each `fᵢ` will be applied to the `i`-th 
        coordinate of the result of `X(p)`.

    Returns
    -------
    Optional[jaxlib.xla_extension.PjitFunction]
        A function `X̃(p)` which returns the transformed coordinates:
        [f₀(X₀(p)), f₁(X₁(p)), ..., fₙ₋₁(Xₙ₋₁(p))].

    Example
    -------
    >>> def X(p): return p  # identity chart
    >>> F = [jnp.sin, jnp.cos, jnp.tan, lambda x: x**2]
    >>> X_tilde = apply_coordinate_transforms(X, F)
    >>> X_tilde(jnp.array([1.0, 2.0, 3.0, 4.0]))
    Array([sin(1.0), cos(2.0), tan(3.0), 16.0], dtype=float32)

    Notes
    -----
    - This function assumes `X(p)` returns a vector of shape `(n,)` and `F` is of length `n`.
    - The returned function is JIT-compatible and differentiable with respect to `p`.
    """
    @jax.jit
    def transformed_X(p: jax.Array) -> jax.Array:
        """
        Parameters
        ----------
        x : jax.Array
            coordinate `p ∈ ℝ^n` to another point in ℝ^n. This is typically the base coordinate chart.
        Returns
        -------
        jax.Array 
            Transformed coordinate p^' obatined by action of an operator f_i(p_i) 
        """
        return jnp.array([f(xi) for f, xi in zip(F, p)])
    
    return transformed_X

def frame_transformed_quantities_functional(
        transformed_frame_fn: Callable[[jax.Array], jax.Array], 
        metric_fn: Callable[[jax.Array], jax.Array]
        ) -> tuple[jax.Array, jax.Array]:
    transformed_metric = lambda x: diffgeo.tensor_frame_transformation(x, transformed_frame_fn, metric_fn, x, [0, 0]) 
    return (transformed_frame_fn, transformed_metric)


if __name__=="__main__":
    # %%%%%%%%%%%%%%%%%% preferable device for vectorizing over is cpus due to OOM errors of Riemann curvature tensors %%%%%%%%%%%%%%%%%% 
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import time
    np.set_printoptions(suppress=True)