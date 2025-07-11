""
import os
import jax
import shutil
import logging
import yaml
jax.config.update('jax_enable_x64', True)
jax.config.update("jax_default_matmul_precision", "highest")

# os.environ["JAX_PLATFORMS"] = "gpu"
from data_generation.utils_generate_data import (validate_config, create_coords_and_vol_el, loop_over_tensor_storing, return_metric_fn, store_other_coord_systems_quantities)

if __name__ == '__main__':
    M = 1.0
    a = 0.7
    other_coordinate_systems = [] # "kerr_schild"  #["cartesian", "eddington_finkelstein"] 
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% main data generation part starts here %%%%%%%%%%%%%%%%%%%%%%%%%%
    config = {
        "metric": "Kerr",
        "metric_args" : {
            "M": M,
            "a": a,
            "Q": 0.0,
            "G": 1.0,
            "c": 1.0,
        },
        "coordinate_system":"kerr_schild_cartesian",
        "other_coordinate_systems": other_coordinate_systems, 
        "grid_shape": [1, 10, 10, 10],
        "grid_range": [
            [0.0, 0.0],
            [-3., 3.],
            [-3., 3.], # always choose (0, \pi) since, zenith angle is always chosen as this
            [0.1, 3.]], # always choose [0, 2\pi) for azimuthal angle from now on, since the angles, modulo phase has been rectified
        "endpoint": [True, True, True, True],
        "store_quantities" : {
            "store_symmetric": True,
            "store_distortion": True,
            "store_GR_tensors": False,
        },
        "compute_volume_element": True,
        "recompute_volume_elements": True, # Not implemented yet
        "problem": "test_script0",
        "data_dir": "/Users/andrei/Documents/dataa/EinFields/data"} 
    
    validate_config(config)
    logging.basicConfig(level=logging.INFO, encoding='utf-8', force=True)

    metric_fn = return_metric_fn(config.get('metric'),
                                 "full",
                                 config.get('coordinate_system'),
                                 config.get('metric_args'))
    
    coords_train, coords_validation, dV_grid, integrating_axes = create_coords_and_vol_el(
        grid_range=config.get('grid_range'),
        grid_shape=config.get('grid_shape'),
        endpoint=config.get('endpoint'),
        compute_volume_element=config.get('compute_volume_element'))

    save_dir = os.path.join(config["data_dir"], config["problem"])
    if os.path.exists(save_dir):
        logging.info(f"Directory {save_dir} already exists. Removing it.")
        shutil.rmtree(save_dir)
    
    os.makedirs(save_dir)

    cfg_file = os.path.join(save_dir, "config.yml")
    
    logging.info(f"Storing the config file at {cfg_file}.")
    with open(cfg_file, 'w') as f:
        yaml.dump(config, f)

    loop_over_tensor_storing(metric_fn,
                             coords_train,
                             coords_validation,
                             config,
                             save_dir,
                             dV_grid,
                             integrating_axes,
                             transform_list=None,)

    ## TODO: Recalculation of volume elements is not implementey yet.
    ## For example, if integrating only the spatial volume,
    ## in a new coordinate system, you might need all 4 components to describe the volume measure,
    ## so the integration axes might change.
    if len(other_coordinate_systems) > 0:
        store_other_coord_systems_quantities(config, save_dir, coords_train, coords_validation, dV_grid, integrating_axes)