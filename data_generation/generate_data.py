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

import os
import jax
import jax.numpy as jnp
import shutil
import logging
import yaml
jax.config.update('jax_enable_x64', True)
jax.config.update("jax_default_matmul_precision", "highest")

os.environ["JAX_PLATFORMS"] = "gpu"
from data_generation.utils_generate_data import (validate_config, create_coords_and_vol_el, loop_over_tensor_storing, return_metric_fn, store_other_coord_systems_quantities)

if __name__ == '__main__':
    M = 1.0
    a = 0.0
    other_coordinate_systems = [] # "kerr_schild"  #["cartesian", "eddington_finkelstein"] 
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% main data generation part starts here %%%%%%%%%%%%%%%%%%%%%%%%%%
    config = {
        "metric": "Schwarzschild",
        "metric_args" : {
            "M": M,
            "a": a,
            "Q": 0.0,  
            "G": 1.0,
            "c": 1.0,
        },
        # if "metric": "GW" comment the above metric_args and uncomment te below metric_args
        # "metric_args" : {
        #     "polarization_amplitudes": (1.e-6, 1.e-6), 
        #     "omega": 2.0},
        "coordinate_system":"spherical",
        "other_coordinate_systems": other_coordinate_systems, 
        "grid_shape": [1,128,128,128],
        "grid_range": [
            [0.0, 0.0],
            [5.0, 140.0],
            [1.e-2, jnp.pi-1.e-2], # always choose (0, \pi) since, zenith angle is always chosen as this
            [0.0, 2.0*jnp.pi]], # always choose [0, 2\pi) for azimuthal angle from now on, since the angles, modulo phase has been rectified
        "endpoint": [True, True, True, False],
        "store_quantities" : {
            "store_symmetric": True,
            "store_distortion": True,
            "store_GR_tensors": False,
        },
        "compute_volume_element": True,
        "recompute_volume_elements": True, # Not implemented yet
        "problem": "geodesic_perihilion",
        "data_dir": ".."} 
    
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