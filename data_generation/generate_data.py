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
os.environ["JAX_PLATFORMS"] = "cpu"
import jax
import jax.numpy as jnp
import shutil
import logging
import yaml
jax.config.update('jax_enable_x64', True)
jax.config.update("jax_default_matmul_precision", "highest")

from data_generation.utils_generate_data import (validate_config, 
                                                 create_coords_and_vol_el, 
                                                 loop_over_tensor_storing, 
                                                 return_metric_fn, 
                                                 store_other_coord_systems_quantities)

if __name__ == '__main__':
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% main data generation part starts here %%%%%%%%%%%%%%%%%%%%%%%%%%

    config_path = "configs/schwarzschild_spherical.yml"

    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
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
        yaml.dump(config, f, sort_keys=False)

    # Example how to use transform_list
    # from data_generation.utils_generate_data import scaled_func
    # transform_list = [scaled_func(1.0, 2.0, 3.0, 4.0), 
    #                   scaled_func(10.2, 1.5, 9.23, 4.3)]
    # You can have as many transform as you want, but the number of arguments needs to match
    # the number of coordinates in the metric function.

    loop_over_tensor_storing(metric_fn,
                             coords_train,
                             coords_validation,
                             config,
                             save_dir,
                             dV_grid,
                             integrating_axes,
                             transform_list=None,)

    # If volume elements is set to True, the same elements will be copied to the other coordinate systems,
    # since all coordinate transformations involved are diffeomorphisms.
    if len(config.get('other_coordinate_systems', [])) > 0:
        store_other_coord_systems_quantities(config=config, 
                                             coords_train=coords_train, 
                                             coords_validation=coords_validation,
                                             save_dir=save_dir)