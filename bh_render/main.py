
import matplotlib.pyplot as plt 
import jax.numpy as jnp
import numpy as np
from bh_render.utils import (generate_phi_samples, 
                             generate_rendering_schwarzschild, 
                             ray_trace_init_cond,
                             create_initial_condition_equatorial_observer)
from general_relativity import schwarzschild_metric_spherical

if __name__ == "__main__":
    phi_samples = generate_phi_samples(sparse_samples=512)
    schwarzs_fn = lambda coords: schwarzschild_metric_spherical(coords, M=1.0)
    observer_pos = jnp.array([0.0, 12.0, jnp.pi/2, 0.0])
    init_cond = create_initial_condition_equatorial_observer(phi_samples, observer_coords=observer_pos)

    trajectories = ray_trace_init_cond(init_cond,
                                       schwarzs_fn,
                                       batch_size=10000)
    
    render_shape = (2000, 4000) # Approximately 4K resolution

    background_img_path = "background_img/milki_way.jpg"  # Replace with actual path

    render_image = generate_rendering_schwarzschild(trajectories,
                                     render_shape,
                                     background_img_path,
                                     phi_samples,
                                     r_exit=200.0)
    
    save_dir = "render_bh_nef_2000x4000.jpg"  # Replace with actual save path
    
    plt.imsave(save_dir, render_image)