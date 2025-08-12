# Black Hole rendering

First of all, I want to give credits to this github repo from which I used the visualization code: https://github.com/Python-simulation/Black-hole-simulation-using-python.

## Introduction

One imagines itself standing somewhere in space and having a camera in hand. Depending on the orientation, where is looking, there must have been a ray of light collected by the camera sensor. Then, solving numerically the geodesic equation with the initial velocity as the unit vector given by the camera orientation, one gets the location from where it came from.

The background image, usually with a ratio width/height of two represents the projection of the surface of a sphere of unit radius onto a 2D plane. Meaning, the two coordinates used to describe this surface $(\phi, \theta)$ are the horizontal and vertical lines on the image. In this perspective, when looking at the left, right, top or down ends of the image, in reality this is somewhere in the back of our view, not in the front as it might seem. 

Hence, the task is to assign for each orientation coordinates of the camera, the orientation where the geodesic touches the background. This exit is set at a radius far away from the black hole position. Rays that are trapped are just rendered black. Alternatively, only rays that are known to escape are computed and the interior region is rendered as black.

## Implementation

There are multiple ways to achieve this, two that I can mention:

- Sample some initial velocity orientations in the range of $\phi \in [-\pi,\pi]$ and $\theta \in [-\pi/2,\pi/2]$ and compute the geodesics.
- Schwarzschild has lots of symmetries due to its simplicity. Most importantly, one can compute only geodesics at the equator, then each orientation velocity not at the equator is in another great circle plane which can be rotated to the equator. In other words, all rays can be found by knowing what happens in the equatorial plane. Moreover, rays starting at orientations in the range $[\pi, 2\pi]$ or $[-\pi, 0]$ will touch the background at $2\pi - x$, where $x$ is the angle where geodesics in $[0, \pi]$ touch the background.

In cases such as Kerr (rotating) black holes, the frame-dragging effect will destroy these symmetries of Schwarzschild.

The crucial aspect is the way to generate the initial velocities. The basis vectors of the geometry (specifically for Schwarzschild) are streched, so assigning for example $v_{\theta}=3$, this vector will be very large due to the $r^{2}$ dependence. A much smaller value is needed. One way to do this is normalize the basis, create the vector components that you would have in a 4D orthonormal frame, then revert back to the original basis. This whole thing is captured by what in general relativity is called a tetrad. For more explanations, you can read chapter `10` in [examples](../example_notebooks/general_relativity/schwarzschild_kerr_ex.ipynb). 

## Misc

The Schwarzschild symmetries do help in faster results due to the reduced number of geodesics that need to be computed. But, one can also generate a small number of geodesics sufficient for a linear/trilinear interpolation method to work. Thats because the whole ray tracing problem is just to identify the map from every camera orientation to the location (the pixel) where the geodesic touches the background image. 

Even with a gpu and `jax.jit` enhanced performance computing geodesics can become relatively expensive for a few hundred thousands to millions of rays. One bottleneck of Jax is the backend of `jnp.linalg.inv` uses LU decomposition which is slow for small matrices. Replacing this with an analytic computation of the inverse (something like here: https://github.com/jax-ml/jax/issues/11321), on CPU it can reduce the computation time up 100x in some cases. On GPU, this factor can be up to 10x. Also, `jnp.einsum` is not well optimized on CPU for the small contractions in GR involving only 4 indices. In most cases, manual indexing/slicing can reduce computation time up to 10x in some cases. 

In the rendered image done by EinFields, you can see a white circle which for the analytic metric doesn't exist. This is because at the event horizon the metric blows up, so the network is just a finite approximation of that. Therefore, closer trajectories might escape the horizon just because in the trained metric there is no singularity there.




 
