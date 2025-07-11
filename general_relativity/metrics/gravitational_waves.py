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

import numpy as np
import jax 
import jax.numpy as jnp

from general_relativity.metrics.minkowski import minkowski_metric

def gravitational_waves_metric_distortion(
    cart_coords: jax.Array, 
    polarization_amplitudes: tuple[float, float], 
    omega: float
) -> jax.Array: 
    """
    Returns the metric perturbation h_{μν}^{TT} in the transverse-traceless (TT) gauge 
    for a plane gravitational wave propagating in the +z direction.

    The function computes the linearized metric distortion h_{μν}^{TT}, 
    representing a gravitational wave in the TT gauge with arbitrary amplitudes 
    for the two physical polarization states: 'plus' (A₊) and 'cross' (A_×). 
    The only nonzero components are transverse to the direction of propagation 
    (z-axis), and the perturbation is traceless and symmetric.

    Parameters
    ----------
    cart_coords : jax.Array
        Cartesian coordinates (t, x, y, z) array of shape (4,).
    polarization_amplitudes : tuple[float, float]
        Tuple (h₊, h_×) specifying the amplitudes for the 'plus' and 'cross' 
        polarizations of the gravitational wave.
    omega : float
        Angular frequency ω of the gravitational wave.

    Returns
    -------
    jax.Array
        A (4, 4) JAX array h_{μν} giving the symmetric metric perturbation 
        in the TT gauge at the given spacetime point.

    Notes
    -----
    The TT gauge metric perturbation for a wave traveling in the +z direction is:
        h_{μν}(t, x, y, z) = h₊ ε_{μν}^{(+)} cos(ω (t - z)) + h_× ε_{μν}^{(×)} cos(ω (t - z))
    where the polarization tensors are:
        ε_{μν}^{(+)} =
            [[0, 0, 0, 0],
             [0, 1, 0, 0],
             [0, 0, -1, 0],
             [0, 0, 0, 0]]
        ε_{μν}^{(×)} =
            [[0, 0, 0, 0],
             [0, 0, 1, 0],
             [0, 1, 0, 0],
             [0, 0, 0, 0]]
    The perturbation is transverse (h_{μ3} = 0 for μ ≠ 3), traceless (h^μ_μ = 0), and symmetric.

    Example
    -------
    >>> coords = jnp.array([0.0, 1.0, 0.0, 0.5])  # (t, x, y, z)
    >>> h = linearized_gravity_metric_distortion(coords, (1e-6, 0.5e-6), omega=2.0)
    >>> h.shape
    (4, 4)
    """
    t, x, y, z = cart_coords  
    h_plus, h_cross = polarization_amplitudes
    # If LIGO experiments to be satisfied, then scipy.constants.G, scipy.constants.c to be inserted
    h_ij_TT = jnp.cos(omega*(t-z))*jnp.array([[0.0, 0.0, 0.0, 0.0], [0.0, h_plus, h_cross, 0.0], [0.0, h_cross, -h_plus, 0.0], [0.0, 0.0, 0.0, 0.0]])
    return h_ij_TT 

def gravitational_waves_metric( 
    cart_coords: jax.Array, 
    polarization_amplitudes: tuple[float, float], 
    omega: float
) -> jax.Array: 
    """"
    Returns the linear approximation g_{μν}  = η_{μν} + h_{μν}^{TT} + O(h^2) with the pertubation
    being transverse-traceless (TT) gauge  for a plane gravitational wave propagating in the +z direction.

    The function computes the linearized metric around a fixed background (in this case Minkowski), 
    using the pertubation metric 

    Parameters
    ----------
    cart_coords : jax.Array
        Cartesian coordinates (t, x, y, z) array of shape (4,).
    polarization_amplitudes : tuple[float, float]
        Tuple (h₊, h_×) specifying the amplitudes for the 'plus' and 'cross' 
        polarizations of the gravitational wave.
    omega : float
        Angular frequency ω of the gravitational wave.

    Returns
    -------
    jax.Array
        A (4, 4) JAX array g_{μν} = η_{μν} + h_{μν}^{TT} + O(h^2) at the given spacetime point.

     Example
    -------
    >>> coords = jnp.array([0.0, 1.0, 0.0, 1.5])  # (t, x, y, z)
    >>> h = linearized_gravity_metric(coords, (1e-6, 0.5e-6), omega=2.0)
    >>> h.shape
    (4, 4)
    """
    return minkowski_metric(cart_coords) + gravitational_waves_metric_distortion(cart_coords, polarization_amplitudes, omega)

if __name__ == '__main__': 

    jax.config.update("jax_enable_x64", True)
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import time
    np.set_printoptions(suppress=True) 



