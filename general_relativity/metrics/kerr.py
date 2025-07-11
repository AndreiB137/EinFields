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

from general_relativity.coordinate_transformations.coord_transform import cartesian_to_oblate_spheroid
from general_relativity.metrics.minkowski import (minkowski_metric, minkowski_metric_oblate_spheroidal, minkowski_metric_eddington_finkelstein_rotating)

def kerr_metric_boyer_lindquist(
    coords: jax.Array,
    M: float,
    a: float,
) -> jax.Array:
    """
    Computes the Kerr metric in Boyer-Lindquist coordinates.

    Returns the spacetime metric tensor  g_{μν}  for a rotating,
    uncharged (or charged, if implemented) black hole using the Kerr geometry
    in Boyer-Lindquist (BL) coordinates `(t, r, 𝜗, φ)`.

    Parameters
    ----------
    coords : jax.Array
        The spacetime coordinates `(t, r, 𝜗, φ)` in Boyer-Lindquist form (oblate spheroidal geometry).
    M : float
        Mass of the black hole.
    a : float
        Spin parameter of the black hole (angular momentum per unit mass).
    
    Returns
    -------
    jax.Array
        A `(4, 4)` JAX array representing the Kerr metric tensor in Boyer-Lindquist coordinates.

    Notes
    -----
    - The Kerr solution reduces to Schwarzschild when `a = 0`.
    - Frequently used in astrophysical black hole and relativistic orbit simulations.
    """
    t, r, theta, phi = coords 
    sigma = r**2 + a**2*jnp.cos(theta)**2 
    delta = r**2 - 2*M*r + a**2
    gij = jnp.array([[-(1-2*M*r/sigma), 0, 0, -2*M*a*r*jnp.sin(theta)**2/sigma], [0, sigma/delta, 0, 0], [0, 0, sigma, 0], [-2*a*jnp.sin(theta)**2*M*r/sigma, 0, 0, (r**2 + a**2 + 2*M*r*a**2*jnp.sin(theta)**2/sigma)*jnp.sin(theta)**2]])
    return gij 

def kerr_metric_boyer_lindquist_distortion(
    oblate_coords: jax.Array,
    M: float,
    a: float,
) -> jax.Array:
    """
    Computes the deviation of the Kerr metric in Boyer-Lindquist coordinates from flat spacetime.

    This function returns the distortion tensor: Δ_{μν} (BL) = g_{μν} (BL) - η_{μν} (BL)
    where g_{μν} is the Kerr metric in Boyer-Lindquist coordinates.

    Parameters
    ----------
    oblate_coords : jax.Array
        The spacetime coordinates `(t, r, 𝜗, φ)` in Boyer-Lindquist form.
    M : float
        Mass of the black hole.
    a : float
        Spin parameter of the black hole.
    Returns
    -------
    jax.Array
        A `(4, 4)` JAX array representing the metric distortion tensor in Boyer-Lindquist coordinates.

    Notes
    -----
    - Useful for perturbative calculations and comparing with Minkowski spacetime.
    - The distortion tensor vanishes when `M = 0` and `a = 0`.
    """ 
    kerr_bl = kerr_metric_boyer_lindquist(oblate_coords, M, a)
    minkowski_bl = minkowski_metric_oblate_spheroidal(oblate_coords, a)
    return kerr_bl - minkowski_bl


def kerr_schild_cartesian_metric(
    cartesian_coords: jax.Array, 
    M: float, 
    a: float
) -> jax.Array: 
    """
    Computes the Kerr metric in Kerr-Schild form using Cartesian coordinates.

    Returns the spacetime metric tensor g_{μν} using the Kerr-Schild
    formulation, which expresses the metric as:
  
        g_{μν} = η_{μν} + 2H(r) l_μ l_ν

    in Cartesian coordinates.

    Parameters
    ----------
    cartesian_coords : jax.Array
        The spacetime coordinates `(t, x, y, z)` in Cartesian form.
    M : float
        Mass of the black hole.
    a : float
        Spin parameter of the black hole.

    Returns
    -------
    jax.Array
        A `(4, 4)` JAX array representing the Kerr-Schild metric tensor in Cartesian coordinates.

    Notes
    -----
    - The Kerr-Schild form is numerically stable near the horizon and useful in numerical relativity (doesn't have coordinate singularities).
    """
    t, x, y, z = cartesian_coords
    w = x**2 + y**2 + z**2 - a**2 
    r = jnp.sqrt(0.5*(w + jnp.sqrt(w**2 + 4*a**2*z**2))) 
    l_mu = jnp.array([1.0, 1.0*(r*x + a*y)/(r**2 + a**2), 1.0*(r*y - a*x)/(r**2 + a**2), z/r]) 
    g_mu_nu =  minkowski_metric(cartesian_coords) + 2*M*r**3/(r**4 + a**2*z**2)*jnp.einsum('i, j', l_mu, l_mu) 
    return g_mu_nu

def kerr_schild_cartesian_metric_distortion(
    cartesian_coords: jax.Array, 
    M: float, 
    a: float
) -> jax.Array: 
    """
    Computes the distortion part of Kerr metric in Kerr-Schild form using Cartesian coordinates.

    Returns the spacetime metric tensor distortion part using the Kerr-Schild
    formulation,
  
        Δ_{μν} (KS) = g_{μν} - η_{μν} :=  2H(r) l_μ l_ν

    in Cartesian coordinates.

    Parameters
    ----------
    cartesian_coords : jax.Array
        The spacetime coordinates `(t, x, y, z)` in Cartesian form.
    M : float
        Mass of the black hole.
    a : float
        Spin parameter of the black hole.

    Returns
    -------
    jax.Array
        A `(4, 4)` JAX array representing the Kerr-Schild metric tensor in Cartesian coordinates.

    Notes
    -----
    - Returns without the Minkowski diag(-1, +1, +1, +1) part.
    """
    t, x, y, z = cartesian_coords
    w = x**2 + y**2 + z**2 - a**2 
    r = jnp.sqrt(0.5*(w + jnp.sqrt(w**2 + 4*a**2*z**2))) 
    l_mu = jnp.array([1.0, 1.0*(r*x + a*y)/(r**2 + a**2), 1.0*(r*y - a*x)/(r**2 + a**2), z/r]) 
    h_mu_nu =  2*M*r**3/(r**4 + a**2*z**2)*jnp.einsum('i, j', l_mu, l_mu) 
    return h_mu_nu

def kerr_metric_eddington_finkelstein( 
    ief_coords: jax.Array,
    M: float,
    a: float
) -> jax.Array:
    """
    Computes the Kerr metric in advanced (ingoing) Eddington-Finkelstein coordinates (as originally presented by Roy Kerr).

    Returns the spacetime metric tensor g_{μν} for a rotating black hole
    using the Kerr geometry in advanced Eddington-Finkelstein coordinates
    `(v, r, θ, 𝜙̃)`.

    Parameters
    ----------
    ief_coords : jax.Array
        The spacetime coordinates `[v, r, θ, 𝜙̃]` in advanced Eddington-Finkelstein form.
    M : float
        Mass of the black hole.
    a : float
        Spin parameter of the black hole.

    Returns
    -------
    jax.Array
        A `(4, 4)` JAX array representing the Kerr metric tensor in advanced (ingoing) Eddington-Finkelstein coordinates.

    Notes
    -----
    - Useful for studying horizon crossing due to regularity at the event horizon.
    - Reduces to Schwarzschild Eddington-Finkelstein when `a = 0`.
    """
    v, r, theta, phi = ief_coords
    sigma = r**2 + a**2*jnp.cos(theta)**2 
    g_ef = jnp.array([[-1.0*(1-2*M/r), 1.0, 0.0, 2.0*M*a*r*jnp.sin(theta)**2/sigma], [1.0, 0.0, 0.0, a*jnp.sin(theta)**2], [0.0, 0.0, sigma, 0], [2.0*M*a*r*jnp.sin(theta)**2/sigma, a*jnp.sin(theta)**2, 0.0, (r**2 + a**2 + 2.0*M*a**2*r*jnp.sin(theta)**2/sigma)*jnp.sin(theta)**2]])
    return g_ef

def kerr_metric_eddington_finkelstein_distortion(
    ief_coords: jax.Array,
    M: float,
    a: float
) -> jax.Array:
    """
    Computes the Kerr metric in advanced (ingoing) Eddington-Finkelstein coordinates (as originally presented by Roy Kerr).

    Returns the distortion part of the metric tensor Δ_{μν} for a rotating black hole
    using the Kerr geometry in advanced Eddington-Finkelstein coordinates
    `(v, r, θ, 𝜙̃)`.

    Parameters
    ----------
    ief_coords : jax.Array
        The spacetime coordinates `[v, r, θ, 𝜙̃]` in advanced Eddington-Finkelstein form.
    M : float
        Mass of the black hole.
    a : float
        Spin parameter of the black hole.

    Returns
    -------
    jax.Array
        A `(4, 4)` JAX array representing the distortion part of Kerr metric tensor in advanced (ingoing) Eddington-Finkelstein coordinates.

    Notes
    -----
    - returns the Δ_{μν} (ingoing EF) = g_{μν} (ingoing EF) - η_{μν} (ingoing EF)
    """ 
    h_ef = kerr_metric_eddington_finkelstein(ief_coords, M, a) - minkowski_metric_eddington_finkelstein_rotating(ief_coords, a)
    return h_ef 
 
def christoffel_analytic_kerr_boyer_lindquist(
    boyer_lindquist_coords: jax.Array, 
    M: float, 
    a: float, 
    G: float = 1.0, 
    c: float = 1.0
) -> jax.Array: 
    """
    Computes the exact Christoffel symbols for the Kerr metric in Boyer-Lindquist (BL) coordinates.

    This function returns the Christoffel symbols Γᵏᵢⱼ as a 4D JAX array with shape `(4, 4, 4)`, 
    using analytic expressions for the rotating (stationary, non-static), uncharged Kerr spacetime.

    Parameters
    ----------
    boyer_lindquist_coords:  jax.Array
        Spacetime coordinates [t, r, 𝜗, φ] in Boyer-Lindquist form.
    M : float
        Mass of the black hole.
    a: float 
        Spin parameter of the black hole. 
    G: float = 1.0 
        Newtons' constant of gravitation (set as 1.0 for naturalized units)
    c: float = 1.0 
        Speed of light (set as 1.0 for naturalized units)

    Returns
    -------
    jax.Array
        A `(4, 4, 4)` JAX array representing the Christoffel symbols Γᵏᵢⱼ in BL form with coordinates `(t, r, 𝜗, φ)`.

    Example
    -------
    >>> coords = jnp.array([0.0, 10.0, jnp.pi / 4, 0.0])
    >>> Gamma = christoffel_analytic_kerr_boyer_lindquist(coords, M=1.0, a=0.7)
    >>> Gamma.shape
    (4, 4, 4)
    """
    t, r, theta, phi = boyer_lindquist_coords 

    rs = 2*G*M/c**2  
    Sigma = r**2 + a**2 * jnp.cos(theta)**2
    Delta = r**2 - rs * r + a**2
    A = (r**2 + a**2)**2 - a**2*jnp.cos(theta)**2

    christoffel = {}
    
    # Γ^t components
    christoffel['Gamma_t_tr'] = rs*(r**2 + a**2) * (r**2 - a**2 * jnp.cos(theta)**2) / (2 * Sigma**2 * Delta)
    christoffel['Gamma_t_ttheta'] = -rs*a**2*r*jnp.sin(theta)*jnp.cos(theta)/Sigma**2
    christoffel['Gamma_t_rphi'] = (rs*a*jnp.sin(theta)**2*(a**2*jnp.cos(theta)**2*(a**2 - r**2) - r**2*(a**2 + 3.0*r**2)))/(2*Sigma**2*Delta) 
    christoffel['Gamma_t_thetaphi'] = rs*a**3*r*jnp.sin(theta)**3*jnp.cos(theta)/Sigma**2

    # Γ^r
    christoffel['Gamma_r_tt'] = rs*Delta * (r**2 - a**2*jnp.cos(theta)**2)/(2*Sigma**3)
    christoffel['Gamma_r_tphi'] = -Delta*rs*a*jnp.sin(theta)**2*(r**2 - a**2*jnp.cos(theta)**2)/(2*Sigma**3)
    christoffel['Gamma_r_rr'] = (2*r*a**2*jnp.sin(theta)**2 - rs*(r**2 - a**2*jnp.cos(theta)**2))/(2*Sigma*Delta)
    christoffel['Gamma_r_rtheta'] = -a**2*jnp.sin(theta)*jnp.cos(theta)/Sigma
    christoffel['Gamma_r_thetatheta'] = -r*Delta/Sigma
    christoffel['Gamma_r_phiphi'] = Delta*jnp.sin(theta)**2/(2*Sigma**3)*(-2.0*r*Sigma**2 + rs*a**2*jnp.sin(theta)**2*(r**2 - a**2*jnp.cos(theta)**2))

    # Γ^\theta
    christoffel['Gamma_theta_tt'] = -c**2 * rs * a**2 * r * jnp.sin(theta) * jnp.cos(theta)/Sigma**3
    christoffel['Gamma_theta_tphi'] = c*rs*a*r*(r**2+a**2)*jnp.sin(theta)*jnp.cos(theta)/Sigma**3
    christoffel['Gamma_theta_rr'] = a**2*jnp.sin(theta)*jnp.cos(theta)/(Sigma*Delta)
    christoffel['Gamma_theta_rtheta'] = r/Sigma
    christoffel['Gamma_theta_thetatheta'] = -a**2*jnp.sin(theta)*jnp.cos(theta)/Sigma
    christoffel['Gamma_theta_phiphi'] = - 1/(Sigma)*jnp.sin(2.0*theta) * (0.5*(r**2 + a**2) + M*r*a**2*jnp.sin(theta)**2/(Sigma)*(2.0 + a**2*jnp.sin(theta)**2/Sigma))
    
    # Γ^\phi
    christoffel['Gamma_phi_tr'] = c*rs*a*(r**2 - a**2*jnp.cos(theta)**2)/(2*Sigma**2*Delta)
    christoffel['Gamma_phi_ttheta'] = -c*rs*a*r/(jnp.tan(theta)*Sigma**2)
    christoffel['Gamma_phi_rphi'] = (2*r*Sigma**2 + rs*(a**4*jnp.sin(theta)**2*jnp.cos(theta)**2 - r**2*(Sigma + r**2 + a**2)))/(2*Sigma**2*Delta)
    christoffel['Gamma_phi_thetaphi'] = 1/(jnp.tan(theta)*Sigma**2)*(Sigma**2 + rs*a**2*r*jnp.sin(theta)**2) 

    christoffel_matrix = jnp.zeros((4, 4, 4)) 
    ids = np.array([[0, 0, 1], [0, 0, 2], [0, 1, 3], [0, 2, 3], [1, 0, 0], [1, 0, 3], [1, 1, 1], [1, 1, 2], [1, 2, 2], [1, 3, 3], [2, 0, 0], [2, 0, 3], [2, 1, 1], [2, 1, 2], [2, 2, 2], [2, 3, 3], [3, 0, 1], [3, 0, 2], [3, 1, 3], [3, 2, 3]])
    ids_2 = ids.copy()
    for lis in ids_2: 
        lis[[1, 2]] = lis[[2, 1]]
    ids_2
    assert len(ids) == len(ids_2) == len(christoffel)
    
    for (lis, lis2, vals) in zip(ids, ids_2, christoffel.values()): 
        christoffel_matrix = christoffel_matrix.at[tuple(lis)].set(vals)  
        christoffel_matrix = christoffel_matrix.at[tuple(lis2)].set(vals)  

    return  christoffel_matrix 

def kretschmann_invariant_kerr_blachole(
    coords: jax.Array,
    M: float, 
    a: float, 
    is_cartesian: bool
) -> float: 
    """
    Computes the curvature invariant (Kretschmann scalar) for the Kerr metric.

    Returns the exact Kretschmann scalar exact Kretschmann scalar K = R^{μνρσ} R_{μνρσ}
    using the analytic expression specific to the Kerr geometry.

    Parameters
    ----------
    coords : jax.Array
        Spacetime coordinates `[t, r, 𝜗, φ]` in Boyer-Lindquist (spherical polar) form.
    M : float
        Mass of the black hole.
    a : float
        Spin parameter of the black hole.

    Returns
    -------
    float
        The Kretschmann scalar invariant `K` evaluated at the given coordinates (invariant under any input coordinate system used).

    Notes
    -----
    The Kretschmann scalar in Kerr geometry reveals the curvature singularity at the ring 
    (r = 0, θ = π/2). An analytic closed-form for the general Kerr Kretschmann scalar is:

        K = 48 G² M² (r⁶ - 15 a² r⁴ cos²𝜗 + 15 a⁴ r² cos⁴𝜗 - a⁶ cos⁶𝜗)/(r² + a² cos²𝜗)⁶

    where:
        - `G` = gravitational constant (default 1.0)
        - `c` = speed of light (default 1.0)

    For `a = 0`, this reduces to the Schwarzschild case:
        K = 48 G² M² / (c⁴ r⁶)

    Example
    -------
    >>> coords = jnp.array([0.0, 5.0, jnp.pi/3, 0.0])
    >>> k = kretschmann_kerr_analytic(coords, M=1.0, a=0.5)
    >>> k
    # Returns the Kretschmann scalar at r = 5, 𝜗 = π/3 for M = 1.0, a = 0.5
    """
    def kretschmann_scalar_cartesian(cartesian_coords): 
        """Returns the Kretschmann scalar with inputs in Cartesian coordinates"""
        t, r, theta, phi = cartesian_to_oblate_spheroid(cartesian_coords, a) 
        Rabcd_Rabcd = 48*M**2*(r**2 - a**2*jnp.cos(theta)**2)*((r**2 + a**2*jnp.cos(theta)**2)**2 - 16.0*r**2*a**2*jnp.cos(theta)**2)/((r**2 + a**2*jnp.cos(theta)**2)**6) 
        return Rabcd_Rabcd
    
    def kretschmann_scalar_boyer_lindquist(boyer_lindquist_coords: jax.Array) -> float:
        """Returns the Kretschmann scalar with inputs in Boyer-Lindquist coordinates"""
        t, r, theta, phi = boyer_lindquist_coords 
        Rabcd_Rabcd = 48*M**2*(r**2 - a**2*jnp.cos(theta)**2)*((r**2 + a**2*jnp.cos(theta)**2)**2 - 16.0*r**2*a**2*jnp.cos(theta)**2)/((r**2 + a**2*jnp.cos(theta)**2)**6) 
        return Rabcd_Rabcd
    
    return jax.lax.cond(is_cartesian, lambda _: kretschmann_scalar_cartesian(coords), lambda _: kretschmann_scalar_boyer_lindquist(coords), None)
    
if __name__ == '__main__': 
    jax.config.update("jax_enable_x64", True)
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    np.set_printoptions(suppress=True) 
