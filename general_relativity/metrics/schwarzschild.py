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
from general_relativity.metrics.minkowski import minkowski_metric_eddington_finkelstein_non_rotating

def schwarzschild_metric_spherical(
    coords: jax.Array, 
    M: float
) -> jax.Array: 
    """
    Computes the Schwarzschild metric tensor in spherical polar coordinates.

    This returns the spherically symmetric, static, non-rotating, uncharged
    Schwarzschild black hole metric in spherical polar coordinates
    (t, r, θ, φ).

    Parameters
    ----------
    coords : jax.Array
        Array of shape (4,) `(t, x, y, z)` representing the
        spacetime point in Cartesian coordinates.
    M : float
        Mass of the black hole.

    Returns
    -------
    jax.Array
        Array of shape (4, 4) representing the Schwarzschild metric tensor
        g_{μν} in spherical polar coordinates.

    Example
    -------
    >>> coords = jnp.array([0.0, 5.0, jnp.pi/2, 0.0])
    >>> g = schwarzschild_metric_spherical(coords, M=1.0)
    """
    t, r, theta, phi = coords
    g_tt = -1.0*(1.0 - 2*M/(r))
    g_rr = 1/(1 - 2*M/(r))
    g_theta_theta = r**2
    g_phi_phi = (r * jnp.sin(theta))**2
    gij = jnp.diag(jnp.array([g_tt, g_rr, g_theta_theta, g_phi_phi]))
    return gij

def schwarzschild_metric_spherical_distortion(
    coords: jax.Array, 
    M: float
) -> jax.Array: 
    """
    Computes the distortion tensor Δ_{μν} = g_{μν} - η_{μν} (in spherical representation)
    for the Schwarzschild metric in spherical polar coordinates.

    Parameters
    ----------
    coords : jax.Array
        Array of shape (4,) containing [t, r, θ, φ].
    M : float
        Mass of the black hole.
    
    Returns
    -------
    jax.Array
        Array of shape (4, 4) representing Δ_{μν} = g_{μν} - η_{μν}.

    Notes
    -----
    Only the diagonal components are nonzero for the Schwarzschild distortion
    in the spherical coordinates.
    """
    t, r, theta, phi = coords
    rs=2.0*M
    h_tt = rs/r
    h_rr = rs/(r-rs)
    h_theta_theta = 0.0
    h_phi_phi = 0.0
    hij = jnp.diag(jnp.array([h_tt, h_rr, h_theta_theta, h_phi_phi]))
    return hij

def schwarzschild_metric_kerr_schild(
    cart_coords: jax.Array, 
    M: float
) -> jax.Array:
    """
    Computes the Schwarzschild metric in Kerr-Schild Cartesian form: g_{μν} = η_{μν} + 2H(r) l_μ l_ν.

    Parameters
    ----------
    cart_coords : jax.Array
        Array of shape (4,) containing [t, x, y, z].
    M : float
        Mass of the black hole.

    Returns
    -------
    jax.Array
        Array of shape (4, 4) representing the Schwarzschild metric g_{μν}
        in Kerr-Schild Cartesian form.
    Example
    -------
    >>> coords = jnp.array([0.0, 2.6, 1.0, -0.5])
    >>> g = schwarzschild_metric_kerr_schild(coords, M=1.0)
    """ 
    t, x, y, z = cart_coords
    r = jnp.sqrt(x*x + y*y + z*z) 
    g_ks = jnp.array([[-1.0*(1-2*M/r), 2*M*x/r**2, 2*M*y/r**2, 2*M*z/r**2], [2*M*x/r**2, (1.0 + 2*M*x**2/r**3), 2.0*M*x*y/r**3, 2.0*M*x*z/r**3], [2*M*y/r**2, 2.0*M*x*y/r**3, (1.0 + 2*M*y**2/r**3), 2.0*M*y*z/r**3], [2*M*z/r**2, 2.0*M*x*z/r**3, 2.0*M*y*z/r**3, (1.0 + 2*M*z**2/r**3)]])
    return g_ks 

def schwarzschild_metric_kerr_schild_distortion(
    cart_coords: jax.Array, 
    M: float
) -> jax.Array: 
    """
    Computes the distortion tensor Δ_{μν} = g_{μν} (Cartesian KS) - η_{μν} (Cartesian) = 2H(r) l_μ l_ν.
    for the Schwarzschild metric in Kerr-Schild Cartesian coordinates.

    Parameters
    ----------
    cart_coords : jax.Array
        Array of shape (4,) containing [t, x, y, z].
    M : float
        Mass of the black hole.

    Returns
    -------
    jax.Array
        Array of shape (4, 4) representing Δ_{μν} for the Kerr-Schild form.
    Example
    -------
    >>> coords = jnp.array([0.0, 2.6, 1.0, -0.5])
    >>> g = schwarzschild_metric_kerr_schild_distortion(coords, M=1.0)
    """
    t, x, y, z = cart_coords
    r = jnp.sqrt(x*x + y*y + z*z) 
    h_ks = jnp.array([[2*M/r, 2*M*x/r**2, 2*M*y/r**2, 2*M*z/r**2], [2*M*x/r**2, 2*M*x**2/r**3, 2.0*M*x*y/r**3, 2.0*M*x*z/r**3], [2*M*y/r**2, 2.0*M*x*y/r**3, 2*M*y**2/r**3, 2.0*M*y*z/r**3], [2*M*z/r**2, 2.0*M*x*z/r**3, 2.0*M*y*z/r**3, 2*M*z**2/r**3]])
    return h_ks 

def schwarzschild_metric_eddington_finkelstein(
    ief_coords: jax.Array, 
    M: float
) -> jax.Array: 
    """
    Computes the Schwarzschild metric in ingoing Eddington-Finkelstein coordinates (v, r, θ, φ).

    Parameters
    ----------
    ief_coords : jax.Array
        Array of shape (4,) containing [v, r, θ, φ].
    M : float
        Mass of the black hole.

    Returns
    -------
    jax.Array
        Array of shape (4, 4) representing the metric g_{μν}
        in ingoing Eddington-Finkelstein coordinates.
    
    >>> coords = jnp.array([0.0, 2.6, 1.0, -0.5])
    >>> g = schwarzschild_metric_kerr_schild_distortion(coords, M=1.0)
    """
    v, r, theta, phi = ief_coords
    g_ef = jnp.array([[-1.0*(1-2*M/r), 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, r**2, 0], [0.0, 0.0, 0.0, r**2*jnp.sin(theta)**2]])
    return g_ef

def schwarzschild_metric_eddington_finkelstein_distortion(
    ief_coords: jax.Array, 
    M: float
) -> jax.Array: 
    """
    Computes the distortion tensor Δ_{μν} (ingoing EF) = g_{μν} (ingoing EF) - η_{μν} (ingoing EF)
    for the Schwarzschild metric in ingoing Eddington-Finkelstein (EF) coordinates.

    Parameters
    ----------
    ief_coords : jax.Array
        Array of shape (4,) containing [v, r, θ, φ].
    M : float
        Mass of the black hole.

    Returns
    -------
    jax.Array
        Array of shape (4, 4) representing Δ_{μν} in Eddington-Finkelstein coordinates.

    Notes
    -----
    Requires the `minkowski_metric_eddington_finkelstein_non_rotating`
    """
    return schwarzschild_metric_eddington_finkelstein(ief_coords, M) - minkowski_metric_eddington_finkelstein_non_rotating(ief_coords) 

 
def christoffel_symbols_analytic(
    coords: jax.Array, 
    M: float
) -> jax.Array: 
    """
    Computes the exact Christoffel symbols for the Schwarzschild geometry in spherical polar coordinates.

    This function returns the Christoffel symbols Γᵏᵢⱼ as a 4D JAX array with shape `(4, 4, 4)`, 
    using analytic expressions for the non-rotating, uncharged Schwarzschild spacetime.

    Parameters
    ----------
    coords : jax.Array
        Array of shape (4,) `(t, r, θ, φ)` in spherical polar form.
    M : float

    Returns
    -------
    jax.Array
        A `(4, 4, 4)` JAX array representing the Christoffel symbols Γᵏᵢⱼ 
        with respect to coordinates `(t, r, θ, φ)`.

    Example
    -------
    >>> coords = jnp.array([0.0, 10.0, jnp.pi / 4, 0.0])
    >>> Gamma = christoffel_symbols_analytic(coords, M=1.0)
    >>> Gamma.shape
    (4, 4, 4)
    """
    t, r, theta, phi = coords 

    rs = 2*M 
    gamma_100 = rs*(r-rs)/(2*r**3) 
    gamma_001 = rs/(2*r*(r- rs)) 
    gamma_111 = -rs/(2*r*(r-rs)) 
    gamma_212 = 1/r 
    gamma_313 = 1/r  
    gamma_122 = -(r-rs) 
    gamma_323 = 1/jnp.tan(theta) 
    gamma_133 = -(r - rs)*jnp.sin(theta)**2 
    gamma_233 = -jnp.sin(theta)*jnp.cos(theta) 

    christoffel = {} 
    # Γ^t components
    christoffel['Gamma_t_tr'] = gamma_001

    # Γ^r
    christoffel['Gamma_r_tt'] = gamma_100
    christoffel['Gamma_r_rr'] = gamma_111 
    christoffel['Gamma_r_thetatheta'] = gamma_122
    christoffel['Gamma_r_phiphi'] = gamma_133 

    # Γ^\theta
    christoffel['Gamma_theta_rtheta'] = gamma_212
    christoffel['Gamma_theta_phiphi'] = gamma_233 
    
    # Γ^\phi
    christoffel['Gamma_phi_rphi'] = gamma_313
    christoffel['Gamma_phi_thetaphi'] = gamma_323 

    christoffel_matrix = jnp.zeros((4, 4, 4))
    ids = jnp.array([[0, 0, 1], [1, 0, 0], [1, 1, 1], [1, 2, 2], [1, 3, 3], [2, 1, 2], [2, 3, 3], [3, 1, 3], [3, 2, 3]])
    ids_2 = ids.at[:, [1, 2]].set(ids[:, [2, 1]]) 

    assert len(ids) == len(ids_2) == len(christoffel)
    
    for (lis, lis2, vals) in zip(ids, ids_2, christoffel.values()): 
        christoffel_matrix = christoffel_matrix.at[tuple(lis)].set(vals)  
        christoffel_matrix = christoffel_matrix.at[tuple(lis2)].set(vals)  

    return  christoffel_matrix

def christoffel_jac_analytic(
    coords: jax.Array, 
    M: float
) -> jax.Array:
    """
    Computes the exact Christoffel symbols for the Schwarzschild geometry in spherical polar coordinates.

    This function returns the Christoffel symbols Jacobian dΓᵏᵢⱼ as a 4D JAX array with shape `(4, 4, 4, 4)`, 
    using analytic expressions for the non-rotating, uncharged Schwarzschild spacetime.

    Parameters
    ----------
    coords : jax.Array
        Array of shape (4,) `(t, r, θ, φ)` in spherical polar form.
    M : float
        Mass of the black hole.

    Returns
    -------
    jax.Array
        A `(4, 4, 4, 4)` JAX array representing the Christoffel symbols dΓᵏᵢⱼ 
        with respect to coordinates `(t, r, θ, φ)`.

    Example
    -------
    >>> coords = jnp.array([0.0, 10.0, jnp.pi / 4, 0.0])
    >>> Gamma = christoffel_jac_analytic(coords, M=1.0)
    >>> Gamma.shape
    (4, 4, 4, 4)
    """
    t, r, theta, phi = coords 

    rs = 2.0*M 
    # Γ^t,.. components
    gamma_001_1 = - (2.0*r - rs)*rs/(2.0*r**2*(r - rs)**2)
    
    # Γ^r,.. components
    gamma_100_1 = - (2.0*r - 3.0*rs)*rs/(2.0*r**4) 
    gamma_111_1 =  (2.0*r - rs)*rs/(2.0*r**2*(r - rs)**2) 
    gamma_122_1 = -1.0 
    gamma_133_1 = -jnp.sin(theta)**2 
    gamma_133_2 = -(r -rs)*jnp.sin(2*theta)

    # Γ^\theta,.. components
    gamma_212_1 = -1.0/r**2 
    gamma_233_2 = -jnp.cos(2.0*theta) 

    # Γ^\phi
    gamma_313_1 = -1.0/r**2  
    gamma_323_2 = -1.0/jnp.sin(theta)**2 

    christoffel_jac = {} 
   
    #   Γ^t components
    christoffel_jac['Gamma_t_tr,r'] = gamma_001_1

    # Γ^r
    christoffel_jac['Gamma_r_tt,r'] = gamma_100_1
    christoffel_jac['Gamma_r_rr,r'] = gamma_111_1 
    christoffel_jac['Gamma_r_thetatheta_r'] = gamma_122_1 
    christoffel_jac['Gamma_r_phiphi_r'] = gamma_133_1
    christoffel_jac['Gamma_r_phiphi_theta'] = gamma_133_2 

    # Γ^\theta
    christoffel_jac['Gamma_theta_rtheta_r'] = gamma_212_1
    christoffel_jac['Gamma_theta_phiphi_theta'] = gamma_233_2 
    
    # Γ^\phi
    christoffel_jac['Gamma_phi_rphi_r'] = gamma_313_1
    christoffel_jac['Gamma_phi_thetaphi_theta'] = gamma_323_2 

    christoffel_jac_matrix = jnp.zeros((4, 4, 4, 4))
    ids = jnp.array([[1, 0, 0, 1], [1, 1, 0, 0], [1, 1, 1, 1], [1, 1, 2, 2], [1, 1, 3, 3], [2, 1, 3, 3], [1, 2, 1, 2], [2, 2, 3, 3], [1, 3, 1, 3], [2, 3, 2, 3]])
    ids_2 = ids.at[:, [2, 3]].set(ids[:, [3, 2]]) 
    print(ids_2)
    
    for (lis, lis2, vals) in zip(ids, ids_2, christoffel_jac.values()): 
        christoffel_jac_matrix = christoffel_jac_matrix.at[tuple(lis)].set(vals)  
        christoffel_jac_matrix = christoffel_jac_matrix.at[tuple(lis2)].set(vals)  

    return  christoffel_jac_matrix

def riemann_tensor_covariant_analytic(
    coords: jax.Array, 
    M: float
) -> jax.Array:
    """
    Computes the exact Riemann curvature tensor (covariant form) for Schwarzschild geometry.
    
    Returns the analytic covariant Riemann tensor components R_{μνρσ} as a 4D JAX array
    with shape `(4, 4, 4, 4)`, using analytic expressions for the Schwarzschild spacetime.

    Parameters
    ----------
    coords : jax.Array
        Array of shape (4,) `(t, r, θ, φ)` in spherical polar form.
    M : float
        Mass of the black hole.

    Returns
    -------
    jax.Array
        A `(4, 4, 4, 4)` JAX array representing the covariant Riemann tensor 
        components R_{μνρσ} with respect to coordinates `(t, r, θ, φ)`.

    Notes
    -----
    The Riemann tensor is computed using the analytic form:
        R_{μνρσ} = ∂_ρ Γ_{μνσ} - ∂_σ Γ_{μνρ} + Γ^λ_{μν} Γ_{λρσ} - Γ^λ_{μρ} Γ_{λνσ}
    where Γ are the Christoffel symbols.

    Example
    -------
    >>> coords = jnp.array([0.0, 6.0, jnp.pi/3, 0.0])
    >>> R = riemann_tensor_covariant_analytic(coords, M=1.0)
    >>> R.shape
    (4, 4, 4, 4)
    """
    t, r, theta, phi = coords 
    rs = 2.0*M 
    # R_t components
    R_0101 = -1.0*rs/r**3 
    R_0202 = 0.5*(r-rs)*rs/r**2
    # R_0303 = 0.5*(r-rs)
    R_0303 = -1.0*0.5*rs*(rs-r)*jnp.sin(theta)**2/r**2  

    # R_r components
    R_1212 = -0.5*(rs/(r - rs)) 
    R_1313 = -0.5*(rs*jnp.sin(theta)**2/(r-rs)) 
    R_2323 = r*rs*jnp.sin(theta)**2

    riemann_tensor = {} 
   
    #   R_t components
    riemann_tensor['R_trtr'] = R_0101 
    riemann_tensor['R_tthetattheta'] = R_0202 
    riemann_tensor['R_tphitphi'] = R_0303
    # R_r components 
    riemann_tensor['R_rthetartheta'] = R_1212  
    riemann_tensor['R_rphirphi'] = R_1313  
    riemann_tensor['R_thetaphithetaphi'] = R_2323
     
    riemann_tensor_matrix = jnp.zeros((4, 4, 4, 4))
    ids = jnp.array([[0,1,0,1],[0,2,0,2],[0,3,0,3],[1,2,1,2],[1,3,1,3],[2,3,2,3]])
    ids_skew1 = ids.at[:, [2, 3]].set(ids[:, [3, 2]]) 
    ids_skew2 = ids.at[:, [0, 1]].set(ids[:, [1, 0]])
    ids_sym = ids_skew1.at[:, [0, 1]].set(ids[:, [1, 0]])
    
    for (lis, lis2, lis3, lis4, vals) in zip(ids, ids_skew1, ids_skew2, ids_sym, riemann_tensor.values()): 
        riemann_tensor_matrix = riemann_tensor_matrix.at[tuple(lis)].set(-1.0*vals)
        riemann_tensor_matrix = riemann_tensor_matrix.at[tuple(lis2)].set(vals)  
        riemann_tensor_matrix = riemann_tensor_matrix.at[tuple(lis3)].set(vals)
        riemann_tensor_matrix = riemann_tensor_matrix.at[tuple(lis4)].set(-1.0*vals)  

    return -1.0*riemann_tensor_matrix

def kretschmann_schwarzschild_analytic(
    coords: jax.Array,
    M: float
) -> float:
    """
    Computes the curvature invariant for Schwarzschild geometry.
    
    Returns the exact Kretschmann scalar K = R^{μνρσ} R_{μνρσ} as a float,
    using the analytic expression for Schwarzschild geometry.

    Parameters
    ----------
    coords : jax.Array
        Array of shape (4,) `(t, r, θ, φ)` in spherical polar form.
    M : float
        Mass of the black hole.

    Returns
    -------
    float
        The Kretschmann scalar invariant K at the given coordinates.

    Notes
    -----
    The analytic expression for the Kretschmann scalar in Schwarzschild geometry is:
        K = 48 M² / r⁶
    where:
        G = gravitational constant (default 1.0)
        c = speed of light (default 1.0)

    Example
    -------
    >>> coords = jnp.array([0.0, 3.0, jnp.pi/2, 0.0])
    >>> k = kretschmann_schwarzschild_analytic(coords, M=2.0)
    >>> k
    48.0 / (3**6)  # ≈ 0.1975 for M=2.0 at r=3.0
    """
    t, r, theta, phi = coords 
    kretschmann_invariant = 48.0*M**2/r**6  
    return kretschmann_invariant


if __name__ == '__main__': 
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import time
    np.set_printoptions(suppress=True)

    