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

def minkowski_metric(coords: jax.Array) -> jax.Array:
    """
    Returns the flat (constant) Minkowski metric tensor `η_{μν}` in Cartesian coordinates.

    The metric has the signature `(-, +, +, +)` in units where `c = 1`, corresponding
    to flat spacetime in special relativity.

    Parameters
    ----------
    coords : jax.Array
        Array of shape (4,) of four scalars `[t, r, θ, φ]` representing the
        spacetime point in Cartesian coordinates.

    Returns
    -------
    jax.Array
        A `(4, 4)` JAX array representing the Minkowski metric tensor `η_{μν}`
        with signature `(-, +, +, +)`.

    Example
    -------
    >>> coords = [0.0, 1.0, 2.0, 3.0]
    >>> g = minkowski_metric(coords)
    >>> g
    [[-1.  0.  0.  0.]
     [ 0.  1.  0.  0.]
     [ 0.  0.  1.  0.]
     [ 0.  0.  0.  1.]]
    """

    minkowski_diag = jnp.array([-1.0, 1.0, 1.0, 1.0])
    return jnp.diag(minkowski_diag)

def minkowski_metric_spherical(sph_coords: jax.Array) -> jax.Array:
    """
    Returns the Minkowski metric tensor `η_{μν}` in spherical representation.

    The metric has signature `(-, +, +, +)` in spherical coordinates:
    `ds^2 = -dt^2 + dr^2 + r^2 dθ^2 + r^2 sin^2θ dφ^2`.

    Parameters
    ----------
    sph_coords : jax.Array
        Array of shape (4,) `(t, r, θ, φ)` representing the spacetime point
        in spherical coordinates.

    Returns
    -------
    jax.Array
        A `(4, 4)` JAX array representing the Minkowski metric tensor in spherical coordinates.

    Example
    -------
    >>> sph_coords = [0.0, 2.0, jnp.pi / 4, 0.0]
    >>> g = minkowski_metric_spherical(sph_coords)
    >>> g.shape
    (4, 4)
    """

    t, r, theta, phi = sph_coords
    eta_polar = jnp.array([-1.0, 1.0, r**2, r**2*jnp.sin(theta)**2])
    return jnp.diag(eta_polar)

def minkowski_metric_oblate_spheroidal(oblate_coords: jax.Array, a: float) -> jax.Array:
    """
    Returns the Minkowski metric `η_{μν}` in oblate spheroidal coordinates.

    This metric is used in the analysis of rotating spacetimes in flat space,
    with the metric taking the form:
        ds^2 = -dt^2 + Σ/(r^2 + a^2) dr^2 + Σ dθ^2 + (r^2 + a^2) sin^2 θ dφ^2,
    where Σ = r^2 + a^2 cos^2θ.

    Parameters
    ----------
    oblate_coords : jax.Array
        Array of shape (4,) `[t, r, θ, φ]` representing the spacetime point
        in oblate spheroidal coordinates.
    a : float
        Rotation parameter defining the oblateness of the coordinates.

    Returns
    -------
    jax.Array
        A `(4, 4)` JAX array representing the Minkowski metric tensor
        in oblate spheroidal coordinates.

    Example
    -------
    >>> coords = [0.0, 2.0, jnp.pi / 4, 0.0]
    >>> g = minkowski_metric_oblate_spheroidal(coords, a=0.5)
    >>> g.shape
    (4, 4)
    """
    t, r, theta, phi = oblate_coords
    sigma = r**2 + a**2 * jnp.cos(theta)**2
    eta_oblate_spheroid = jnp.array([
        -1.0,
        sigma / (r**2 + a**2),
        sigma,
        (r**2 + a**2) * jnp.sin(theta)**2
    ])
    return jnp.diag(eta_oblate_spheroid)

def minkowski_metric_eddington_finkelstein_non_rotating(ief_coords: jax.Array) -> jax.Array:
    """
    Returns the Minkowski metric `η_{μν}` in ingoing Eddington-Finkelstein coordinates
    for a non-rotating spacetime.

    The metric in these coordinates has signature `(-, +, +, +)` with off-diagonal
    terms:
        ds^2 = -dv^2 + 2 dv dr + r^2 dθ^2 + r^2 sin^2θ dφ^2.

    Parameters
    ----------
    ief_coords : jax.Array
        Array of shape (4,) `[v, r, θ, 𝜙̃]` representing the spacetime point
        in ingoing (advanced) Eddington-Finkelstein coordinates.

    Returns
    -------
    jax.Array
        A `(4, 4)` JAX array representing the Minkowski metric tensor
        in non-rotating Eddington-Finkelstein coordinates.

    Example
    -------
    >>> ief_coords = [0.0, 2.0, jnp.pi / 4, 0.0]
    >>> g = minkowski_metric_eddington_finkelstein_non_rotating(ief_coords)
    >>> g.shape
    (4, 4)
    """
    v, r, theta, phi = ief_coords
    eta_ef = jnp.array([-1.0, 0.0, r**2, r**2 * jnp.sin(theta)**2])
    eta_ef = jnp.diag(eta_ef)
    eta_ef = eta_ef.at[0, 1].set(1.0)
    eta_ef = eta_ef.at[1, 0].set(1.0)
    return eta_ef

def minkowski_metric_eddington_finkelstein_rotating(ief_coords: jax.Array, a: float) -> jax.Array:
    """
    Returns the Minkowski metric `η_{μν}` in ingoing (advanced) Eddington-Finkelstein
    coordinates adapted to a rotating frame.

    This metric includes cross terms that reflect frame dragging in flat space:
        ds^2 = -dv^2 + 2 dv dr + Σ dθ^2 + (r^2 + a^2) sin^2θ dφ^2 + 2 a sin^2θ dr dφ,
    where Σ = r^2 + a^2 cos^2θ.

    Parameters
    ----------
    ief_coords : jax.Array
        Array of shape (4,) `[v, r, θ, 𝜙̃]` representing the spacetime point
        in ingoing Eddington-Finkelstein coordinates for a rotating system.
    a : float
        Rotation parameter defining the angular momentum per unit mass.

    Returns
    -------
    jax.Array
        A `(4, 4)` JAX array representing the Minkowski metric tensor
        in rotating Eddington-Finkelstein coordinates.

    Example
    -------
    >>> coords = [0.0, 2.0, jnp.pi / 4, 0.0]
    >>> g = minkowski_metric_eddington_finkelstein_rotating(coords, a=0.5)
    >>> g.shape
    (4, 4)
    """

    v, r, theta, phi = ief_coords
    sigma = r**2 + a**2 * jnp.cos(theta)**2

    eta_efr = jnp.array([
        -1.0,
        0.0,
        sigma,
        (r**2 + a**2) * jnp.sin(theta)**2
    ])
    eta_efr = jnp.diag(eta_efr)
    eta_efr = eta_efr.at[0, 1].set(1.0)
    eta_efr = eta_efr.at[1, 0].set(1.0)
    eta_efr = eta_efr.at[1, 3].set(a * jnp.sin(theta)**2)
    eta_efr = eta_efr.at[3, 1].set(a * jnp.sin(theta)**2)
    return eta_efr

if __name__ == '__main__': 
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import time
    np.set_printoptions(suppress=True)
