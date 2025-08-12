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

import jax 
import jax.numpy as jnp 

def cartesian_to_spherical(
    cartesian_coords: jax.Array
) -> jax.Array: 
    """
    Converts Cartesian coordinates to spherical coordinates.

    Transforms (t, x, y, z) in Cartesian coordinates to
    (t, r, θ, φ) in spherical coordinates.

    Parameters
    ----------
    cartesian_coords : jax.Array
        Array of shape `(4,)` containing (t, x, y, z) coordinates.

    Returns
    -------
    jax.Array
        Array of shape `(4,)` with spherical coordinates
        (t, r, θ, φ).

    Notes
    -----
    - Commonly used to convert cartesian coords initial data to spherical coordinates for GR simulations.
    """
    t, x, y, z = cartesian_coords
    r = jnp.sqrt(x**2 + y**2 + z**2)
    theta = jnp.arccos(z/r)
    phi = jnp.arctan2(y, x) % (2 * jnp.pi)
    return jnp.array([t, r, theta, phi])


def spherical_to_cartesian(
    spherical_coords: jax.Array
) -> jax.Array: 
    """
    Converts spherical coordinates to Cartesian coordinates.

    Transforms (t, r, θ, φ) in spherical coordinates to
    (t, x, y, z) in Cartesian coordinates.

    Parameters
    ----------
    spherical_coords : jax.Array
        Array of shape (4,) containing (t, r, θ, φ) coordinates.

    Returns
    -------
    jax.Array
        Array of shape `(4,)` with Cartesian coordinates
        `(t, x, y, z)`.

    Notes
    -----
    - Useful when switching from spherical grids to Cartesian grids for visualization or integration.
    """
    t, r, theta, phi = spherical_coords 
    x = r*jnp.sin(theta)*jnp.cos(phi)
    y = r*jnp.sin(theta)*jnp.sin(phi)
    z = r*jnp.cos(theta) 
    return jnp.stack([t, x, y, z], axis=-1)

def spherical_to_kerr_schild_cartesian(
    spherical_coords: jax.Array,
    M: float,
) -> jax.Array:
    """
    Converts spherical coordinates to Kerr-Schild Cartesian coordinates.

    Parameters
    ----------
    spherical_coords : jax.Array
        Spherical coordinates (t, r, θ, φ).
    M : float
        Mass of the black hole.

    Returns
    -------
    jax.Array
        Kerr-Schild Cartesian coordinates (t, x, y, z).
    
    Notes
    -----
    - Useful for horizon-penetrating simulations in Kerr spacetime.
    """
    return boyer_lindquist_to_kerr_schild(
        spherical_coords,
        M=M,
        a=0.
    )

def kerr_schild_cartesian_to_spherical(
    kerr_schild_coords: jax.Array,
    M: float,
) -> jax.Array:
    """
    Converts Kerr-Schild Cartesian coordinates to spherical coordinates.

    Parameters
    ----------
    kerr_schild_coords : jax.Array
        Kerr-Schild Cartesian coordinates (t, x, y, z).
    
    Returns
    -------
    jax.Array
        Spherical coordinates (t, r, θ, φ).

    Notes
    -----
    - Useful for interpreting quantities in the standard spherical chart.
    """
    return kerr_schild_to_boyer_lindquist(
        kerr_schild_coords,
        M=M,
        a=0.
    )

def ingoing_eddington_finkelstein_to_spherical(
    ief_coords: jax.Array, 
    M: float
) -> jax.Array: 
    """
    Converts ingoing Eddington-Finkelstein coordinates to spherical coordinates.

    Transforms (v, r, θ, φ)  in ingoing Eddington-Finkelstein coordinates
    to (t, r, θ, φ)  in spherical coordinates.

    Parameters
    ----------
    ief_coords : jax.Array
        Array of shape (4,) containing (v, r, θ, φ) coordinates.
    M : float
        Mass of the black hole.

    Returns
    -------
    jax.Array
        Array of shape `(4,)` with spherical coordinates
        `(t, r, theta, phi)`.

    Notes
    -----
    - `t = v - r*` where `r*` is the tortoise coordinate.
    - Frequently used to switch from horizon-penetrating coordinates back to BL time for analysis (devoid of coordinate singularities).
    """
    v, r, theta, phi = ief_coords 
    r_star = r + 2*M*jnp.log(jnp.abs(r/(2*M) - 1))
    t = v - r_star
    return jnp.stack([t, r, theta, phi], axis=-1)


def spherical_to_ingoing_eddington_finkelstein(
    spherical_coords: jax.Array, 
    M: float
) -> jax.Array: 
    """
    Converts spherical coordinates to ingoing Eddington-Finkelstein coordinates.

    Transforms (t, r, θ, φ) to (v, r, θ, φ) where
    `v = t + r*`, with `r*` being the tortoise coordinate.

    Parameters
    ----------
    spherical_coords : jax.Array
        Sequence containing (t, r, θ, φ) coordinates.
    M : float
        Mass of the black hole.

    Returns
    -------
    jax.Array
        Array of shape `(4,)` with ingoing Eddington-Finkelstein
        coordinates (v, r, θ, φ).

    Notes
    -----
    - Useful for switching to horizon-penetrating coordinates in numerical simulations.
    """
    t, r, theta, phi = spherical_coords 
    r_star = r + 2*M*jnp.log(jnp.abs(r/(2*M) - 1))
    v = t + r_star
    return jnp.stack([v, r, theta, phi], axis=-1)


def outgoing_eddington_finkelstein_to_spherical(
    oef_coords: jax.Array, 
    M: float
) -> jax.Array:
    """
    Converts outgoing Eddington-Finkelstein coordinates to spherical coordinates.

    Transforms (u, r, θ, φ) in outgoing EF to (t, r, θ, φ) in spherical coordinates.

    Parameters
    ----------
    oef_coords : jax.Array
        Sequence containing (u, r, θ, φ) coordinates.
    M : float
        Mass of the black hole.

    Returns
    -------
    jax.Array
        Array of shape `(4,)` with spherical coordinates
        `(t, r, theta, phi)`.

    Notes
    -----
    - `t = u + r*` where `r*` is the tortoise coordinate.
    """ 
    u, r, theta, phi = oef_coords 
    r_star = r + 2*M*jnp.log(jnp.abs(r/(2*M) - 1))
    t = u + r_star
    return jnp.stack([t, r, theta, phi], axis=-1)


def spherical_to_outgoing_eddington_finkelstein(
    spherical_coords: jax.Array, 
    M: float
) -> jax.Array:
    """
    Converts spherical coordinates to outgoing Eddington-Finkelstein coordinates.

    Transforms `(t, r, θ, φ)` to `(u, r, θ, φ)` where
    `u = t - r*`, with `r_star` being the tortoise coordinate.

    Parameters
    ----------
    spherical_coords : jax.Array
        Sequence containing `(t, r, θ, φ)` coordinates.
    M : float
        Mass of the black hole.

    Returns
    -------
    jax.Array
        Array of shape `(4,)` with outgoing Eddington-Finkelstein
        coordinates `(u, r, θ, φ)`.

    Notes
    -----
    - Useful for outgoing null coordinate-based horizon analyses and wave extraction.
    """
    t, r, theta, phi = spherical_coords 
    r_star = r + 2*M*jnp.log(jnp.abs(r/(2*M) - 1))
    u = t - r_star
    return jnp.stack([u, r, theta, phi], axis=-1)

def cartesian_to_oblate_spheroid(
    cartesian_coords: jax.Array, 
    a: float
) -> jax.Array:  
    """
    Converts Cartesian coordinates to oblate spheroidal coordinates.

    Transforms `(t, x, y, z)` to `(t, r, θ, φ)` in the `m -> 0` limit
    of Boyer-Lindquist geometry.

    Parameters
    ----------
    cartesian_coords : jax.Array
        Sequence containing `(t, x, y, z)` coordinates.
    a : float
        Spin parameter of the black hole (relevant for oblate deformation).

    Returns
    -------
    jax.Array
        Array of shape `(4,)` with oblate spheroidal coordinates
        `(t, r, θ, φ)`.

    Notes
    -----
    - Reduces to spherical coordinates when `a = 0`.
    """ 
    t, x, y, z = cartesian_coords
    w = x**2 + y**2 + z**2 - a**2 
    r = jnp.sqrt(0.5*(w + jnp.sqrt(w**2 + 4*a**2*z**2)))
    theta = jnp.arccos(z/r)
    phi = jnp.arctan2(y, x) % (2 * jnp.pi)
    return jnp.array([t, r, theta, phi])


def oblate_spheroid_to_cartesian(
    oblate_coords: jax.Array, 
    a: float
) -> jax.Array:
    """
    Converts oblate spheroidal coordinates to Cartesian coordinates.

    Transforms `(t, r, θ, φ)` in oblate spheroidal coordinates
    (Boyer-Lindquist in `m -> 0` limit) to `(t, x, y, z)`.

    Parameters
    ----------
    oblate_coords : jax.Array
        Sequence containing `(t, r, θ, φ)` coordinates.
    a : float
        Spin parameter of the black hole.

    Returns
    -------
    jax.Array
        Array of shape `(4,)` with Cartesian coordinates
        `(t, x, y, z)`.

    Notes
    -----
    - When `a = 0`, this reduces to the standard spherical to Cartesian transformation.
    """
    t, r, theta, phi = oblate_coords 
    x = jnp.sqrt(r**2 + a**2)*jnp.sin(theta)*jnp.cos(phi)
    y = jnp.sqrt(r**2 + a**2)*jnp.sin(theta)*jnp.sin(phi)
    z = r*jnp.cos(theta) 
    return jnp.stack([t, x, y, z], axis=-1)


def oblate_spheroid_to_kerr_schild(
    oblate_coords: jax.Array, 
    a: float
) -> jax.Array:
    """
    Converts oblate spheroidal coordinates to Kerr-Schild coordinates.

    Parameters
    ----------
    oblate_coords : jax.Array
        Oblate spheroidal coordinates `(t, r, θ, φ)`.
    a : float
        Spin parameter of the black hole (angular momentum per unit mass).

    Returns
    -------
    jax.Array
        The corresponding Kerr-Schild coordinates `(t, x, y, z)`.

    Notes
    -----
    - Useful for horizon-penetrating simulations where Kerr-Schild avoids coordinate singularities.
    - The polar angle θ and azimuthal angle φ are interpreted in the oblate spheroidal system.
    """
    t, r, theta, phi = oblate_coords
    phi_offset = jnp.arctan2(a, r) % (2 * jnp.pi)
    x = jnp.sqrt(r**2 + a**2)*jnp.sin(theta)*jnp.cos(phi + phi_offset)
    y = jnp.sqrt(r**2 + a**2)*jnp.sin(theta)*jnp.sin(phi + phi_offset)
    z = r*jnp.cos(theta) 
    return jnp.stack([t, x, y, z], axis=-1)

 
def kerr_schild_to_oblate_spheroid(
    kerr_schild_coords: jax.Array, 
    a: float
) -> jax.Array:
    """
    Converts Kerr-Schild coordinates to oblate spheroidal coordinates.

    Parameters
    ----------
    kerr_schild_coords : jax.Array
        Kerr-Schild coordinates `(t, x, y, z)`.
    a : float
        Spin parameter of the black hole.

    Returns
    -------
    jax.Array
        Oblate spheroidal coordinates `(t, r, θ, φ)`.

    Notes
    -----
    - Useful for analyzing Kerr spacetime quantities in the oblate spheroidal system.
    """
    t, x, y, z = kerr_schild_coords
    w = x**2 + y**2 + z**2 - a**2
    r = jnp.sqrt(0.5 * (w + jnp.sqrt(w**2 + 4 * a**2 * z**2)))
    theta = jnp.arccos(z / r)
    phi = (jnp.arctan2(y, x) - jnp.arctan2(a / r)) % (2 * jnp.pi)
    return jnp.stack([t, r, theta, phi], axis=-1)


def boyer_lindquist_to_kerr_schild(
    boyer_lindquist_coords: jax.Array,
    M: float, 
    a: float
) -> jax.Array:
    """
    Converts Boyer-Lindquist coordinates to Kerr-Schild coordinates.

    Parameters
    ----------
    boyer_lindquist_coords : jax.Array
        Boyer-Lindquist coordinates `(t, r, θ, φ)` with black hole mass M > 0.
    M : float
        Mass of the black hole.
    a : float
        Spin parameter of the black hole.

    Returns
    -------
    jax.Array
        The corresponding Kerr-Schild coordinates `(t, x, y, z)`.

    Notes
    -----
    - The polar angle θ and modified azimuthal angle φ are used in the transformation.
    - Kerr-Schild coordinates are regular across the horizon.
    """
    t_bl, r_bl, theta_bl, phi_bl = boyer_lindquist_coords
    phi_offset = jnp.arctan2(a, r_bl) % (2 * jnp.pi)
    t = t_bl + M * jnp.log(jnp.abs(r_bl**2 - 2 * M * r_bl + a**2)) + M**2 / jnp.sqrt(M**2 - a**2) * jnp.log(jnp.abs((r_bl - (M + jnp.sqrt(M**2 - a**2))) / (r_bl - (M - jnp.sqrt(M**2 - a**2)))))
    phi = phi_bl + a / (2 * jnp.sqrt(M**2 - a**2)) * jnp.log(jnp.abs((r_bl - (M + jnp.sqrt(M**2 - a**2))) / (r_bl - (M - jnp.sqrt(M**2 - a**2))))) + phi_offset
    x = jnp.sqrt(r_bl**2 + a**2)*jnp.sin(theta_bl)*jnp.cos(phi)
    y = jnp.sqrt(r_bl**2 + a**2)*jnp.sin(theta_bl)*jnp.sin(phi)
    z = r_bl*jnp.cos(theta_bl)
    return jnp.stack([t, x, y, z], axis=-1)

 
def kerr_schild_to_boyer_lindquist(
    kerr_schild_coords: jax.Array, 
    M: float, 
    a: float
) -> jax.Array:
    """
    Converts Kerr-Schild coordinates to Boyer-Lindquist coordinates.

    Parameters
    ----------
    kerr_schild_coords : jax.Array
        Kerr-Schild coordinates `(t, x, y, z)`.
    M : float
        Mass of the black hole.
    a : float
        Spin parameter of the black hole.

    Returns
    -------
    jax.Array
        Boyer-Lindquist coordinates `(t, r, θ, φ)`.

    Notes
    -----
    - The returned The polar angle 'θ' is modified in Boyer-Lindquist (not to be confused with the spherical polar angle θ).
    - Useful for interpreting quantities in the standard Boyer-Lindquist chart.
    """
    t_ks, x_ks, y_ks, z_ks = kerr_schild_coords
    w = x_ks**2 + y_ks**2 + z_ks**2 - a**2
    r_bl = jnp.sqrt(0.5 * (w + jnp.sqrt(w**2 + 4 * a**2 * z_ks**2)))
    t_bl = t_ks - (M * jnp.log(jnp.abs(r_bl**2 - 2 * M * r_bl + a**2)) + M**2 / jnp.sqrt(M**2 - a**2) * jnp.log(jnp.abs((r_bl - (M + jnp.sqrt(M**2 - a**2))) / (r_bl - (M - jnp.sqrt(M**2 - a**2))))))
    theta_bl = jnp.arccos(z_ks / r_bl)
    phi_bl = (jnp.arctan2(y_ks, x_ks) - jnp.arctan2(a, r_bl) - a / (2 * jnp.sqrt(M**2 - a**2)) * jnp.log(jnp.abs((r_bl - (M + jnp.sqrt(M**2 - a**2))) / (r_bl - (M - jnp.sqrt(M**2 - a**2)))))) % (2 * jnp.pi)

    return jnp.stack([t_bl, r_bl, theta_bl, phi_bl], axis=-1)

def boyer_lindquist_to_eddington_finkelstein(
    boyer_lindquist_coords: jax.Array, 
    M: float, 
    a: float
) -> jax.Array: 
    """
    Converts Boyer-Lindquist coordinates to ingoing Eddington-Finkelstein (IEF) coordinates.

    Parameters
    ----------
    boyer_lindquist_coords : jax.Array
        Boyer-Lindquist coordinates `(t, r, θ, 𝜙̃)`.
    M : float
        Mass of the black hole.
    a : float
        Spin parameter of the black hole.

    Returns
    -------
    jax.Array
        Ingoing Eddington-Finkelstein (IEF) coordinates `(v, r, θ, 𝜙̃)`.

    Notes
    -----
    - The transformation uses the polar angle θ and modified azimuthal angle 𝜙̃.
    - IEF coordinates are horizon-penetrating and suitable for numerical simulations near the horizon.
    """ 
    r_plus, r_minus = M + jnp.sqrt(M**2 - a**2), M - jnp.sqrt(M**2 - a**2)  
    
    def r_star_fn(r1, r2):
        def tortoise_coordinates(r):
            return (
                r
                + (2*M*r_plus / (r_plus - r_minus)) * jnp.log(jnp.abs(r - r_plus))
                - (2*M*r_minus / (r_plus - r_minus)) * jnp.log(jnp.abs(r - r_minus))
            )
        return tortoise_coordinates(r2) - tortoise_coordinates(r1)
    
    def phi_star_fn(r1: float, r2: float) -> float:
        return (a / (r_plus - r_minus)) * (
            jnp.log(jnp.abs(r2 - r_plus)) - jnp.log(jnp.abs(r2 - r_minus))
            - jnp.log(jnp.abs(r1 - r_plus)) + jnp.log(jnp.abs(r1 - r_minus))
        )
    r_ref = 100.0*M # hardcoded as some reference radius. Ideally can be changed as per required 
    # coordinate transformations 
    def bl_to_ef_coords(
        bl_coords: jax.Array
    ) -> jax.Array:
        t, r, theta, phi = bl_coords 
        r_star = r_star_fn(r, r_ref)
        phi_star = phi_star_fn(r, r_ref) 
        v, r, theta, phi_tilde = t + r_star, r, theta, phi + phi_star 
        return jnp.array([v, r, theta, phi_tilde])
    
    return bl_to_ef_coords(bl_coords=boyer_lindquist_coords)

def eddington_finkelstein_to_boyer_lindquist(
    ief_coords: jax.Array, 
    M: float, 
    a: float
) -> jax.Array: 
    """
    Converts ingoing Eddington-Finkelstein (IEF) coordinates to Boyer-Lindquist coordinates.

    Parameters
    ----------
    ief_coords : jax.Array
        Ingoing Eddington-Finkelstein coordinates `(v, r, θ, 𝜙̃)`.
    M : float
        Mass of the black hole.
    a : float
        Spin parameter of the black hole.

    Returns
    -------
    jax.Array
        Boyer-Lindquist coordinates `(t, r, θ, 𝜙̃)`.

    Notes
    -----
    - The polar angle θ and modified azimuthal angle 𝜙̃ remain unchanged during the transformation.
    - Useful for mapping horizon-penetrating data back to the Boyer-Lindquist chart for analysis.
    """
    r_plus, r_minus = M + jnp.sqrt(M**2 - a**2), M - jnp.sqrt(M**2 - a**2)  
    
    def r_star_fn(r1: float, r2: float) -> float:
        def tortoise_coordinates(r):
            return (
                r
                + (2*M*r_plus / (r_plus - r_minus)) * jnp.log(jnp.abs(r - r_plus))
                - (2*M*r_minus / (r_plus - r_minus)) * jnp.log(jnp.abs(r - r_minus))
            )
        return tortoise_coordinates(r2) - tortoise_coordinates(r1)
    
    def phi_star_fn(r1: float, r2: float) -> float:
        return (a / (r_plus - r_minus)) * (
            jnp.log(jnp.abs(r2 - r_plus)) - jnp.log(jnp.abs(r2 - r_minus))
            - jnp.log(jnp.abs(r1 - r_plus)) + jnp.log(jnp.abs(r1 - r_minus))
        )
    r_ref = 100.0*M # hardcoded as some reference radius. Ideally can be changed as per required 
    def ef_to_bl_coords(ief_coords: jax.Array) -> jax.Array:
        v, r, theta, phi_tilde = ief_coords 
        r_star = r_star_fn(r, r_ref)
        phi_star = phi_star_fn(r, r_ref) 
        print(phi_star)
        t, r, theta, phi = v - r_star, r, theta, phi_tilde - phi_star
        return jnp.array([t, r, theta, phi])
    
    return ef_to_bl_coords(ef_coords=ief_coords)

if __name__ == '__main__': 
    jax.config.update("jax_enable_x64", True)
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
