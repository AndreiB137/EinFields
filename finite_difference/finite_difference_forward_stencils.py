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
from typing import Callable
import jax
import jax.numpy as jnp  

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#          Finite differencing forward stencils for different orders
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def jacobian_fd_first_order_stencil(
    coords: jax.Array, 
    h: float, 
    tensor_func: Callable[[jax.Array], jax.Array]
) -> jax.Array: 
    """
    Computes the Jacobian of a tensor-valued function using a forward finite-difference stencil.

    Approximates:
        J_{μ,ρ} ≈ [f(x + h * e_ρ) - f(x)] / h

    Parameters
    ----------
    coords : jax.Array
        A `(4,)` JAX array representing the spacetime coordinates 
        (e.g., (t, r, 𝜗, 𝜙) at which to compute the Jacobian.
    h : float
        The finite-difference step size.
    tensor_func : Callable[[jax.Array], jax.Array]
        A tensor-valued function `f: ℝ⁴ → ℝ^n` whose Jacobian with respect to the
        spacetime coordinates is to be computed.

    Returns
    -------
    jax.Array
        A `(4, ...)` JAX array containing the approximate Jacobian, where the
        first axis corresponds to the partial derivatives with respect to
        each coordinate direction, and the remaining shape matches the output
        shape of `tensor_func(coords)`.

    Notes
    -----
    - Implements the first-order forward finite-difference stencil.
    """
    x0, x1, x2, x3 = coords 
    gradient0 = (1.0/h)*(tensor_func(jnp.array([x0+1.0*h, x1, x2, x3])) - tensor_func(jnp.array([x0, x1, x2, x3])))
    gradient1 = (1.0/h)*(tensor_func(jnp.array([x0, x1+1.0*h, x2, x3])) - tensor_func(jnp.array([x0, x1, x2, x3])))
    gradient2 = (1.0/h)*(tensor_func(jnp.array([x0, x1, x2+1.0*h, x3])) - tensor_func(jnp.array([x0, x1, x2, x3])))
    gradient3 = (1.0/h)*(tensor_func(jnp.array([x0, x1, x2, x3+1.0*h])) - tensor_func(jnp.array([x0, x1, x2, x3])))

    return jnp.stack((gradient0, gradient1, gradient2, gradient3), axis=0)

def jacobian_fd_second_order_stencil(
    coords: jax.Array, 
    h: float, 
    tensor_func: Callable[[jax.Array], jax.Array]
) -> jax.Array: 
    """
    Computes the Jacobian of a tensor-valued function using a second-order forward finite-difference stencil.

 
    Parameters
    ----------
    coords : jax.Array
        A `(4,)` JAX array representing the spacetime coordinates
        (e.g., (t, r, 𝜗, 𝜙̃)) at which to compute the Jacobian.
    h : float
        The finite-difference step size.
    tensor_func : Callable[[jax.Array], jax.Array]
        A tensor-valued function `f: ℝ⁴ → ℝ^n` whose Jacobian with respect to the
        spacetime coordinates is to be computed.

    Returns
    -------
    jax.Array
        A `(4, ...)` JAX array containing the approximate Jacobian, where the
        first axis corresponds to the partial derivatives with respect to
        each coordinate direction, and the remaining shape matches the output
        shape of `tensor_func(coords)`.

    Notes
    -----
    - Implements a Second-order stencil.
    """ 
    x0, x1, x2, x3 = coords 
    gradient0 = -1.0/(2.0*h)*(tensor_func(jnp.array([x0+2*h, x1, x2, x3])) - 4.0*tensor_func(jnp.array([x0+h, x1, x2, x3])) + 3.0*tensor_func(jnp.array([x0, x1, x2, x3])))
    gradient1 = -1.0/(2.0*h)*(tensor_func(jnp.array([x0, x1+2*h, x2, x3])) - 4.0*tensor_func(jnp.array([x0, x1+h, x2, x3])) + 3.0*tensor_func(jnp.array([x0, x1, x2, x3])))
    gradient2 = -1.0/(2.0*h)*(tensor_func(jnp.array([x0, x1, x2+2*h, x3])) - 4.0*tensor_func(jnp.array([x0, x1, x2+h, x3])) + 3.0*tensor_func(jnp.array([x0, x1, x2, x3])))
    gradient3 = -1.0/(2.0*h)*(tensor_func(jnp.array([x0, x1, x2, x3+2*h])) - 4.0*tensor_func(jnp.array([x0, x1, x2, x3+h])) + 3.0*tensor_func(jnp.array([x0, x1, x2, x3])))

    return jnp.stack((gradient0, gradient1, gradient2, gradient3), axis=0)

def jacobian_fd_third_order_stencil(
    coords: jax.Array, 
    h: float, 
    tensor_func: Callable[[jax.Array], jax.Array]
) -> jax.Array: 
    """
    Computes the Jacobian of a tensor-valued function using a third-order forward finite-difference stencil.

    Approximates:
        J_{μ,ρ} ≈ [-11f(x) + 18f(x+h) - 9f(x+2h) + 2f(x+3h)] / (6h)

    Parameters
    ----------
    coords : jax.Array
        A `(4,)` JAX array representing the spacetime coordinates
        (e.g., (t, r, 𝜗, 𝜙̃)) at which to compute the Jacobian.
    h : float
        The finite-difference step size.
    tensor_func : Callable[[jax.Array], jax.Array]
        A tensor-valued function `f: ℝ⁴ → ℝ^n` whose Jacobian with respect to the
        spacetime coordinates is to be computed.

    Returns
    -------
    jax.Array
        A `(4, ...)` JAX array containing the approximate Jacobian.

    Notes
    -----
    - Third-order accurate forward stencil, useful when backward evaluations are not feasible (e.g., near boundaries).
    """
    x0, x1, x2, x3 = coords 
    gradient0 = (1/h)*((-11.0/6.0)*tensor_func(jnp.array([x0, x1, x2, x3])) + 3.0*tensor_func(jnp.array([x0+h, x1, x2, x3])) - (3.0/2.0)*tensor_func(jnp.array([x0+2*h, x1, x2, x3])) + (1.0/3.0)*tensor_func(jnp.array([x0+3.0*h, x1, x2, x3])))
    gradient1 = (1/h)*((-11.0/6.0)*tensor_func(jnp.array([x0, x1, x2, x3])) + 3.0*tensor_func(jnp.array([x0, x1+h, x2, x3])) - (3.0/2.0)*tensor_func(jnp.array([x0, x1+2*h, x2, x3])) + (1.0/3.0)*tensor_func(jnp.array([x0, x1+3.0*h, x2, x3])))
    gradient2 = (1/h)*((-11.0/6.0)*tensor_func(jnp.array([x0, x1, x2, x3])) + 3.0*tensor_func(jnp.array([x0, x1, x2+h, x3])) - (3.0/2.0)*tensor_func(jnp.array([x0, x1, x2+2*h, x3])) + (1.0/3.0)*tensor_func(jnp.array([x0, x1, x2+3.0*h, x3])))
    gradient3 = (1/h)*((-11.0/6.0)*tensor_func(jnp.array([x0, x1, x2, x3])) + 3.0*tensor_func(jnp.array([x0, x1, x2, x3+h])) - (3.0/2.0)*tensor_func(jnp.array([x0, x1, x2, x3+2*h])) + (1.0/3.0)*tensor_func(jnp.array([x0, x1, x2, x3+3.0*h])))

    return jnp.stack((gradient0, gradient1, gradient2, gradient3), axis=0)

def jacobian_fd_fourth_order_stencil(
    coords: jax.Array, 
    h: float, 
    tensor_func: Callable[[jax.Array], jax.Array]
) -> jax.Array: 
    """
    Computes the Jacobian of a tensor-valued function using a fourth-order forward finite-difference stencil.

    Approximates:
        J_{μ,ρ} ≈ [ -25f(x) + 48f(x+h) - 36f(x+2h) + 16f(x+3h) - 3f(x+4h) ] / (12h)

    Parameters
    ----------
    coords : jax.Array
        A `(4,)` JAX array representing the spacetime coordinates
        (e.g., (t, r, 𝜗, 𝜙̃)) at which to compute the Jacobian.
    h : float
        The finite-difference step size.
    tensor_func : callable
        A tensor-valued function `f: ℝ⁴ → ℝ^n` whose Jacobian with respect to the
        spacetime coordinates is to be computed.

    Returns
    -------
    jax.Array
        A `(4, ...)` JAX array containing the approximate Jacobian, where the first axis
        corresponds to the partial derivatives with respect to each coordinate direction,
        and the remaining shape matches the output shape of `tensor_func(coords)`.

    Notes
    -----
    - Fourth-order forward difference stencil for higher-accuracy derivatives in boundary-adjacent regions.
    - Useful for metric derivative calculations in numerical relativity pipelines when backward points are unavailable.
    - Requires five function evaluations per coordinate direction.
    """ 
    x0, x1, x2, x3 = coords 
    gradient0 = -1.0/(12.0*h)*(25.0*tensor_func(jnp.array([x0, x1, x2, x3])) - 48.0*tensor_func(jnp.array([x0+1.0*h, x1, x2, x3])) + 36.0*tensor_func(jnp.array([x0+2.0*h, x1, x2, x3])) -16.0*tensor_func(jnp.array([x0+3.0*h, x1, x2, x3])) + 3.0*tensor_func(jnp.array([x0+4.0*h, x1, x2, x3])))
    gradient1 = -1.0/(12.0*h)*(25.0*tensor_func(jnp.array([x0, x1, x2, x3])) - 48.0*tensor_func(jnp.array([x0, x1+1.0*h, x2, x3])) + 36.0*tensor_func(jnp.array([x0, x1+2.0*h, x2, x3])) -16.0*tensor_func(jnp.array([x0, x1+3.0*h, x2, x3])) + 3.0*tensor_func(jnp.array([x0, x1+4.0*h, x2, x3])))
    gradient2 = -1.0/(12.0*h)*(25.0*tensor_func(jnp.array([x0, x1, x2, x3])) - 48.0*tensor_func(jnp.array([x0, x1, x2+1.0*h, x3])) + 36.0*tensor_func(jnp.array([x0, x1, x2+2.0*h, x3])) -16.0*tensor_func(jnp.array([x0, x1, x2+3.0*h, x3])) + 3.0*tensor_func(jnp.array([x0, x1, x2+4.0*h, x3])))
    gradient3 = -1.0/(12.0*h)*(25.0*tensor_func(jnp.array([x0, x1, x2, x3])) - 48.0*tensor_func(jnp.array([x0, x1, x2, x3+1.0*h])) + 36.0*tensor_func(jnp.array([x0, x1, x2, x3+2.0*h])) -16.0*tensor_func(jnp.array([x0, x1, x2, x3+3.0*h])) + 3.0*tensor_func(jnp.array([x0, x1, x2, x3+4.0*h])))

    return jnp.stack((gradient0, gradient1, gradient2, gradient3), axis=0)

def jacobian_fd_sixth_order_stencil(
    coords: jax.Array, 
    h: float, 
    tensor_func: Callable[[jax.Array], jax.Array]
) -> jax.Array: 
    """
    Computes the Jacobian of a tensor-valued function using a sixth-order forward finite-difference stencil.

    Approximates:
        J_{μ,ρ} ≈ [ -(49/20)f(x) + 6f(x+h) - (15/2)f(x+2h) + (20/3)f(x+3h)
                     - (15/4)f(x+4h) + (6/5)f(x+5h) - (1/6)f(x+6h) ] / (60h)

    Parameters
    ----------
    coords : jax.Array
        A `(4,)` JAX array representing the spacetime coordinates
        (e.g., (t, r, 𝜗, 𝜙̃)) at which to compute the Jacobian.
    h : float
        The finite-difference step size.
    tensor_func : callable
        A tensor-valued function `f: ℝ⁴ → ℝ^n` whose Jacobian with respect to the
        spacetime coordinates is to be computed.

    Returns
    -------
    jax.Array
        A `(4, ...)` JAX array containing the approximate Jacobian, where the first axis
        corresponds to the partial derivatives with respect to each coordinate direction,
        and the remaining shape matches the output shape of `tensor_func(coords)`.

    Notes
    -----
    - Sixth-order forward difference stencil providing very high accuracy for smooth functions.
    - Requires seven function evaluations per coordinate direction.
    - Avoids backward evaluations, making it suitable for causally constrained evolution near horizon or outer boundary regions.
    """ 
    x0, x1, x2, x3 = coords 
    gradient0 = (1.0/h)*((-49.0/20.0)*tensor_func(jnp.array([x0, x1, x2, x3])) + 6.0*tensor_func(jnp.array([x0+1.0*h, x1, x2, x3])) - (15.0/2.0)*tensor_func(jnp.array([x0+2.0*h, x1, x2, x3])) + (20.0/3.0)*tensor_func(jnp.array([x0+3.0*h, x1, x2, x3])) - (15.0/4.0)*tensor_func(jnp.array([x0+4.0*h, x1, x2, x3])) + (6.0/5.0)*tensor_func(jnp.array([x0+5.0*h, x1, x2, x3])) - (1.0/6.0)*tensor_func(jnp.array([x0+6.0*h, x1, x2, x3])))
    gradient1 = (1.0/h)*((-49.0/20.0)*tensor_func(jnp.array([x0, x1, x2, x3])) + 6.0*tensor_func(jnp.array([x0, x1+1.0*h, x2, x3])) - (15.0/2.0)*tensor_func(jnp.array([x0, x1+2.0*h, x2, x3])) + (20.0/3.0)*tensor_func(jnp.array([x0, x1+3.0*h, x2, x3])) - (15.0/4.0)*tensor_func(jnp.array([x0, x1+4.0*h, x2, x3])) + (6.0/5.0)*tensor_func(jnp.array([x0, x1+5.0*h, x2, x3])) - (1.0/6.0)*tensor_func(jnp.array([x0, x1+6.0*h, x2, x3])))
    gradient2 = (1.0/h)*((-49.0/20.0)*tensor_func(jnp.array([x0, x1, x2, x3])) + 6.0*tensor_func(jnp.array([x0, x1, x2+1.0*h, x3])) - (15.0/2.0)*tensor_func(jnp.array([x0, x1, x2+2.0*h, x3])) + (20.0/3.0)*tensor_func(jnp.array([x0, x1, x2+3.0*h, x3])) - (15.0/4.0)*tensor_func(jnp.array([x0, x1, x2+4.0*h, x3])) + (6.0/5.0)*tensor_func(jnp.array([x0, x1, x2+5.0*h, x3])) - (1.0/6.0)*tensor_func(jnp.array([x0, x1, x2+6.0*h, x3])))
    gradient3 = (1.0/h)*((-49.0/20.0)*tensor_func(jnp.array([x0, x1, x2, x3])) + 6.0*tensor_func(jnp.array([x0, x1, x2, x3+1.0*h])) - (15.0/2.0)*tensor_func(jnp.array([x0, x1, x2, x3+2.0*h])) + (20.0/3.0)*tensor_func(jnp.array([x0, x1, x2, x3+3.0*h])) - (15.0/4.0)*tensor_func(jnp.array([x0, x1, x2, x3+4.0*h])) + (6.0/5.0)*tensor_func(jnp.array([x0, x1, x2, x3+5.0*h])) - (1.0/6.0)*tensor_func(jnp.array([x0, x1, x2, x3+6.0*h])))

    return jnp.stack((gradient0, gradient1, gradient2, gradient3), axis=0)

if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)

