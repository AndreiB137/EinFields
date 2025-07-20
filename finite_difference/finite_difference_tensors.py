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

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#           Finite-difference application on tensor fields (+ helper functions)
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

import numpy as np 
import jax
import jax.numpy as jnp  
import jaxlib
from typing import Callable

def metric_jacobian_forwarddiff(
    metric: Callable[[jax.Array], jax.Array], 
    fd_stencil: Callable[[jax.Array], jax.Array]
) -> Callable[[jax.Array, float], jax.Array]: 
    """
    Compute the metric Jacobian at a given point p on the manifold.

    Parameters
    ----------
    metric : Callable
            Functional form of metric tensor evaluated at point p on the manifold.
        
    fd_stencil: Callable 
        n-th order FD stencil of choice for performing finite-difference on tensors
    
    h: float 
        infinitesimal element for finite-difference

    Returns
    -------
    metric_jac : jaxlib.xla_extension.PjitFunction
        Functional form of Christoffel symbol for a particular choice of metric and Forward Diff FD stencil of order n.
    """
    @jax.jit
    def metric_jac(p: jax.Array, h: float) -> Callable: 
        """
        Compute the metric Jacobian at a given point p on the manifold.

        Parameters
        ----------
        p : jax.Array
            point p on the manifold at which the metric jacobian is to be evaluated on.
                
        h: float 
            infinitesimal element for finite-difference

        Returns
        -------
        jacobian_metric : jax.Array
           metric Jacobian evaluated at point p for the choice of h of a forward fd-stencil
        """
        jacobian_metric=fd_stencil(p, h, metric)
        return jacobian_metric 
    
    return metric_jac

def christoffel_symbol_forwarddiff( 
    metric: Callable[[jax.Array], jax.Array], 
    fd_stencil: Callable[[jax.Array], jax.Array]
) -> Callable[[jax.Array, float], jax.Array]:
    """
    Compute the Christoffel symbol at a given point p on the manifold.

    Parameters
    ----------
    metric : Callable
            Functional form of metric tensor evaluated at point p on the manifold.
        
    fd_stencil: Callable 
        n-th order FD stencil of choice for performing finite-difference on tensors
    
    h: float 
        infinitesimal element for finite-difference

    Returns
    -------
    christoffel : Callable
        Functional form of Christoffel symbol for a particular choice of metric and Forward Diff FD stencil of order n.
    """
    @jax.jit
    def christoffel(p: jax.Array, h: float) -> Callable:
        """
        Parameters
        ----------
        p : jax.Array
        Coordinates of the point p on the manifold at which the Christoffel symbol is evaluated.

        Returns
        -------
        christoffel : jaxlib.xla_extension.PjitFunction
            Functional form of Christoffel symbol for a particular choice of metric and Forward Diff FD stencil of order n at point p.
        """
        g=metric(p)
        inv_g = jnp.linalg.inv(g)
        dg=fd_stencil(p, h, metric)
        dgamma=jnp.einsum('jki -> kij', dg) + jnp.einsum('ikj -> kij', dg) - dg   
        dgamma=0.5*jnp.einsum('mk,kij -> mij', inv_g, dgamma)
        return dgamma 
    
    return christoffel

def christoffel_symbols_jac_forwarddiff(
    christoffel: Callable[[jax.Array], jax.Array], 
    fd_stencil:  Callable[[jax.Array], jax.Array]
) -> Callable[[jax.Array, float], Callable]: 
    """
    Compute the Jacobian of Christoffel symbols at a given point p on the manifold.
    
    Parameters
    ----------
    christoffel_fd : Callable
        Functional form of christoffel symbol evaluated at point p on the manifold.
        
    fd_stencil: Callable 
        n-th order FD stencil of choice for performing finite-difference on tensors
    
    h: float 
        infinitesimal element for finite-difference
    
    Returns
    -------
    christoffel_jac : jaxlib.xla_extension.PjitFunction
        Functional form of Christoffel symbol Jacobian for a particular choice of metric and Forward Diff stencil of order n at point p.
    """     
    @jax.jit
    def christoffel_jac(p:jax.Array, h: float) -> Callable: 
        """
        Parameters
        ----------
        p : jax.Array
        Coordinates of the point p on the manifold at which the Christoffel symbol is evaluated.

        Returns
        -------
        dgamma : Callable
            Functional form of Christoffel symbol Jacobian for a particular choice of christoffel symbol and Forward diff stencil of order n at point p.
        """
        christoffel_fn = lambda p: christoffel(p, h)
        dgamma = fd_stencil(p, h, christoffel_fn)
        return dgamma
    
    return christoffel_jac

def levi_civita_connection_custom(
    coords: jax.Array, 
    tensor: Callable[[jax.Array], jax.Array], 
    christoffel: Callable[[jax.Array], jax.Array], 
    rank: tuple) -> jax.Array: 
    """
    A function used to calculate covariant derivatives of general tensors of rank (r, s), where r is the number
    of upper indices and s of lower indices. The general rule involves differentiating the tensor, then followed by
    adding various contractions of the christoffel symbols with the upper and lower indices of the tensor. For a 
    contraction with an upper index there is a + sign and a - sign for a lower index. This function allows custom christoffel symbols 
    that is specified by the end-user, namely an FD stencil version or AD version or analytic christoffel version 

    Parameters
    ---------
    coords: jax.Array
        Coordinates of the point at which to cumpute the covariant derivative.

    tensor: Callable
        A python function with a single input corresponding to the coordinates where to output the tensor returned by
        tensor_.

    christoffel: Callable 
        Functional form of the symmetric Levi-Civita components 

    rank : jax.Array/np.ndarray/list/tuple
        A tuple having only values of zero and one corresponding to either lower or upper index. If the tensor is a
        scalar, then len(rank) = 0.
        0 - lower index (covariant)
        1 - upper index (contravariant)
        For e.g: scalar field: (), covariant vector field: (0), contravariant vector field: (1)  covariant metric field: (0, 0), contravariant metric field (1, 1)
    Returns
    -------
    compute_covariant_derivative : jaxlib.xla_extension.PjitFunction(jax.Array) == jax.Array
        Functional form of the covariant derivative with the symmetric affine connection (Levi-Civita conenction) evaluated at the given point p on the mainfold.        
    """
    n = len(rank)
    if n == 0:
        return jax.jacfwd(tensor)(coords)
    
    dtensor = jax.jacfwd(tensor)
    init_index = np.arange(n + 1)
    permuted_index = (init_index + n) % (n + 1)
    tensor = tensor(coords)
    dtensor = jnp.einsum(dtensor(coords), init_index, permuted_index)

    christoffel = christoffel(coords)

    tensor_index = np.arange(n)
    result_index = np.concatenate((np.array([n + 1]), tensor_index))
    for i in range(len(rank)):
        result_index[i + 1] = n
        if rank[i] == 1:
            christoffel_index = np.array([n, i, n + 1])
            dtensor += jnp.einsum(christoffel, christoffel_index, tensor, tensor_index, result_index)
        else:
            christoffel_index = np.array([i, n, n + 1])
            dtensor -= jnp.einsum(christoffel, christoffel_index, tensor, tensor_index, result_index)
            
        result_index[i + 1] = i

    return dtensor

def cov_riemann_tensor_static_custom(
    metric: Callable[[jax.Array], jax.Array], 
    christoffel: Callable[[jax.Array], jax.Array], 
    christoffel_jac: Callable[[jax.Array], jax.Array]
) -> jax.Array:
    """
    Compute the Riemann curvature tensor

    Parameters
    ----------
    christoffel: jax.Array  
        Christoffel symbols of type {p; qr} evaluated over the entire set of grid (collocation) points: shape (4, 4, 4). 

    christoffel_jac: jax.Array 
        Christoffel Jacobian of type {p; qrs} evaluated over the entire set of grid (collocation) points: shape (4, 4, 4, 4)
    Returns
    -------
    curvature tensor : jax.Array
        Riemann curvature tensor of rank {1; 3} evaluated over the entire set of grid (collocation) points: shape (4, 4, 4, 4).
    """
    @jax.jit
    def cov_riem(coord: jax.Array) -> jax.Array:
        curvature_tensor = (
            jnp.einsum('iljk -> lkij', christoffel_jac(coord))
            - jnp.einsum('jlik -> lkij', christoffel_jac(coord))
            + jnp.einsum('lip, pjk -> lkij', christoffel(coord), christoffel(coord))
            - jnp.einsum('ljp, pik -> lkij', christoffel(coord), christoffel(coord))
        )
        g = metric(coord) 
        curvature_tensor = jnp.einsum('mjkl,im -> ijkl', curvature_tensor, g)
        return curvature_tensor

    return cov_riem

def bianchi_identity_second(
    antisymmetric_cov_rank_four_tensor: Callable[[jax.Array], jax.Array], 
    christoffel: Callable[[jax.Array], jax.Array]
) -> jax.Array:
    """
    The second Bianchi identity asserts that for a covariant antisymmetric curvature tensor A_{abmn; l} + A_{ablm; n} + A_{abnl;m} = 0
        Parameters
    ----------
    antisymmetric_cov_rank_four_tensor : Callable
        skew-symmetric field strength tensor such as Riemann tensor, Weyl tensor of rank (0, 4) 
    
    Returns
    -------
    bianchi_conservation : jaxlib.xla_extension.PjitFunction(jax.Array) == jax.Array
        Functional form of the bianchi identity two
    """
    @jax.jit
    def bianchi_conservation(p: jax.Array) -> jax.Array: 
        antisymmetric_tensor_jac = levi_civita_connection_custom(p, antisymmetric_cov_rank_four_tensor, christoffel, (0, 0, 0, 0))
        return antisymmetric_tensor_jac + jnp.einsum('ablmn -> abmnl', antisymmetric_tensor_jac) + jnp.einsum('abnlm -> abmnl', antisymmetric_tensor_jac)
    return bianchi_conservation 

if __name__ == '__main__': 
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)

