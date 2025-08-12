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
import numpy as np 
from typing import Callable
from .levi_civita_symbol.levi_civita_symbol import levi_civita_symbol

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#             Main Differential Geometry and Tensor_calculus package
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

class DifferentialGeometry: 
    """
        Introduction
        ------------
        A computational toolkit for differential geometry operations in curved spacetime.

        This class computes geometric quantities from metric tensors using automatic differentiation.
        Common use cases:
        - Geometric analysis of curved manifolds
        - General relativity applications
        - Testing neural representations of spacetime metrics
        - Operations on tensors in curved spaces

        The `DifferentialGeometry` class serves as the primary interface, providing
        automatic differentiation-based computation of curvature tensors, connections,
        and geometric invariants from a metric tensor specification. 

        The primary goal is to provide ways to test the training of EinFields, our newly developed neural representation of the metric
        tensor, with all the objects general relativity has to offer. Although the metric tensor in general relativity is the main
        object of interest, the functionalities of the class are not limited to only four dimensional spacetime, but can easily be used 
        in other dimensions as well. Having said this, the class is designed to work equally well in a more general setting whether it is
        a Riemannian or a pseudo-Riemannian manifold.

        General relativity
        ------------------
        Einstein questioned whether the gravitational acceleration is somehow an "illusion". In a vacuum, where the gravitational field is 
        constant, bodies fall at the same rate and they would not be able to tell whether there is a force or not. This very simplified view 
        actually describes the essence of general relativity in "small" regions of spacetime, or in other words, in a local inertial frame. 
        In reality, if two bodies are sufficiently far apart and released at some distance above the surface of the Earth, they will start 
        to approach each other. 
        
        This leads to the idea that if freely falling bodies are following geodesics, then this approach (or divergence)
        is due to non-local effects, namely non-constant gravitational field which translates into the geometric concept of geodesic deviation, 
        described by the Riemann curvature tensor. Then, general relativity becomes a description of the geometry of spacetime, more exactly,
        a pseudo-Riemannian manifold with a metric tensor field encoding the effects of gravity. 
        
        All Einstein's ideas are converted into
        the mathematical description of differential geometry. Locally, the metric tensor can be approximated by the Minkowski metric, which 
        reduces the problem to flat spacetimes, and so bodies can't "feel" gravity from their local perspective. However, the variation of the
        gravitational field (or tidal forces) translated as the global curvature of the spacetime manifold introduces non-local effects. Moreover,
        the theory is independent of the coordinate system used to describe the spacetime in the sense that physical laws aren't modified.

        Class overview
        --------------
        As it happens for a Riemannian or pseudo-Riemannian manifold, all the geometric quantities can be derived from the metric tensor. Thus,
        the `DifferentialGeometry` class follows a linear workflow, where everything starts from the metric tensor and proceeds forward. 
        
        The key 
        computational pipeline:

        `metric` → `christoffel_symbols` → `riemann_tensor` → `ricci_tensor` → `ricci_scalar`.

        To respect JAX's functional programming paradigm, the `metric` is a Python function (def, class method, static method or lambda),
        which takes only one input, a `jax.Array` representing the coordinates of a point on the manifold, and returns a `jax.Array` representing the 
        metric tensor at that point, a 4x4 matrix in the case of four-dimensional spacetime. The `metric` function is not allowed to be wrapped
        in `jax.vmap` before passing it to the class.

        All tensor operations are implemented with `jnp.einsum` for maximum performance from the various optimizations einsum provides. A list of
        available and most important differential geometry quantities is provided below:
        - `christoffel_symbols`
        - `riemann_tensor`
        - `ricci_tensor`
        - `ricci_scalar`
        - `weyl_curvature_tensor`
        - `einstein_tensor`
        - `kretschmann_invariant`
        - `tetrad`

        Most of the methods (apart from the static ones) have only the coordinates `jax.Array` as input, making it easy to vectorize and compute
        the quantities over a batch of coordinates. For more details on the differential geometry quantities, please refer to our paper in the
        appendix section [...].

        Operations on tensors
        ---------------------
        Two essential geometric operations are the `lie_derivative` and the `levi_civita_connection` (covariant derivative).
        Because the Lie derivative is independent of connection and thus of the metric tensor it is a static method, while the covariant derivative
        is a method of the class, as it requires the Christoffel symbols from the metric. 

        These can be called as follows:

        ```
            from .../tensor_calculus import DifferentialGeometry as diffgeo
        
            coords = jnp.array([0.0, 4.0, jnp.pi/2, 0.0])  # Example coordinates
            vector_field = lambda coords: coords

            schwarzschild_geo = diffgeo(schwarzschild_metric)

            lie_derivative = diffgeo.lie_derivative(coord=coords, 
            vector_field=vector_field, 
            tensor=schwarzschild_metric, 
            rank=[0, 0])

            covariant_derivative = schwarzschild_geo.levi_civita_connection(coords=coords, 
            tensor=schwarzschild_metric, rank=[0, 0])

        ```
        

        Parameters
        ----------
        metric : Callable[[jax.Array], jax.Array]
            A function that takes a `jax.Array` of coordinates and returns the metric tensor at that point.
            The metric tensor must be a square matrix of shape (dim, dim), where dim is the dimension of the manifold.
            For example, if the jax.Array has shape (dim,), then the metric function should return a jax.Array of shape (dim, dim).

        Examples
        --------
        >>> # Schwarzschild black hole
        >>> from gr_package.gr_problems.schwarzschild import schwarzschild_metric_spherical
        >>> schwarzschild_fn = lambda coords: schwarzschild_metric_spherical(coords, M=1.0)
        >>> schwarzschild_geo = DifferentialGeometry(metric_fn)
        >>> coords = jnp.array([0.0, 4.0, jnp.pi/2, 0.0])  # Outside horizon
        >>>  # Compute various geometric quantities
        >>> christoffel = schwarzschild_geo.christoffel_symbols(coords) # shape (4, 4, 4)
        >>> riemann_tensor = schwarzschild_geo.riemann_tensor(coords) # shape (4, 4, 4, 4)
        >>> ricci_tensor = schwarzschild_geo.ricci_tensor(coords) # shape (4, 4)
        >>> ricci_scalar = schwarzschild_geo.ricci_scalar(coords) # shape ()
        >>> weyl_tensor = schwarzschild_geo.weyl_curvature_tensor(coords) # shape (4, 4, 4, 4)
        >>> kretschmann_scalar = schwarzschild_geo.kretschmann_invariant(coords) # shape ()


        Future work
        -----------
        Although most operations on tensors don't require a complicated handling of the rank,
        there can be scenarios where it might be useful to compose some of this operations, 
        such as the composition of Levi-Civita connection on a tensor field multiple times 
        increases the covariant rank. For more flexibility and ease of use with this differential
        geometry perspective, we plan to implement a tensor field class which should, among other things,
        keep track of the rank. Moreover, the compatibility with JAX's functional programming paradigm
        will require the tensor field object have a `__call__` method, which will allow it to be used
        as a function that takes coordinates and returns the array representation at those coordinates.

    """
    def __init__(self, metric: Callable[[jax.Array], jax.Array]) -> None: 
        self.metric = metric

        if not callable(metric):
            raise TypeError("The metric must be a callable function that takes jax.Array as input and returns jax.Array as output.")
    
    def metric_jacobian(self, coords: jax.Array) -> jax.Array: 
        """
        Compute the Jacobian of the metric tensor at a given coordinate.
        The AD Jacobian has the derivative index ∂_{μ} in the last dimension of the array,
        while in physics it is always set as the first index. 

        Parameters
        ----------
        coords : jax.Array
            Coordinates of the point at which to compute the metric tensor Jacobian.
        
        Returns
        -------

        metric_derivative : jax.Array
            Metric tensor Jacobian ∂_{μ}g_{μν} evaluated at the given coordinate.

        """
        metric_derivative = jnp.transpose(jax.jacfwd(self.metric)(coords), [2, 0, 1])
        return metric_derivative
    
    def metric_hessian(self, coords: jax.Array) -> jax.Array: 
        """
        Compute the Hessian of the metric tensor at a given coordinate.
        The AD Hessian has the derivative indices ∂_{μ}∂_{ν} in the last dimensions of the array,
        while in physics it is always set as the first indices. 

        Parameters
        ----------

        coords: jax.Array
            Coordinates of the point at which to compute the metric tensor Hessian.

        Returns
        -------
        metric_hessian : jax.Array
            Metric tensor Hessian ∂_{μ}∂_{ν}g_{ρσ} evaluated at the given coordinate.

        """
        return jnp.transpose(jax.jacfwd(self.metric_jacobian)(coords), [3, 0, 1, 2])

    def christoffel_symbols(self, coords: jax.Array) -> jax.Array:
        """
        Compute the Christoffel symbols at a given coordinate.

        Note, Christoffel symbols don't form a tensor.

        Parameters
        ----------

        coords: jax.Array
            Coordinates of the point at which to compute the Christoffel symbols.
        
        Returns
        -------
        christoffel_symbols : jax.Array
            Christoffel symbols at the given coordinate. The first index is upper, the rest two indices are lower.
        """
        g = self.metric(coords)
        inv_g = jnp.linalg.inv(g)
        dg = self.metric_jacobian(coords)
        dgamma = jnp.einsum('jki -> kij', dg, precision=jax.lax.Precision.HIGHEST) + jnp.einsum('ikj -> kij', dg, precision=jax.lax.Precision.HIGHEST) - dg   
        dgamma = 0.5*jnp.einsum('mk,kij -> mij', inv_g, dgamma, precision=jax.lax.Precision.HIGHEST)
        return dgamma
    
    def christoffel_jac(self,coords: jax.Array) -> jax.Array: 
        """
        Compute the Jacobian of the Christoffel symbols at a given coordinate.
        The derivative index ∂_{μ} is the first of the array.

        Parameters
        ----------
        coords: jax.Array
            Coordinates of the point at which to compute the Christoffel symbols Jacobian.

        Returns
        -------
        christoffel_jacobian : jax.Array
            Christoffel symbols Jacobian ∂_{μ}Γ^{ν}_{ρσ} evaluated at the given coordinate.

        """
        return jnp.transpose(jax.jacfwd(self.christoffel_symbols)(coords), [3, 0, 1, 2])

    def riemann_tensor(self,coords: jax.Array) -> jax.Array:
        """
        Compute the Riemann curvature tensor at a given coordinate.

        Parameters
        ----------
        coords: jax.Array
            Coordinates of the point at which to compute the Riemann curvature tensor.
        
        Returns
        -------
        riemann_tensor : jax.Array
            Riemann curvature tensor of rank {1; 3} evaluated at the given coordinate. The first index is upper, the rest three indices are lower.

        """
        Gamma = self.christoffel_symbols(coords)
        dGamma = self.christoffel_jac(coords)
        
        Rlkij = DifferentialGeometry.riemann_tensor_static(Gamma, dGamma)
        return Rlkij
    
    
    @staticmethod
    def riemann_tensor_static(christoffel: jax.Array, christoffel_jac: jax.Array) -> jax.Array:
        """
        Compute the Riemann curvature tensor of rank {1; 3} given the Christoffel symbols and its Jacobian
        evaluated at some coordinate.

        Parameters
        ----------

        christoffel: jax.Array  
            Christoffel symbols of rank {1; 2} evaluated at some coordinate. 

        christoffel_jac: jax.Array 
            Christoffel Jacobian of rank {1; 2} evaluated at some coordinate,

        Returns
        -------

        curvature tensor : jax.Array
            Riemann curvature tensor of rank {1; 3} evaluated at some coordinate

        """
        curvature_tensor = (
            jnp.einsum('kilj -> ijkl', christoffel_jac, precision=jax.lax.Precision.HIGHEST)
            - jnp.einsum('likj -> ijkl', christoffel_jac, precision=jax.lax.Precision.HIGHEST)
            + jnp.einsum('ikp, plj -> ijkl', christoffel, christoffel, precision=jax.lax.Precision.HIGHEST)
            - jnp.einsum('ilp, pkj -> ijkl', christoffel, christoffel, precision=jax.lax.Precision.HIGHEST)
        )
        
        return curvature_tensor

    def covariant_riemann_tensor(self, coords: jax.Array) -> jax.Array: 
        """
        Computes the Riemann curvature tensor at a given coordinate with all indices down by contracting first upper index 
        with the metric tensor g: R_{ijkl} = g_{ip}*R^{p}_{jkl}.
        
        Parameters
        ----------
        coords : jax.Array
            Coordinates of the point at which to compute the covariant Riemann curvature tensor.
        
        Returns
        -------
        riemann_covariant : jax.Array
            Covariant Riemann curvature tensor of rank {0, 4} evaluated at the given coordinate
        """

        g = self.metric(coords)
        riemann_tensor = self.riemann_tensor(coords)
        riemann_covariant = jnp.einsum('mjkl,im -> ijkl', riemann_tensor, g)

        return riemann_covariant
    
    @staticmethod
    def covariant_riemann_tensor_static(metric_tensor: jax.Array, riemann_tensor: jax.Array) -> jax.Array: 
        """
        Compute the covariant Riemann curvature tensor with all indices down given the 
        metric tensor and the Riemann curvature tensor evaluated at some coordinate.

        Parameters
        ----------
        metric_tensor: jax.Array
            Metric tensor of rank {0, 2} evaluated at some coordinate.
            
        riemann_tensor: jax.Array
            Riemann curvature tensor of rank {1, 3} evaluated at some coordinate.
        
        Returns
        -------
        riemann_covariant: jax.Array
            Covariant Riemann curvature tensor of rank {0; 4} evaluated at some coordinate.    
        
        """

        return jnp.einsum('mjkl,im -> ijkl', riemann_tensor, metric_tensor)
    
    def weyl_curvature_tensor(self, coords: jax.Array) -> jax.Array:
        """Compute the Weyl tensor c_{ijkl} = R_{ijkl} + 1/(n-2) [R_{im} g_{kl} - R_{il} g_{km} + R_{kl} g_{im} - R_{km} g_{il}] + 1/(n-1)(n-2) R[g_{il}g_{km} - g_{im}g_{kl}] 
        to keep track of tidal forces at a given coordinate.
        
        Parameters
        ----------
        coords : jax.Array
            Coordinates of the point at which to compute the Weyl curvature tensor.
        
        Returns
        -------
        weyl tensor : jax.Array
            Weyl curvature tensor (tidal forces) of rank {0, 4} evaluated at the given coordinate.

        """

        g = self.metric(coords)
        ricci_scalar = self.ricci_scalar(coords)
        ricci_tensor = self.ricci_tensor(coords)
        dim = g.shape[-1]

        return self.covariant_riemann_tensor(coords) + 1 / ((dim - 2) * (dim - 1)) * ricci_scalar * (jnp.einsum('ik,jl -> ijkl', g, g) - jnp.einsum('il,jk -> ijkl', g, g)) + 1 / (dim - 2) * (jnp.einsum('il,jk -> ijkl', ricci_tensor, g) - jnp.einsum('ik,jl -> ijkl', ricci_tensor, g) + jnp.einsum('jk,il -> ijkl', ricci_tensor, g) - jnp.einsum('jl,ik -> ijkl', ricci_tensor, g))

    @staticmethod
    def weyl_curvature_tensor_static(metric_tensor: jax.Array, riemann_tensor: jax.Array) -> jax.Array:
        """
        Compute the Weyl tensor c_{ijkl} = R_{ijkl} + 1/(n-2) [R_{im} g_{kl} - R_{il} g_{km} + R_{kl} g_{im} - R_{km} g_{il}] + 1/(n-1)(n-2) R[g_{il}g_{km} - g_{im}g_{kl}]   
        given the metric tensor, the Riemann curvature tensor evaluated at some coordinate. Additionally,
        the computation is done with the help of the Kularni-Nomizu product.
        
        Parameters
        ----------
        metric_tensor: jax.Array
            Metric tensor of rank {0, 2} evaluated at some coordinate.
        
        riemann_tensor: jax.Array
            Riemann curvature tensor of rank {1, 3} evaluated at some coordinate.
        
        Returns
        -------
        weyl tensor : jax.Array
            Weyl curvature tensor (tidal forces) of rank {0, 4} evaluated at some coordinate.
        """
        g = metric_tensor
        riemann_covariant_tensor = jnp.einsum('pjkl, ip -> ijkl',riemann_tensor,g) # Riemann tensor of rank (0, 4) [all covariant indices]
        n = riemann_covariant_tensor.shape[-1]
        ricci_tensor = jnp.einsum('kikj -> ij', riemann_tensor)
        g_inv = jnp.linalg.inv(g)
        ricci_scalar = jnp.einsum('ii -> ', g_inv @ ricci_tensor)

        return riemann_covariant_tensor - 1.0/(n - 2) * DifferentialGeometry.kulkarni_nomizu_product((ricci_tensor- ricci_scalar/n*g), g) - ricci_scalar/(2.0*n*(n-1))*DifferentialGeometry.kulkarni_nomizu_product(g, g)
    

    def ricci_tensor(self,coords: jax.Array) -> jax.Array:
        """
        Compute the Ricci tensor R_{ij} = R^{k}_{ikj} at a given coordinate.

        Parameters
        ----------
        coords : jax.Array
            Coordinates of the point at which to compute the Ricci tensor.

        Returns
        -------
        ricci_tensor : jax.Array
            Ricci tensor of rank {0; 2} evaluated at the given coordinate.
        """
        return jnp.einsum('lilj -> ij', self.riemann_tensor(coords))
    
    
    @staticmethod
    def ricci_tensor_static(riemann_tensor: jax.Array) -> jax.Array:
        """
        Compute the Ricci tensor R_{ij} = R^{k}_{ikj} given the Riemann curvature tensor at some coordinate.

        Parameters
        ----------
        riemann_tensor : jax.Array
            Riemann curvature tensor of rank {1; 3} evaluated at some coordinate.

        Returns
        -------
        ricci_tensor : jax.Array
            Ricci tensor of rank {0; 2} evaluated at some coordinate.
        """

        return jnp.einsum('lilj -> ij', riemann_tensor)


    def ricci_scalar(self, coords: jax.Array) -> jax.Array: 
        """Compute the Ricci scalar R (Trace of Ricci tensor) evaluated at a given coordinate.
        
        Parameters
        ----------
        coords : jax.Array
            Coordinates of the point at which to compute the Ricci scalar.

        Returns
        -------
        ricci_scalar : jax.Array
            Ricci scalar at the given collocation point. 
        """
        g_inv = jnp.linalg.inv(self.metric(coords))   
        ricci_tensor = self.ricci_tensor(coords) 
        contracted_ricci_tensor = g_inv @ ricci_tensor
        ricci_scalar = jnp.einsum('ii -> ', contracted_ricci_tensor)
        return ricci_scalar
           
           
    @staticmethod
    def ricci_scalar_static(metric_tensor: jax.Array, ricci_tensor: jax.Array) -> jax.Array:
        """
        Compute the Ricci scalar R (Trace of Ricci tensor) given the metric tensor and the Ricci tensor 
        evaluated at some coordinate.

        Parameters
        ----------
        metric_tensor: jax.Array
            Metric tensor of rank {0; 2} evaluated at some coordinate.
        
        ricci_tensor: jax.Array
            Ricci tensor of rank {0; 2} evaluated at some coordinate.

        Returns
        -------
        ricci_scalar : jax.Array
            Ricci scalar evaluated at some coordinate.

        """
        inv_metric = jnp.linalg.inv(metric_tensor)
        contract_ricci = inv_metric @ ricci_tensor 
        ricci_scalar = jnp.einsum('ii -> ', contract_ricci)
        return ricci_scalar


    def kretschmann_invariant(self, coords: jax.Array) -> jax.Array:
        """
        Compute the curvature strength, the Kretschmann scalar K = R_{ijkl}*R^{ijkl} at a given coordinate.

        Parameters
        ----------
        coords: jax.Array
            Coordinates of the point at which to compute the Kretschmann invariant.

        Returns
        -------
        kretschmann_invariant : jax.Array
            Kretschmann scalar evaluated at the given coordinate.

        """
        g = self.metric(coords)
        R = self.riemann_tensor(coords)
        return DifferentialGeometry.kretschmann_invariant_static(g, R)
    
    
    @staticmethod
    def kretschmann_invariant_static(metric_tensor: jax.Array, riemann_tensor: jax.Array) -> jax.Array:
        """
        Compute the Kretschmann invariant given the metric tensor and the Riemann curvature tensor
        evaluated at some coordinate.

        Parameters
        ----------
        metric_tensor: jax.Array  
            metric tensor grid each of rank {0; 2} evaluated at some coordinate. 

        riemann_tensor: jax.Array 
            Riemann tensor grid each of rank {1; 3} evaluated at some coordinate.

        Returns
        -------
        kretschmann_invariant: jax.Array
            Kretschmann scalar evaluated at some coordinate.
        """
        g = metric_tensor
        g_inv = jnp.linalg.inv(g)
        R = riemann_tensor
        R_cov = jnp.einsum('ijkl,im -> mjkl', R, g)
        R_cont = R
        R_cont = jnp.einsum('ijkl,jm -> imkl', R_cont, g_inv)
        R_cont = jnp.einsum('ijkl,km -> ijml', R_cont, g_inv)
        R_cont = jnp.einsum('ijkl,lm -> ijkm', R_cont, g_inv)

        return jnp.einsum('ijkl,ijkl', R_cov, R_cont)
        
    
    def schouten_tensor(self, coords: jax.Array) -> jax.Array: 
        """

        Compute the Schouten tensor S_{ij} = 1/(n-2) * (R_{ij} - 0.5 * R g_{ij} / (n-1)) at a given coordinate.
        
        Parameters 
        -----------
        coords: jax.Array 
            Coordinates of the point at which to compute the Schouten tensor.
        
        Returns 
        ----------- 
        Schouten tensor: jax.Array 
            Schouten tensor of rank {0; 2} evaluated at the given coordinate.
        """
        g = self.metric(coords)
        ricci_tensor = self.ricci_tensor(coords)
        ricci_scalar = self.ricci_scalar(coords)
        dim = g.shape[-1]

        return 1 / (dim - 2) * (ricci_tensor - 0.5 * ricci_scalar * g / (dim - 1))

    def cotton_tensor(self, coords : jax.Array) -> jax.Array:
        
        """
        Compute the Cotton tensor C_{ijk} = del_{k}R_{ij} - del_{j}R_{ik} + 0.5 / (n - 1) * (g_{ik}del_{j}R - g_{ij}del_{k}R)
        at a given coordinate, where R is the ricci scalar and n is the dimension of space-time,
        del_{i} is the covariant derivative,
        R_{ij} is the ricci tensor and
        g_{ij} is the metric tensor.

        Parameters
        ---------
        coords : jax.Array
            Coordinates of the point at which to cumpute the Cotton tensor. 
            
        Returns
        -------

        cotton_tensor : jax.Array
            Cotton tensor evaluated at the given coordinate.

        """

        ricci_t_cov = self.levi_civita_connection(coords, self.ricci_tensor, rank=[0, 0])
        ricci_s_cov = self.levi_civita_connection(coords, self.ricci_scalar, rank=[])
        g = self.metric(coords)
        dim = g.shape[-1]

        cotton_tensor = jnp.einsum('kij -> ijk', ricci_t_cov) - jnp.einsum('jik -> ijk', ricci_t_cov) + 0.5 / (dim - 1) * (jnp.einsum('j,ik -> ijk', ricci_s_cov, g) - jnp.einsum('k,ij -> ijk', ricci_s_cov, g))

        return cotton_tensor 
    
    
    @staticmethod 
    def schouten_tensor_static(metric_tensor: jax.Array, ricci_tensor: jax.Array, ricci_scalar: float) -> jax.Array: 
        """
        
        Compute the Schouten tensor S_{ij} = 1/(n-2) * (R_{ij} - 0.5 * R g_{ij} / (n-1)) given the metric tensor,
        the Ricci tensor and the Ricci scalar at some coordinate.

        Parameters
        ----------
        metric_tensor: jax.Array
            Metric tensor of rank {0; 2} evaluated at some coordinate.
        ricci_tensor: jax.Array
            Ricci tensor of rank {0; 2} evaluated at some coordinate.
        ricci_scalar: float
            Ricci scalar evaluated at some coordinate.
        
        Returns
        -------
        schouten_tensor: jax.Array
            Schouten tensor of rank {0; 2} evaluated at some coordinate.

        """
        n = ricci_tensor.shape[-1]
        return 1/(-2)*(ricci_tensor - 0.5 * ricci_scalar * metric_tensor/(n-1))
    
    def einstein_tensor(self, coords: jax.Array) -> jax.Array: 
        """
        Compute the Einstein tensor G_{ij} = R_{ij} - 1/2 g_{ij} R (without cosmological constant term) at a given coordinate.

        Parameters
        ----------
        coords : jax.Array
            Coordinates of the point at which to compute the Einstein tensor.
        
        Returns
        -------
        einstein_tensor : jax.Array
            Einstein tensor of rank {0; 2} evaluated at the given coordinate.

        """ 
        metric_tensor = self.metric(coords)
        ricci_tensor = self.ricci_tensor(coords)
        ricci_scalar = self.ricci_scalar(coords)
        return ricci_tensor - 0.5*metric_tensor*ricci_scalar 
    
    @staticmethod
    def einstein_tensor_static(metric_tensor: jax.Array, ricci_tensor: jax.Array, ricci_scalar: float) -> jax.Array: 
        """
        Compute the Einstein tensor G_{ij} = R_{ij} - 1/2 g_{ij} R (without cosmological constant term) given the metric tensor,
        the Ricci tensor and the Ricci scalar at some coordinate.

        Parameters
        ----------
        metric_tensor: jax.Array
            Metric tensor of rank {0; 2} evaluated at some coordinate.
        ricci_tensor: jax.Array
            Ricci tensor of rank {0; 2} evaluated at some coordinate.
        ricci_scalar: float
            Ricci scalar evaluated at some coordinate.

        Returns
        -------
        einstein_tensor: jax.Array
            Einstein tensor of rank {0; 2} evaluated at some coordinate.    

        """
        einstein_tensor = ricci_tensor - 0.5*metric_tensor*ricci_scalar 
        return einstein_tensor
    
    @staticmethod
    def check_bianchi_identity_first(riemann_tensor: jax.Array, eps: float = 1e-5) -> jax.Array:
        """
        The first Bianchi identity asserts that for the covariant Riemann curvature tensor: R_{ijkl} + R_{jkil} + R_{kijl} = 0

        Parameters
        ----------
        riemann_covariant : jax.Array
            Covariant Riemann curvature tensor of rank {0, 4} evaluated at some coordinate.
        eps : float
            Absolute offset or tolerance in asserting the difference between the AD computation and the ground truth.

        Returns
        -------
        check_bianchi_first: bool
            A boolean value indicating whether the first Bianchi identity holds or not.
        """
        bianchi = riemann_tensor + jnp.einsum('jkil -> ijkl', riemann_tensor) + jnp.einsum('kijl -> ijkl', riemann_tensor)
        zeros = jnp.zeros_like(bianchi)
        return jnp.allclose(bianchi, zeros, rtol=0, atol=eps)
    
    def bianchi_identity_second(self, coords: jax.Array, antisymmetric_cov_rank_four_tensor: Callable[[jax.Array], jax.Array]) -> jax.Array: 
        """
        The second Bianchi identity asserts that for a covariant antisymmetric curvature tensor A_{abmn; l} + A_{ablm; n} + A_{abnl;m} = 0
        
        Parameters
        ----------

        coords : jax.Array
            Coordinates of the point at which to compute the second Bianchi identity.

        antisymmetric_cov_rank_four_tensor: Callable[[jax.Array], jax.Array]
            A python function with a single input corresponding to the coordinates where to output the antisymmetric covariant rank four tensor.

        Returns
        -------
        bianchi_second_identity : jax.Array
            Second Bianchi identity obtained via cyclic antisymmetric sum over first 3 three indices. 
        """
        antisymmetric_tensor_jac = self.levi_civita_connection(coords, antisymmetric_cov_rank_four_tensor, rank=(0, 0, 0, 0))
        return antisymmetric_tensor_jac + jnp.einsum('ablmn -> abmnl', antisymmetric_tensor_jac) + jnp.einsum('abnlm -> abmnl', antisymmetric_tensor_jac)

    @staticmethod
    def bianchi_identity_second_static(antisymmetric_cov_rank_four_tensor_der: jax.Array) -> jax.Array:
        """
        The second Bianchi identity asserts that for a covariant antisymmetric curvature tensor A_{abmn; l} + A_{ablm; n} + A_{abnl;m} = 0
        
        Parameters
        ----------
        antisymmetric_cov_rank_four_tensor_der: jax.Array
            Covariant derivative of an antisymmetric rank four tensor of rank {0; 4} evaluated.

        Returns
        -------
        bianchi_second_identity : jax.Array
            Second Bianchi identity obtained via cyclic antisymmetric sum over first 3 three indices. 
        """
        return antisymmetric_cov_rank_four_tensor_der + jnp.einsum('ablmn -> abmnl', antisymmetric_cov_rank_four_tensor_der) + jnp.einsum('abnlm -> abmnl', antisymmetric_cov_rank_four_tensor_der)

    def levi_civita_tensor(self, coords : jax.Array) -> jax.Array:
        """
            Compute Levi-Civita tensor of rank {0; 4} ε_{μνρσ} = √|det(g)| × e_{μνρσ}.
        """
        metric_output = self.metric(coords)
        dim = metric_output.shape[-1]
        det_g = jnp.linalg.det(metric_output)
        return jnp.sqrt(jnp.abs(det_g)) * levi_civita_symbol(dim)
    
    def levi_civita_connection(self, coords: jax.Array, tensor: Callable[[jax.Array], jax.Array], rank: tuple[bool, ...]) -> jax.Array: 
        """
        Compute the covariant derivative of a general tensor of rank (r, s), where r is the number
        of upper indices (contravariant) and s of lower indices (covariant). The general rule involves differentiating the tensor, then followed by
        adding various contractions of the christoffel symbols with the upper and lower indices of the tensor. For a 
        contraction with an upper index there is a + sign and a - sign for a lower index. For a rank (r, s) tensor,
        the covariant derivative of it is a rank (r, s + 1) tensor.

        Parameters
        ---------
        coords: jax.Array
            Coordinates of the point at which to cumpute the covariant derivative.

        tensor: Callable[[jax.Array], jax.Array]
            A python function with a single input corresponding to the coordinates where to output the tensor array.

        rank : tuple[bool, ...]
            A tuple having only values of zero and one corresponding to either lower or upper index. If tensor is a
            scalar, then len(rank) = 0.
            0 - lower index (covariant)
            1 - upper index (contravariant)
        
        Returns
        -------
        covariant_derivative : jax.Array
            Covariant derivative of tensor evaluated at the given coordinate.
            The first index is the derivative index ∇_{μ}, while the rest of the indices
            correspond to the tensor indices.   
        """
        n = len(rank)
        tensor_eval = tensor(coords)
        if n != len(tensor_eval.shape):
            raise ValueError(f"The rank of the tensor {rank} does not match the shape of the tensor {tensor_eval.shape}.")
        
        if n == 0:
            return jax.jacfwd(tensor)(coords)
        
        dtensor = jax.jacfwd(tensor)
        init_index = np.arange(n + 1)
        permuted_index = (init_index + n) % (n + 1)
        tensor = tensor_eval
        # Permutting the indices so that the first index is the derivative index
        dtensor = jnp.einsum(dtensor(coords), init_index, permuted_index)
        
        christoffel = self.christoffel_symbols(coords)

        tensor_index = np.arange(n) + 1
        result_index = np.arange(n + 1)
        for i in range(n):
            tensor_index[i] = n + 1
            if rank[i] == 1:
                christoffel_index = np.array([i + 1, n + 1, 0])
                dtensor += jnp.einsum(christoffel, christoffel_index, tensor, tensor_index, result_index)
            else:
                christoffel_index = np.array([n + 1, i + 1, 0])
                dtensor -= jnp.einsum(christoffel, christoffel_index, tensor, tensor_index, result_index)
            tensor_index[i] = i + 1

        return dtensor
            
    @staticmethod
    def lie_derivative(coords: jax.Array,
                    vector_field : Callable[[jax.Array], jax.Array], 
                    tensor: Callable[[jax.Array], jax.Array],
                    rank: tuple[bool, ...],
                    connection: Callable[[jax.Array, Callable[[jax.Array], jax.Array], tuple[int, ...]], jax.Array] = None) -> jax.Array:
        """
        Compute the directional-derivative of a tensor field along a vector field for a general 
        tensor of rank (r, s), where r is the number of upper indices (contravariant) and s of lower indices (covariant). The general rule 
        involves contracting the vector field index with the derivative of the tensor field, then followed by
        adding various contractions of the vector field's derivative with the upper and lower indices of the tensor.
        For a contraction with an upper index there is a - sign and a + sign for a lower index. In case a connection is used,
        the partial derivative is replaced by the covariant derivative. For a rank (r, s) tensor,
        the Lie derivative of it is a tensor of the same rank. 
        
        Note, the Lie derivative is defined independent of connection. If a connection is provided and is torsion-free,
        then replacing the partial derivative with the covariant derivative will yield the same result.

        Parameters
        ---------
        coord: jax.Array
            Coordinates of the point at which to compute the Lie derivative.
        
        vector_field: Callable[[jax.Array], jax.Array]
            A python function with a single input corresponding to the coordinate where to output the vector array.
        
        tensor: Callable[[jax.Array], jax.Array]
            A python function with a single input corresponding to the coordinate where to output the tensor array.

        rank : tuple
            A tuple having only values of zero and one corresponding to either lower or upper index. If the tensor is a
            scalar, then len(rank) = 0.
            0 - lower index (covariant)
            1 - upper index (contravariant)
        
        connection: Callable[[jax.Array, Callable[[jax.Array], jax.Array], tuple[int, ...]], jax.Array], optional
            A python function corresponding to the covariant derivative operator defined by the type of connection. Usually,
            this is the Levi-Civita connection. The signature follows the form
            of the Levi-Civita connection in the `DifferentialGeometry` class. If None, the function will use the
            partial derivative instead of the covariant derivative.
        
        Returns
        -------
        lie_tensor : jax.Array
            Lie derivative of tensor evaluated at the given coordinate.
        """
        n = len(rank)
        tensor_eval = tensor(coords)
        if n != len(tensor_eval.shape):
            raise ValueError(f"The rank of the tensor {rank} does not match the shape of the tensor {tensor_eval.shape}.")
        vector = vector_field(coords)
        init_index = np.arange(n + 1)
        if connection is not None:
            dtensor = connection(coords, tensor, rank)
        else:
            dtensor = jax.jacfwd(tensor)
            permuted_index = (init_index + n) % (n + 1)
            dtensor = jnp.einsum(dtensor(coords), init_index, permuted_index)

        tensor = tensor_eval
        lie_tensor = jnp.einsum(vector, [0], dtensor, init_index)

        if n == 0:
            return lie_tensor
        
        if connection is not None:
            dvector_field = connection(coords, vector_field, (1,))
        else:
            dvector_field = jax.jacfwd(vector_field)(coords)
            dvector_field = jnp.einsum(dvector_field, [0, 1], [1, 0])

        tensor_index = np.arange(n)
        result_index = np.arange(n)
        for i in range(n):
            result_index[i] = n
            if rank[i] == 1:
                vector_index = np.array([i, n])
                lie_tensor -= jnp.einsum(dvector_field, vector_index, tensor, tensor_index, result_index)
            else:
                vector_index = np.array([n, i])
                lie_tensor += jnp.einsum(dvector_field, vector_index, tensor, tensor_index, result_index)
            result_index[i] = i

        return lie_tensor
    
    @staticmethod
    def tensor_frame_transformation(coords: jax.Array,
                                    coord_transform : Callable[[jax.Array], jax.Array], 
                                    tensor: Callable[[jax.Array], jax.Array], 
                                    rank: tuple[bool, ...]) -> jax.Array:
        """
        Change the components of a tensor field under a change of coordinates. It follows the general rule
        of contracting the Jacobian of the coordinate transformation for every contravariant index and the inverse Jacobian for every covariant index.  

        Parameters
        ---------

        coords: jax.Array
            The point where to evaluate the tensor field in the old coordinate.

        coord_transform: Callable[[jax.Array], jax.Array]
            A python function with a single input corresponding to the change between the old to the new coordinate system.

        tensor: Callable[[jax.Array], jax.Array]
            A python function with a single input corresponding to the coordinates where to output the tensor field.

        rank : tuple[bool, ...]
            A tuple having only values of zero and one corresponding to either lower or upper index. If the tensor is a
            scalar, then len(rank) = 0.
            0 - lower index (covariant)
            1 - upper index (contravariant)

        Returns
        -------
        new_tensor : jax.Array
            Array form of the tensor field evaluated at the given coordinate.
        """
        n = len(rank)

        old_tensor = tensor(coords)

        if n != len(old_tensor.shape):
            raise ValueError(f"The rank of the tensor {rank} does not match the shape of the tensor {old_tensor.shape}.")
        
        up_index = np.where(np.array(rank) == 1)[0]
        down_index = np.where(np.array(rank) == 0)[0]
        up = len(up_index)
        down = len(down_index)

        jacobian = jax.jacfwd(coord_transform)(coords)

        if down > 0:
            inv_jacobian = jnp.linalg.inv(jacobian)

        new_tensor = old_tensor

        tensor_index = np.arange(n)

        result_index = np.arange(n)

        for i in range(up):
            result_index[up_index[i]] = n
            new_tensor = jnp.einsum(jacobian, [n, up_index[i]], new_tensor, tensor_index, result_index)
            result_index[up_index[i]] = up_index[i]

        for i in range(down):
            result_index[down_index[i]] = n
            new_tensor = jnp.einsum(inv_jacobian, [down_index[i], n], new_tensor, tensor_index, result_index)
            result_index[down_index[i]] = down_index[i]


        return new_tensor
    
    @staticmethod
    def kulkarni_nomizu_product(T: jax.Array, S:jax.Array) -> jax.Array: 
        """
        Compute the Kulkarni-Nomizu product {O^} between two symmetric tensors T, S each of rank {0; 2} in a particular basis.

        Parameters
        ----------
        T: jax.Array
            Symmetric tensor of rank {0; 2}
            
        S: jax.Array
            Symmetric tensor of rank {0; 2}

        Returns
        -------
        T_knprod_S: jax.Array
            Kulkarni-Nomizu product between (T {O^} S): skew-symmetric resultant tensor of rank {0; 4}.
        """
        T_knprod_S = jnp.einsum('im, kl -> iklm', T, S) - jnp.einsum('il, km -> iklm', T, S) + jnp.einsum('kl, im -> iklm', T, S) - jnp.einsum('km, il -> iklm', T, S) 
        return T_knprod_S
    
    
    def tetrad(self, coords : jax.Array) -> jax.Array:
        """
        
        Compute the tetrad basis at a given coordinate. 
        The tetrad basis is a set of four orthonormal vectors (w.r.t the manifold spacetime metric) 
        that span the tangent space at a given point.

        Note, tetrad basis is not unique, and the choice of tetrad basis is arbitrary.
        The tetrad basis is computed by diagonalizing the metric tensor at the given coordinate.
        The tetrad is also not a tensor.

        Parameters
        ----------
        coords : jax.Array
            Coordinates of the point at which to compute the tetrad basis.

        Returns
        -------
        tetrad : jax.Array
            Tetrad basis at the given coordinate. The first index is the spacetime and the second is the local Lorentz frame index.

        """
        metric = self.metric(coords)
        key = jax.random.PRNGKey(0)
        noise = jax.random.normal(key, metric.shape) * 1e-12
        noise = (noise + noise.T) / 2  # Keep symmetric
    
        regularized_metric = metric + noise
        eigvals, eigvecs = jnp.linalg.eigh(regularized_metric)
        neg_indx = jnp.argmin(eigvals)
        indices = jnp.concatenate([jnp.array([neg_indx]), jnp.delete(jnp.arange(eigvals.shape[0]), neg_indx)])
        eigvecs = eigvecs[:, indices]
        
        return eigvecs * jnp.sqrt(jnp.abs(eigvals))[None, :]
    
    def spin_connection_coefficients(self, coords : jax.Array) -> jax.Array:
        """
        Compute the spin connection coefficients ω_{μ}^{ab} at a given coordinate.

        Note, the spin connection coefficients don't form a tensor.

        Parameters
        ----------
        coords : jax.Array
            Coordinates of the point at which to compute the spin connection.

        Returns
        -------

        spin connection : jax.Array
            Spin connection coefficients at the given coordinate. First index is the spacetime metric tensor index,
            and the second and third indices are w.r.t the Minkowski diagonalization of the metric tensor. 

        """

        christoffel = self.christoffel_symbols(coords)
        e = self.tetrad(coords)
        e_contra = lambda coords : jnp.einsum('ij,ki -> kj', self.tetrad(coords), jnp.linalg.inv(self.metric(coords)))
        jac_e_contra = jnp.einsum('ijk -> kij', jax.jacfwd(e_contra)(coords))
        spin_connection = jnp.einsum('ij,ikl,km -> ljm', e, christoffel, e_contra(coords)) + jnp.einsum('ij,kil -> kjl', e, jac_e_contra)
        return spin_connection


    def first_invariant_weyl(self, coords : jax.Array) -> float:
        """
        Compute the first invariant of the Weyl tensor, which is the "square" of the Weyl curvature tensor.

        Parameters
        ----------
        coords : jax.Array
            Coordinates of the point at which to compute the first invariant of the Weyl tensor.
        
        Returns
        -------
        first_invariant_weyl : float
            First invariant of the Weyl tensor evaluated at the given coordinate.

        """
        weyl_tensor = self.weyl_curvature_tensor(coords)
        g_inv = jnp.linalg.inv(self.metric(coords))
        weyl_contra = jnp.einsum('ijkl,im -> mjkl', weyl_tensor, g_inv)
        weyl_contra = jnp.einsum('ijkl,jm -> imkl', weyl_contra, g_inv)
        weyl_contra = jnp.einsum('ijkl,km -> ijml', weyl_contra, g_inv)
        weyl_contra = jnp.einsum('ijkl,lm -> ijkm', weyl_contra, g_inv)
        return jnp.einsum('ijkl,ijkl', weyl_tensor, weyl_contra)
    
    def second_invariant_weyl(self, coords : jax.Array) -> float:
        """
        Compute the second invariant of the Weyl tensor, which is the contraction of the
        dual Weyl curvature tensor with itself.

        Parameters
        ----------
        coords : jax.Array
            Coordinates of the point at which to compute the second invariant of the Weyl tensor.
        
        Returns
        -------
        second_invariant_weyl : float
            Second invariant of the Weyl tensor evaluated at the given coordinate.

        """
        g_inv = jnp.linalg.inv(self.metric(coords))
        weyl_tensor = self.weyl_curvature_tensor(coords) 
        weyl_contra = jnp.einsum('ijkl,im -> mjkl', weyl_tensor, g_inv)
        weyl_contra = jnp.einsum('ijkl,jm -> imkl', weyl_contra, g_inv)
        dual_weyl = 0.5 * jnp.einsum('ijkl,klmn -> ijmn', self.levi_civita_tensor(coords), weyl_contra)
        weyl_contra = jnp.einsum('ijkl,km -> ijml', weyl_contra, g_inv)
        weyl_contra = jnp.einsum('ijkl,lm -> ijkm', weyl_contra, g_inv)
        return jnp.einsum('ijkl,ijkl', dual_weyl, weyl_contra)
    
    def chern_pontryagin_scalar(self, coords : jax.Array) -> float:
        """
        Compute the Chern-Pontryagin scalar, which is the contraction of the dual
        Riemann tensor with itself.

        Parameters
        ----------
        coords : jax.Array
            Coordinates of the point at which to compute the Chern-Pontryagin scalar.  
        
        Returns
        -------
        chern_pontryagin_scalar : float
            Chern-Pontryagin scalar evaluated at the given coordinate.

        """
        riemann_tensor = self.riemann_tensor(coords)
        g_inv = jnp.linalg.inv(self.metric(coords))
        riemann_double_contra = jnp.einsum('ijkl,jm -> imkl', riemann_tensor, g_inv)
        dual_riemann = 0.5 * jnp.einsum('ijkl,klmn -> ijmn', self.levi_civita_tensor(coords), riemann_double_contra)
        riemann_contra = jnp.einsum('ijkl,km -> ijml', riemann_double_contra, g_inv)
        riemann_contra = jnp.einsum('ijkl,lm -> ijkm', riemann_contra, g_inv)
        return jnp.einsum('ijkl,ijkl', dual_riemann, riemann_contra)
    
    def euler_scalar(self, coords : jax.Array) -> float:
        """
        Compute the Euler scalar, which is the contraction of the left and right dual of the 
        Riemann tensor with itself.

        Parameters
        ----------
        coords : jax.Array
            Coordinates of the point at which to compute the Euler scalar.
        
        Returns
        -------
        euler_scalar : float
            Euler scalar evaluated at the given coordinate.

        """
        levi_civita_tensor = self.levi_civita_tensor(coords)
        g_inv = jnp.linalg.inv(self.metric(coords))
        riemann_tensor = self.riemann_tensor(coords)
        riemann_contra = jnp.einsum('ijkl,jm -> imkl', riemann_tensor, g_inv)
        riemann_contra = jnp.einsum('ijkl,km -> ijml', riemann_contra, g_inv)
        riemann_contra = jnp.einsum('ijkl,lm -> ijkm', riemann_contra, g_inv)
        dual_dual_riemann = 1 / 4 * jnp.einsum('ijkl,mnop,klop', levi_civita_tensor, levi_civita_tensor, riemann_contra)
        return jnp.einsum('ijkl,ijkl', dual_dual_riemann, riemann_contra)

    @staticmethod
    def pushforward_vector_field_on_function(coords: jax.Array,
                                            phi: Callable[[jax.Array], jax.Array],
                                            Y: Callable[[jax.Array], jax.Array],
                                            f: Callable[[jax.Array], jax.Array]
                                        ) -> float:
        """
        Compute (φ_* Y)(f)(coords) = Y^μ ∂_μ (f ∘ φ)(coords).

        Parameters
        ----------

            coords: jax.Array
                Coordinates coords ∈ ℝ^n
            phi: Callable[[jax.Array], jax.Array]
                Map φ: ℝ^n → ℝ^m
            Y: Callable[[jax.Array], jax.Array]
                Vector field Y: ℝ^n → T_x ℝ^n
            f: Callable[[jax.Array], jax.Array]
                Function f: ℝ^m → ℝ

        Returns
        -------

        jvp: jax.Array
            A scalar value representing Y^μ ∂_μ (f ∘ φ)(coords).
                
        """
        Y_vec = Y(coords)  # Vector at coords
        composed = lambda coords: f(phi(coords))  # f ∘ φ: ℝ^n → ℝ
        _, jvp = jax.jvp(composed, (coords,), (Y_vec,)) # JVP of f ∘ φ with respect to Y

        return jvp 

    @staticmethod
    def pullback_one_form(coords: jax.Array,
                        phi: Callable[[jax.Array], jax.Array],
                        alpha: Callable[[jax.Array], jax.Array],
                        tangent_vector: jax.Array
                    ) -> float:
        """
        Compute (φ^* α)(v) = α(dφ(v)).

        Parameters
        ----------
            coords: jax.Array
                Coordinates coords ∈ ℝ^n
            phi: Callable[[jax.Array], jax.Array]
                Map φ: ℝ^n → ℝ^m
            alpha: Callable[[jax.Array], jax.Array]
                Covector field (one-form) α: ℝ^m → ℝ^m 
            tangent_vector: jax.Array
                Tangent vector tangent_vector ∈ T_coords ℝ^n

        Returns
        -------
            A scalar value representing (φ^* α)(v) = α(φ(v)).

        """
        # Pushforward the tangent vector: dφ(v)
        alpha_cov = alpha(phi(coords))  # One-form coefficients at φ(coords)
        _, pushforward_vector = jax.jvp(phi, (coords,), (tangent_vector,))  # JVP of φ with respect to the tangent vector

        return jnp.dot(alpha_cov, pushforward_vector)  # Pullback contraction

if __name__=="__main__":
    jax.config.update("jax_enable_x64", True)  # Enable 64-bit precision for better accuracy in computations

    