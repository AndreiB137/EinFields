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
#        Automatic Vectorization version of tensor_calculus.DifferentialGeometry
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

import jax
import jax.numpy as jnp
from .tensor_calculus import DifferentialGeometry as diffgeo

class VectorizedDifferentialGeometry(): 
    def __init__(self, metric_geo: diffgeo) -> None: 
        self.diffgeo = metric_geo
    
    def metric(self, coords_batch: jax.Array) -> jax.Array:
        gij_batch = jax.vmap(self.diffgeo.metric)(coords_batch)
        return gij_batch
    
    def metric_jacobian(self, coords_batch: jax.Array) -> jax.Array:
        dgij_batch = jax.vmap(self.diffgeo.metric_jacobian)(coords_batch)
        return dgij_batch
    
    def metric_hessian(self, coords_batch: jax.Array) -> jax.Array:
        d2gij_batch = jax.vmap(self.diffgeo.metric_hessian)(coords_batch)
        return d2gij_batch
    
    def christoffel_symbols(self, coords_batch: jax.Array) -> jax.Array:
        gamma_ijk_batch = jax.vmap(self.diffgeo.christoffel_symbols)(coords_batch) 
        return gamma_ijk_batch
    
    def christoffel_jac(self, coords_batch: jax.Array) -> jax.Array:
        dgamma_ijk_batch = jax.vmap(self.diffgeo.christoffel_jac)(coords_batch) 
        return dgamma_ijk_batch
    
    def riemann_tensor(self, coords_batch: jax.Array) -> jax.Array:
        Rijkl_batch = jax.vmap(self.diffgeo.riemann_tensor)(coords_batch) 
        return Rijkl_batch
    
    def ricci_tensor(self, coords_batch: jax.Array) -> jax.Array: 
        Rij_batch = jax.vmap(self.diffgeo.ricci_tensor)(coords_batch) 
        return Rij_batch
    
    def ricci_scalar(self, coords_batch: jax.Array) -> jax.Array: 
        R_batch = jax.vmap(self.diffgeo.ricci_scalar)(coords_batch)
        return R_batch

    def einstein_tensor(self, coords_batch: jax.Array) -> jax.Array: 
        G_batch = jax.vmap(self.diffgeo.einstein_tensor)(coords_batch)
        return G_batch 

    def schouten_tensor(self, coords_batch: jax.Array) -> jax.Array:
        P_batch = jax.vmap(self.diffgeo.schouten_tensor)(coords_batch) 
        return P_batch
       
    def kretschmann_invariant(self, coords_batch: jax.Array) -> jax.Array:
        K_batch = jax.vmap(self.diffgeo.kretschmann_invariant)(coords_batch) 
        return K_batch
    
    def cotton_tensor(self, coords_batch: jax.Array) -> jax.Array:
        C_batch = jax.vmap(self.diffgeo.cotton_tensor)(coords_batch) 
        return C_batch

    @staticmethod
    def riemann_tensor_static(christoffel_batch: jax.Array, christoffel_jac_batch: jax.Array) -> jax.Array:
        riemann_tensor_batch = jax.vmap(diffgeo.riemann_tensor_static, in_axes=(0, 0))(christoffel_batch, christoffel_jac_batch)
        return riemann_tensor_batch

    @staticmethod
    def riemann_tensor_covariant_static(metric_tensor_batch: jax.Array, riemann_tensor_batch: jax.Array) -> jax.Array:
        riemann_tensor_cov_batch = jax.vmap(diffgeo.covariant_riemann_tensor_static, in_axes=(0, 0))(metric_tensor_batch, riemann_tensor_batch)
        return riemann_tensor_cov_batch
    
    @staticmethod
    def weyl_tensor_static(metric_tensor_batch: jax.Array, riemann_tensor_batch: jax.Array) -> jax.Array: 
        weyl_tensor_batch = jax.vmap(diffgeo.weyl_curvature_tensor_static, in_axes=(0, 0))(metric_tensor_batch, riemann_tensor_batch)
        return weyl_tensor_batch
    
    @staticmethod
    def ricci_tensor_static(riemann_tensor_batch: jax.Array) -> jax.Array:
        ricci_tensor_batch = jax.vmap(diffgeo.ricci_tensor_static)(riemann_tensor_batch)
        return ricci_tensor_batch

    @staticmethod
    def ricci_scalar_static(metric_tensor_batch: jax.Array, riemann_tensor_batch: jax.Array) -> jax.Array:
        ricci_tensor_batch = VectorizedDifferentialGeometry.ricci_tensor_static(riemann_tensor_batch)
        ricci_scalar_batch = jax.vmap(diffgeo.ricci_scalar_static, in_axes=(0, 0))(metric_tensor_batch, ricci_tensor_batch)
        return ricci_scalar_batch

    @staticmethod
    def schouten_tensor_static(metric_tensor_batch: jax.Array, riemann_tensor_batch: jax.Array) -> jax.Array: 
        ricci_tensor_batch = VectorizedDifferentialGeometry.ricci_tensor_static(riemann_tensor_batch) 
        ricci_scalar_batch = VectorizedDifferentialGeometry.ricci_scalar_static(metric_tensor_batch, riemann_tensor_batch)
        schouten_tensor_batch = jax.vmap(diffgeo.schouten_tensor_static, in_axes=(0, 0, 0))(metric_tensor_batch, ricci_tensor_batch, ricci_scalar_batch)
        return schouten_tensor_batch
    
    @staticmethod
    def einstein_tensor_static(metric_tensor_batch: jax.Array, riemann_tensor_batch: jax.Array) -> jax.Array:
        ricci_tensor_batch = VectorizedDifferentialGeometry.ricci_tensor_static(riemann_tensor_batch)
        ricci_scalar_batch = VectorizedDifferentialGeometry.ricci_scalar_static(metric_tensor_batch, riemann_tensor_batch)
        einstein_tensor_batch = ricci_tensor_batch - 0.5*jnp.einsum('a, aij -> aij', ricci_scalar_batch, metric_tensor_batch)
        return einstein_tensor_batch

    @staticmethod
    def kretschmann_invariant_static(metric_tensor_batch: jax.Array, riemann_tensor_batch: jax.Array) -> jax.Array: 
        return jax.vmap(diffgeo.kretschmann_invariant_static)(metric_tensor_batch, riemann_tensor_batch)