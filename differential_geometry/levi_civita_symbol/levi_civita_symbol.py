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

import os
import jax
import jax.numpy as jnp
import numpy as np
from sympy.combinatorics.permutations import Permutation
from itertools import permutations
import time

_LEVI_CIVITA_SYMBOLS = {}

def _compute_Levi_Civita_symbol(dim : int) -> jax.Array:
    """returns the Levi-Civita symbol"""
    
    epsilon = np.zeros(tuple([dim] * dim))

    for perm in permutations(range(dim)):
        sign = Permutation(perm).signature()
        epsilon[perm] = sign

    return jnp.array(epsilon)

def levi_civita_symbol(dim: int) -> jax.Array:
    """
    Returns the Levi-Civita symbol for a given dimension.

    Parameters
    ----------
    dim : int
        Dimension of the Levi-Civita symbol (2, 3, or 4).

    Returns
    -------
    jax.Array
        The Levi-Civita symbol of the specified dimension.
    """
    if dim not in _LEVI_CIVITA_SYMBOLS:
        raise ValueError(f"Levi-Civita symbol for dimension {dim} is not implemented.")
    
    return _LEVI_CIVITA_SYMBOLS[dim]

_current_dir = os.path.dirname(os.path.abspath(__file__))
symbol_saves_dir = os.path.join(_current_dir, "symbol_saves")

_LEVI_CIVITA_SYMBOLS[2] = jnp.load(os.path.join(symbol_saves_dir, "levi_civita_symbol_2D.npy"))
_LEVI_CIVITA_SYMBOLS[3] = jnp.load(os.path.join(symbol_saves_dir, "levi_civita_symbol_3D.npy"))
_LEVI_CIVITA_SYMBOLS[4] = jnp.load(os.path.join(symbol_saves_dir, "levi_civita_symbol_4D.npy"))

if __name__ == '__main__': 
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import time
    np.set_printoptions(suppress=True)