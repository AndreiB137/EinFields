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

from differential_geometry import diffgeo
import diffrax
from diffrax import ODETerm, diffeqsolve
import jax
import jax.numpy as jnp
from typing import Callable

def geodesic_equation(t, y, args):
    x, v = y[:4], y[4:]
    metric = args[0]
    coord_transform = args[1]

    dx = v
    dv = - jnp.einsum('ijk,j,k', metric.christoffel_symbols(coord_transform(x)), v, v)

    return jnp.concatenate([dx, dv], axis=0)

def solver(metric : diffgeo,
           t0 : float, 
           t1 : float, 
           dt : float, 
           saveat : diffrax.SaveAt,
           solver : diffrax.AbstractSolver = diffrax.Kvaerno5(),
           stepsize_controller : diffrax.AbstractStepSizeController = diffrax.ConstantStepSize(),
           throw : bool = False,
           max_steps : int = 100,
           coord_transform : Callable[[jax.Array], jax.Array] = lambda coords: coords,
           progress_meter : diffrax.AbstractProgressMeter = diffrax.NoProgressMeter(),
           ):
    """
        Solves the geodesic equation for a given metric, that has to be initialized with its corresponding differential geometry object.

        For further details about the solver, consult the official diffrax documentation: https://docs.kidger.site/diffrax/.

        Parameters
        ----------

        metric : DifferentialGeometry
            The differential geometry object initialized with the metric.
        

        init_condition: jax.Array 
            The initial condition for the geodesic equation. It can be a batch of initial conditions.

        t0: float
            The initial time.

        t1: float 
            The final time.
        
        dt: float
            The initial time step at t0.

        saveat: diffrax.SaveAt
            The timesteps where to save the solution.

        solver: diffrax.AbstractSolver 
            The ODE solver to use.

        stepsize_controller: diffrax.AbstractStepSizeController 
            Adaptive stepsize controller.

        throw: bool 
            Whether to throw an error if the solver fails (such as reaching max_steps).
        
        max_steps: int 
            The maximum number of steps allowed to reached the tollerance region provided by the stepsize controller.
            When saveAt(steps=True), the solution will be an array of shape (max_steps, ...) and if the solver doesn't need all the steps, it will be padded with NaNs.
        
        coord_transform: Callable[[jax.Array], jax.Array]
            A function that transforms the coordinates acoording to the training domain of the neural field. 
        
        progress_meter: diffrax.AbstractProgressMeter
            The progress meter to use.
        
        Returns
        -------

        geodesic_y0: Callable[[jax.Array], diffrax.Solution]
            The function that takes an initial condition and returns the solution of the geodesic equation.
            The solution is a diffrax.Solution object, which contains the time steps and the corresponding
            geodesic coordinates and velocities. This function is vectorized for handling batches 
            and then JIT compiled for performance.
    
    """
    def geodesic_y0(init_condition : jax.Array):
        return diffeqsolve(
            terms=ODETerm(geodesic_equation),
            t0=t0,
            t1=t1,
            dt0=dt,
            y0=init_condition,
            args=(metric, coord_transform,),
            solver=solver,
            stepsize_controller=stepsize_controller,
            progress_meter=progress_meter,
            saveat=saveat,
            throw=throw,
            max_steps=max_steps
        )
    return jax.jit(jax.vmap(geodesic_y0))

if __name__ == '__main__': 
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    