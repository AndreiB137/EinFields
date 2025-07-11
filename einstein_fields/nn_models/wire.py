import jax.numpy as jnp
from flax import linen as nn
from typing import Any

import jax
from .utils import custom_uniform
from jax.nn.initializers import Initializer
    

def complex_kernel_uniform_init(numerator : float = 6,
                                 mode : str = "fan_in",
                                dtype : jnp.dtype = jnp.float32,
                                distribution: str = "uniform") -> Initializer:
    def init(key: jax.random.key, shape: tuple, dtype: Any = dtype) -> Any:
        real_kernel = custom_uniform(numerator=numerator, mode=mode, distribution=distribution)(key, shape, dtype)
        imag_kernel = custom_uniform(numerator=numerator, mode=mode, distribution=distribution)(key, shape, dtype)

        return real_kernel + 1j * imag_kernel
        
    return init


class WIRE(nn.Module):
    output_dim: int
    hidden_dim: int
    num_layers: int
    hidden_omega_0: float
    first_omega_0: float
    scale: float
    complexgabor: bool = False
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        if self.complexgabor:
            WIRElayer = ComplexGaborLayer
            dtype = jnp.complex64
        else:
            WIRElayer = RealGaborLayer
            dtype = self.dtype
        self.kernel_net = [
            WIRElayer(
                output_dim=self.hidden_dim,
                omega_0=self.first_omega_0,
                s_0=self.scale,
                is_first_layer=True,
                dtype=dtype
            )
        ] + [
            WIRElayer(
                output_dim=self.hidden_dim,
                omega_0=self.hidden_omega_0,
                s_0=self.scale,
                is_first_layer=False,
                dtype=dtype
            )
            for _ in range(self.num_layers)
        ]

        self.output_linear = nn.Dense(
            features=self.output_dim,
            use_bias=True,
            kernel_init=custom_uniform(numerator=1, mode="fan_in", distribution="normal"),
            param_dtype=self.dtype,
        )

    def __call__(self, x):
        for layer in self.kernel_net:
            x = layer(x)

        out = jnp.real(self.output_linear(x))

        return out


class ComplexGaborLayer(nn.Module):
    output_dim: int
    omega_0: float
    s_0: float
    is_first_layer: bool = False
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        c = 1 if self.is_first_layer else 6 / self.omega_0**2
        distrib = "uniform_squared" if self.is_first_layer else "uniform"

        if self.is_first_layer:
            dtype = self.dtype
        else:
            dtype = jnp.complex64

        self.linear = nn.Dense(
            features=self.output_dim,
            use_bias=True,
            kernel_init=complex_kernel_uniform_init(numerator=c, mode="fan_in", distribution=distrib),
            param_dtype=dtype
        )

    def __call__(self, x):
        omega = self.omega_0 * self.linear(x)
        scale = self.s_0 * self.linear(x)

        return jnp.exp(1j * omega - (jnp.abs(scale)**2))


class RealGaborLayer(nn.Module):
    output_dim: int
    omega_0: float
    s_0: float
    is_first_layer: bool = False
    dtype: jnp.dtype = jnp.float32

    def setup(self):

        c = 1 if self.is_first_layer else 6 / self.omega_0**2
        distrib = "uniform_squared" if self.is_first_layer else "uniform"

        self.freqs = nn.Dense(
            features=self.output_dim,
            kernel_init=custom_uniform(numerator=c, mode="fan_in", distribution=distrib, dtype=self.dtype),
            use_bias=True,
            param_dtype=self.dtype
        )

        self.scales = nn.Dense(
            features = self.output_dim,
            kernel_init=custom_uniform(numerator=c, mode="fan_in", distribution=distrib, dtype=self.dtype),
            use_bias=True,
            param_dtype=self.dtype
        )

    def __call__(self, x):
        omega = self.omega_0 * self.freqs(x)
        scale = self.s_0 * self.scales(x)

        return jnp.cos(omega) * jnp.exp(-(scale**2))
