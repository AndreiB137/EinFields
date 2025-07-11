# Flax
import jax.numpy as jnp
from flax import linen as nn
import jax
from typing import Callable

class MLP(nn.Module):
    hidden_dim: int
    output_dim: int
    num_layers: int
    act: Callable = nn.silu
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(
            features=self.hidden_dim,
            use_bias=True,
            kernel_init=nn.initializers.glorot_normal(dtype=self.dtype),
            param_dtype=self.dtype
        )(x)
        x = self.act(x)
        for _ in range(self.num_layers):
            x = nn.Dense(
                features=self.hidden_dim,
                use_bias=True,
                kernel_init=nn.initializers.glorot_normal(dtype=self.dtype),
                param_dtype=self.dtype
            )(x)
            x = self.act(x)
        x = nn.Dense(
            features=self.output_dim,
            use_bias=True,
            kernel_init=nn.initializers.glorot_normal(dtype=self.dtype),
            param_dtype=self.dtype
        )(x)
        return x

if __name__ == "__main__":
    # Example usage
    x = jax.random.uniform(jax.random.PRNGKey(0), (1, 3), minval=-3, maxval=3)
    model = MLP(hidden_dim=32, output_dim=16, num_layers=3)
    params = model.init(jax.random.PRNGKey(0), x)
    model_fn = lambda params, x : model.apply(params, x)
    print(model_fn(params, x).shape)