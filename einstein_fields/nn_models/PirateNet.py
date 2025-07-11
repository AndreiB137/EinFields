import jax
import jax.numpy as jnp
import flax.linen as nn
from .utils import Dense, FourierEmbs
from typing import Union, Dict, Callable

class PIModifiedBottleneck(nn.Module):
    hidden_dim: int
    output_dim: int
    act: Callable
    nonlinearity: float
    reparam: Union[None, Dict]
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x, u, v):
        identity = x

        x = Dense(features=self.hidden_dim, reparam=self.reparam, dtype=self.dtype)(x)
        x = self.act(x)

        x = x * u + (1 - x) * v

        x = Dense(features=self.hidden_dim, reparam=self.reparam, dtype=self.dtype)(x)
        x = self.act(x)

        x = x * u + (1 - x) * v

        x = Dense(features=self.output_dim, reparam=self.reparam, dtype=self.dtype)(x)
        x = self.act(x)

        alpha = self.param("alpha", nn.initializers.constant(self.nonlinearity), (1,))
        x = alpha * x + (1 - alpha) * identity

        return x

class PirateNet(nn.Module):
    num_layers: int
    hidden_dim: int
    output_dim: int
    act: Callable = nn.silu
    nonlinearity: float = 0.0
    pi_init: Union[None, jnp.ndarray] = None
    reparam : Union[None, Dict] = None
    fourier_emb : Union[None, Dict] = None
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x):
        embs = FourierEmbs(**self.fourier_emb)(x)
        x = embs

        u = Dense(features=self.hidden_dim, reparam=self.reparam, dtype=self.dtype)(x)
        u = self.act(u)

        v = Dense(features=self.hidden_dim, reparam=self.reparam, dtype=self.dtype)(x)
        v = self.act(v)

        for _ in range(self.num_layers):
            x = PIModifiedBottleneck(
                hidden_dim=self.hidden_dim,
                output_dim=x.shape[-1],
                act=self.act,
                nonlinearity=self.nonlinearity,
                reparam=self.reparam,
                dtype=self.dtype
            )(x, u, v)

        if self.pi_init is not None:
            kernel = self.param("pi_init", nn.initializers.constant(self.pi_init, dtype=self.dtype), self.pi_init.shape)
            y = jnp.dot(x, kernel)

        else:
            y = Dense(features=self.output_dim, reparam=self.reparam, dtype=self.dtype)(x)

        return x, y
    
if __name__ == "__main__":
    # Example usage
    from activations import cauchy
    cauchy_mod = lambda x : cauchy()(x)
    model = PirateNet(num_layers=3, hidden_dim=32, output_dim=16, act=cauchy_mod, reparam=None, fourier_emb={'embed_scale': 1.0, 'embed_dim': 64})
    params = model.init(jax.random.PRNGKey(0), jnp.ones(3))
    output = model.apply(params, jnp.ones(3))
    print(params)