# Flax
import jax.numpy as jnp
from flax import linen as nn
from typing import Callable, Union, Dict
from .utils import Dense, FourierEmbs

# Modified MLP version based on the state-of-the-art practicies in PINN training:
# Fourier embeddings and random weight factorization
# You can read more about it in the paper: https://arxiv.org/pdf/2210.01274


class MLP_PINN(nn.Module):
    hidden_dim: int
    output_dim: int
    num_layers: int
    act: Callable = nn.silu
    dtype: jnp.dtype = jnp.float32
    reparam : Union[None, Dict] = None
    fourier_emb : Union[None, Dict] = None

    @nn.compact
    def __call__(self, x):
        if self.fourier_emb is not None:
            x = FourierEmbs(**self.fourier_emb)(x)
        else:
            x = Dense(
                features=self.hidden_dim,
                reparam=self.reparam,
                dtype=self.dtype
            )(x)
            x = self.act(x)
        for _ in range(self.num_layers):
            x = Dense(
                features=self.hidden_dim,
                reparam=self.reparam,
                dtype=self.dtype
            )(x)
            x = self.act(x)
        x = Dense(
            features=self.output_dim,
            reparam=self.reparam,
            dtype=self.dtype
        )(x)
        return x
