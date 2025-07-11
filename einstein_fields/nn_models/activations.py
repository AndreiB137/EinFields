import jax
import jax.numpy as jnp
import flax.linen as nn

activation_function = {
    "relu": nn.relu,
    "gelu": nn.gelu,
    "silu": nn.silu,
    "swish": nn.silu,
    "tanh": nn.tanh,
    "sigmoid": nn.sigmoid,
    "softplus": nn.softplus,
    "softmax": nn.softmax,
    "leaky_relu": nn.leaky_relu,
    "elu": nn.elu,
    "selu": nn.selu,
    "telu": lambda x: x * jnp.tanh(jnp.exp(x)),
    "mish": lambda x: x * jnp.tanh(nn.softplus(x)),
    "cauchy": lambda x: cauchy()(x),
    "identity": lambda x: x,
    "react": lambda x: react()(x),
}

# https://arxiv.org/pdf/2503.02267v1
class react(nn.Module):
    @nn.compact
    def __call__(self, x):
        a = self.param(
            'a',
            jax.nn.initializers.normal(0.1),
            ()
        )
        b = self.param(
            'b',
            jax.nn.initializers.normal(0.1),
            ()
        )
        
        c = self.param(
            'c',
            jax.nn.initializers.normal(0.1),
            ()
        )
        d = self.param(
            'd',
            jax.nn.initializers.normal(0.1),
            ()
        )

        return (1 - jnp.exp(a * x + b)) / (1 + jnp.exp(c * x + d))

# https://arxiv.org/abs/2409.19221
class cauchy(nn.Module):
    @nn.compact
    def __call__(self, x):
        l1 = self.param(
            'lambda1',
            jax.nn.initializers.constant(1.0),
            ()
        )
        l2 = self.param(
            'lambda2',
            jax.nn.initializers.constant(1.0),
            ()
        )
        d = self.param(
            'd',
            jax.nn.initializers.constant(1.0),
            ()
        )

        return l1 * x / (x**2 + d**2) + l2 / (x**2 + d**2)

def get_activation(name):
    """
    Get the activation function by name.

    Args:
        name (str): Name of the activation function.

    Returns:
        Callable: The activation function.
    """
    if name not in activation_function:
        raise ValueError(f"Activation function `{name}` is not supported. Supported activations are : {list(activation_function.keys())}")
    return activation_function[name]

def list_activations():
    """
    List all available activation functions.

    Returns:
        list: A list of activation names.
    """
    return list(activation_function.keys())