
import jax
import jax.flatten_util
import jax.numpy as jnp
from typing import Union

def get_cosine_similarity(grad1 : jnp.ndarray, grad2 : jnp.ndarray) -> float:
    """
    Computes the cosine similarity between two gradients.
    
    Args:
        grad1 (jnp.ndarray): First gradient.
        grad2 (jnp.ndarray): Second gradient.
        
    Returns:
        float: Cosine similarity between the two gradients.
    """
    norm1 = jnp.linalg.norm(grad1)
    norm2 = jnp.linalg.norm(grad2)

    # Compute the cosine similarity
    cos_angle = jnp.dot(grad1, grad2) / (norm1 * norm2)
    
    return cos_angle

def get_alignment(grads : jax.Array):
    """
    Computes the alignment of gradients.
    
    Args:
        grads (list[jax.Array]): List of gradients.
        
    Returns:
        jnp.ndarray: Alignment of gradients.
    """
    
    return 2 * jnp.linalg.norm(jnp.sum(grads / jnp.linalg.norm(grads, axis=1, keepdims=True), axis=0) / grads.shape[0])**2 - 1


def Config_update_double(grad1 : dict, grad2 : dict) -> dict:
    grad1, unflatten = jax.flatten_util.ravel_pytree(grad1)
    grad2, _ = jax.flatten_util.ravel_pytree(grad2)[0] 

    norm1 = jnp.linalg.norm(grad1)
    norm2 = jnp.linalg.norm(grad2)

    unit_1 = grad1 / norm1
    unit_2 = grad2 / norm2

    # Compute the cosine similarity
    cos_angle = get_cosine_similarity(grad1, grad2)

    or_2 = grad1 - norm1 * cos_angle * unit_2
    or_1 = grad2 - norm2 * cos_angle * unit_1

    unit_or1 = or_1 / jnp.linalg.norm(or_1)
    unit_or2 = or_2 / jnp.linalg.norm(or_2)
    weights = jnp.ones(2)
    coef1, coef2 = (jnp.dot(unit_or2, unit_1) / (weights[0] / weights[1] * jnp.dot(unit_or1, unit_2)),
                    1,
            )

    best_direction = coef1 * unit_or1 + coef2 * unit_or2

    unit_best_direction = best_direction / jnp.linalg.norm(best_direction)

    return unflatten(unit_best_direction * (jnp.dot(grad1, unit_best_direction) + jnp.dot(grad2, unit_best_direction)))

def config_update(grads):
    grads_l = []
    grads0, unflatten = jax.flatten_util.ravel_pytree(grads[0])

    grads[0] = grads0
    for i in range(1, len(grads)):
        grads[i] = jax.flatten_util.ravel_pytree(grads[i])[0]

    grads_l = jnp.stack(grads, axis=0)

    weights = jnp.ones(grads_l.shape[0])

    units = jnp.nan_to_num(grads_l / jnp.linalg.norm(grads_l, axis=1, keepdims=True), 0.)

    best_direction = jnp.linalg.pinv(units) @ weights

    unit_best_direction = best_direction / jnp.linalg.norm(best_direction)

    return unflatten(jnp.sum(
        jnp.stack([jnp.dot(grad_i, unit_best_direction) for grad_i in grads_l])
    ) * unit_best_direction)

def estimate_conflict(gradients: jnp.ndarray) -> jnp.ndarray:

    grad_sum = gradients.sum(axis=0)
    direct_sum = grad_sum / jnp.linalg.norm(grad_sum)
    unit_grads = gradients / jnp.linalg.norm(gradients, axis=1, keepdims=True)
    return unit_grads @ direct_sum







    