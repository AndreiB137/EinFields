import jax
import jax.numpy as jnp 
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from typing import Union


class MiniBatchTensorDatasetDict:
    """mini-batches the data into chunks for reducing memory of activation layers + backpropagation steps"""
    def __init__(self, data: dict, num_batches: int, num_devices : int = 1, sharding : bool = False):
        # coords_grid: Union[np.memmap, jnp.ndarray], metric_tensor_grid: Union[np.memmap,jnp.ndarray],
        # metric_jacobian_grid: Union[np.memmap,jnp.ndarray],
        self.data = data 
        self.sharding = sharding
        self.num_batches = num_batches
        self.num_samples = self.data["coords"].shape[0]
        self.chunk_size = self.num_samples // self.num_batches
        self.prng_key = jax.random.PRNGKey(0)
        self.device_shard = self.no_shard_batch
        if self.num_samples % self.chunk_size != 0:
            self.num_batches += 1

        if sharding:
            self.num_devices = num_devices
            self.devices_mesh = np.array(jax.devices()[:self.num_devices]).reshape(self.num_devices)
            self.mesh = Mesh(self.devices_mesh, axis_names=('batch',))
            self.pinn_sharding = NamedSharding(self.mesh, P('batch'))
            self.gpu_sharding = self.pinn_sharding.with_memory_kind('device')
            self.device_shard = self.shard_batch_gpu
        

    def __len__(self): 
        return self.num_batches

    def __iter__(self):
        # Shuffle indices to avoid bias
        self.prng_key, _ = jax.random.split(self.prng_key)
        if isinstance(self.data['coords'], (np.ndarray, np.memmap)):
            indices = np.arange(self.num_samples)
            np.random.shuffle(indices)
        elif isinstance(self.data['coords'], jnp.ndarray):
            indices = jnp.arange(self.num_samples)
            indices = jax.random.permutation(self.prng_key, indices)

        for i in range(0, self.num_samples, self.chunk_size):
            batch_idx = indices[i: i + self.chunk_size]
            yield {k: self.device_shard(v[batch_idx]) for k, v in self.data.items()}

    def no_shard_batch(self, batch: Union[np.memmap, np.ndarray, jax.Array]) -> jax.Array:
        return batch

    def shard_batch_gpu(self, batch: Union[np.memmap, np.ndarray, jax.Array]) -> jax.Array:
        return jax.device_put(batch, self.gpu_sharding)