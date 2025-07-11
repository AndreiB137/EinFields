import jax
import jax.numpy as jnp

def take_symmetric_metric(metric : jnp.ndarray) -> jnp.ndarray:
    """returns only the symmetric components of the metric tensor [d, d] -> [d*(d+1)/2]"""
    d = metric.shape[-1]
    tri_met_idx = jnp.triu_indices(d, k=0)
    return metric[tri_met_idx[0], tri_met_idx[1]].flatten()


def take_symmetric_jacobian(jacobian: jnp.ndarray) -> jnp.ndarray:
    """ [k, d, d] -> [k, d*(d+1)/2] """
    d = jacobian.shape[-1]
    tri_jac_idx = jnp.triu_indices(d, k=0)
    return jacobian[..., tri_jac_idx[0], tri_jac_idx[1]].flatten()


def take_symmetric_hessian(hessian: jnp.ndarray) -> jnp.ndarray:
    """ [k, k, d, d] -> [k*(k+1)/2, d*(d+1)/2] """
    id_der = jnp.triu_indices(hessian.shape[0], k=0)
    hessian =  hessian[id_der[0], id_der[1], ...]
    id_comp = jnp.triu_indices(hessian.shape[-1], k=0) 
    return hessian[..., id_comp[0], id_comp[1]].flatten()

# %%%%%%%%%%%%%%%%%%%%%%% reconstruct from tensor symmetric arrays to full tensors %%%%%%%%%%%%%%%%%%%%%
 
def reconstruct_full_metric(metric_sym: jax.Array, n : int) -> jax.Array: 
    """returns the fully reconstructed (n, n) metric tensor from the symmetry reduced metric""" 
    i, j = jnp.triu_indices(n, k=0)
    matrix = jnp.zeros((n, n))
    matrix = matrix.at[i, j].set(metric_sym)
    matrix = matrix.at[j, i].set(metric_sym)
    return matrix

def reconstruct_full_metric_jacobian(metric_jacobian_sym: jax.Array, dims: tuple) -> jax.Array: 
    """returns the fully constructed metric jacobian from the symmetry reduced jacobian"""
    input_dim, output_dim = dims
    jacobian = jnp.zeros((input_dim, output_dim, output_dim))
    i, j = jnp.triu_indices(output_dim, k=0)
    jacobian = jacobian.at[:, i, j].set(metric_jacobian_sym.reshape(input_dim, -1))
    jacobian = jacobian.at[:, j, i].set(metric_jacobian_sym.reshape(input_dim, -1))
    return jacobian

def reconstruct_full_metric_hessian(metric_hessian_sym: jax.Array, dims: tuple) -> jax.Array: 
    """returns the fully constructed metric hessian from the symmetry reduced hessian"""

    input_dim, output_dim = dims
    hessian = jnp.zeros((input_dim, input_dim, output_dim, output_dim))
    i_out, j_out = jnp.triu_indices(output_dim, k=0)
    i_in, j_in = jnp.triu_indices(input_dim, k=0)
    sym_in_dim = input_dim * (input_dim + 1) // 2
    sym_out_dim = output_dim * (output_dim + 1) // 2

    in_idx_mesh, out_idx_mesh = jnp.meshgrid(jnp.arange(sym_in_dim), jnp.arange(sym_out_dim), indexing='ij')

    hessian = hessian.at[i_in[in_idx_mesh], j_in[in_idx_mesh], i_out[out_idx_mesh], j_out[out_idx_mesh]].set(metric_hessian_sym.reshape(sym_in_dim, sym_out_dim))
    hessian = hessian.at[i_in[in_idx_mesh], j_in[in_idx_mesh], j_out[out_idx_mesh], i_out[out_idx_mesh]].set(metric_hessian_sym.reshape(sym_in_dim, sym_out_dim))
    hessian = hessian.at[j_in[in_idx_mesh], i_in[in_idx_mesh], i_out[out_idx_mesh], j_out[out_idx_mesh]].set(metric_hessian_sym.reshape(sym_in_dim, sym_out_dim))
    hessian = hessian.at[j_in[in_idx_mesh], i_in[in_idx_mesh], j_out[out_idx_mesh], i_out[out_idx_mesh]].set(metric_hessian_sym.reshape(sym_in_dim, sym_out_dim))

    return hessian


if __name__=="__main__":
    # coords_grid = np.load("/system/user/publicwork/crangano/grnef_data/cartesian_kerr/coord_grid.npy", mmap_mode="r")
    # metric_grid = np.load("/system/user/publicwork/crangano/grnef_data/cartesian_kerr/metric_grid.npy", mmap_mode="r")
    # jacobian_grid = np.load("/system/user/publicwork/crangano/grnef_data/cartesian_kerr/jacobian_grid.npy", mmap_mode="r")
    # hessian_grid =  np.load("/system/user/publicwork/crangano/grnef_data/cartesian_kerr/hessian_grid.npy", mmap_mode="r")

    # dims = (int(3), int(4)) ## for the current Kerr usecase
    # input_dims, output_dims = dims[0], dims[1]

    i, j = jnp.triu_indices(4, k=0)

    metric = jax.random.uniform(jax.random.PRNGKey(0), (100, 4, 4, 4, 4))

    metric = metric.at[:, :, :, j, i].set(metric[:, :, :, i, j])
    metric = metric.at[:, j, i, :, :].set(metric[:, i, j, :, :])  # ensure symmetry

    # print(metric)
    # print(jax.vmap(take_symmetric_hessian)(metric).shape)
    # print(take_symmetric_jacobian(metric).shape)
    # print(jnp.equal(reconstruct_full_metric_hessian(take_symmetric_hessian(metric), (4, 4)), metric).all())
    # print(jax.vmap(take_symmetric_hessian)(metric).shape)
    print(jnp.equal(jax.vmap(reconstruct_full_metric_hessian, in_axes=[0, None])(jax.vmap(take_symmetric_hessian)(metric), (4, 4)), metric).all())
    # print(reconstruct_full_metric_hessian(take_symmetric_hessian(metric), (4, 4))[0])
    # coords_rows = coords_grid.reshape(-1, input_dims) 
    # metric_rows = metric_grid.reshape(-1, output_dims, output_dims)
    # jacobian_rows = jacobian_grid.reshape(-1, input_dims, output_dims, output_dims)
    # hessian_rows = hessian_grid.reshape(-1, input_dims, input_dims, output_dims, output_dims)
        
    # metric_sym = jax.vmap(take_symmetric_metric)(metric_rows)
    # jacobian_sym = jax.vmap(take_symmetric_jacobian)(jacobian_rows)
    # hessian_sym = jax.vmap(take_symmetric_hessian)(hessian_rows) 

    # %%%%% reproduce full differential geometry tensors %%%%
    # metric_vals = jax.vmap(reconstruct_full_metric, in_axes=(0, None))(metric_sym, dims) 
    # jacobian_vals = jax.vmap(reconstruct_full_metric_jacobian, in_axes=(0, None))(jacobian_sym, dims) 
    # hessian_vals = jax.vmap(reconstruct_full_metric_hessian, in_axes=(0, None))(hessian_sym, dims) 
    
    # ## asserting if we are regenerating the correct one again ## 
    # import tqdm
    # from tqdm import tqdm
    # for i in tqdm(range(len(hessian_vals))):
    #     hessian_diff = hessian_vals[i] - hessian_rows[i].flatten() 
    #     print(jnp.max(hessian_diff), jnp.min(hessian_diff))





