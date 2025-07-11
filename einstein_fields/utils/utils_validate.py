import jax
import jax.numpy as jnp
import optax
from .mini_batch import MiniBatchTensorDatasetDict
from .utils_norm import minkowski_norm_sq, divergence_papuc_operator_canonical
from differential_geometry import diffgeo, vdiffgeo
from data_generation.utils_generate_data import return_metric_fn

def validate(config, model, params, data, data_config, key) -> dict:

    validation_dict = {}

    validation_dict.update({
        'metric_mse_val': [],
        'jacobian_mse_val': [],
        'hessian_mse_val': [],
        'ricci_mse_val': [],
    })

    if config.training.norm == "minkowski":
        validation_dict['metric_minkowski_val'] = []
    elif config.training.norm == "papuc":
        validation_dict['metric_papuc_val'] = []

    full_metric_fn = return_metric_fn(data_config['metric'], 'full', data_config['coordinate_system'], data_config['metric_args'])
    distortion_fn = return_metric_fn(data_config['metric'], 'distortion', data_config['coordinate_system'], data_config['metric_args'])

    flat_metric = lambda coords: full_metric_fn(coords) - distortion_fn(coords)

    loader = MiniBatchTensorDatasetDict(data, config.wandb.validation_num_batches)

    @jax.jit
    def validation_step(params, data, subkey):
        coords = data["coords"]
        batch_validation_dict = {}
        if config.training.metric_type.endswith('_sym'):
            metric_pred = model.v_para_metric_sym(params, coords)
            jacobian_pred = model.v_para_metric_sym_jacobian(params, coords)
            hessian_pred = model.v_para_metric_sym_hessian(params, coords)
        else:
            metric_pred = model.v_para_metric(params, coords)
            jacobian_pred = model.v_para_metric_jacobian(params, coords)
            hessian_pred = model.v_para_metric_hessian(params, coords)
        
        batch_validation_dict['metric_mse_val'] = optax.l2_loss(data["metric"], metric_pred.reshape(-1, data['metric'].shape[-1])).mean()
        batch_validation_dict['jacobian_mse_val'] = optax.l2_loss(data["jacobian"], jacobian_pred.reshape(-1, data['jacobian'].shape[-1])).mean()
        batch_validation_dict['hessian_mse_val'] = optax.l2_loss(data["hessian"], hessian_pred.reshape(-1, data['hessian'].shape[-1])).mean()
        if config.training.metric_type.startswith('distortion'):
            para_metric_fn = lambda coords: flat_metric(coords) + model.para_metric(params, coords)
        else:
            para_metric_fn = lambda coords: model.para_metric(params, coords)

        spacetime = vdiffgeo(diffgeo(para_metric_fn))

        batch_validation_dict['ricci_mse_val'] = jnp.square(spacetime.ricci_tensor(coords)).mean()

        if config.training.norm == 'minkowski':
            if config.training.metric_type.startswith('distortion'):
                full_metric = data['metric_full'].reshape(-1, 4, 4)
            else:
                full_metric = data["metric"].reshape(-1, 4, 4)
            if config.training.integration:
                minkowski_val = jnp.dot(jax.vmap(minkowski_norm_sq, in_axes=[0, 0])(metric_pred - data['metric'].reshape(-1, 4, 4), full_metric), data["inv_vol_els"])
            else:
                minkowski_val = jax.vmap(minkowski_norm_sq, in_axes=[0, 0])(metric_pred - data['metric'].reshape(-1, 4, 4), full_metric).mean()
            batch_validation_dict['metric_minkowski_val'] = minkowski_val
        elif config.training.norm == 'papuc':
            if config.training.metric_type.startswith('distortion'):
                if config.training.integration:
                    papuc_val = jnp.dot(jax.vmap(divergence_papuc_operator_canonical, in_axes=[0, 0, None, None])(metric_pred + (data['metric_full'].reshape(-1, 4, 4) - data['metric'].reshape(-1, 4, 4)), data['metric_full'].reshape(-1, 4, 4), jnp.array([-1, 0, 0, 0]), subkey), data["inv_vol_els"])
                else:
                    papuc_val = jax.vmap(divergence_papuc_operator_canonical, in_axes=[0, 0, None, None])(metric_pred + (data['metric_full'].reshape(-1, 4, 4) - data['metric'].reshape(-1, 4, 4)), data['metric_full'].reshape(-1, 4, 4), jnp.array([-1, 0, 0, 0]), subkey).mean()
            elif config.training.metric_type.startswith('full_flatten'):
                if config.training.integration:
                    papuc_val = jnp.dot(jax.vmap(divergence_papuc_operator_canonical, in_axes=[0, 0, None, None])(metric_pred, data['metric'].reshape(-1, 4, 4), jnp.array([-1., 0, 0., 0.]), subkey), data["inv_vol_els"])
                else:
                    papuc_val = jax.vmap(divergence_papuc_operator_canonical, in_axes=[0, 0, None, None])(metric_pred, data['metric'].reshape(-1, 4, 4), jnp.array([-1., 0, 0., 0.]), subkey).mean()
                batch_validation_dict['metric_papuc_val'] = papuc_val

        return batch_validation_dict
    
    for _, data_batched in enumerate(loader):
        key, subkey = jax.random.split(key)  

        batch_validation_dict = validation_step(params, data_batched, subkey)

        for k, v in batch_validation_dict.items():
            validation_dict[k].append(v)

    if config.training.integration:
        volume = jnp.sum(data["inv_vol_els"])
    
    for k, v in validation_dict.items():
        if isinstance(v, list):
            validation_dict[k] = jnp.array(v)
            if k == 'metric_minkowski_val' or k == 'metric_papuc_val':
                if config.training.integration:
                    validation_dict[k] = jnp.sum(validation_dict[k]) / volume
                else:
                    validation_dict[k] = jnp.mean(validation_dict[k])
            else:
                validation_dict[k] = jnp.mean(validation_dict[k])
    
    return validation_dict