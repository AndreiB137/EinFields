import logging
import jax
import yaml
import jax.numpy as jnp
import optax
import os
from ml_collections import ConfigDict, FrozenConfigDict
import kfac_jax
import orbax.checkpoint as ocp
from ..nn_models import get_model
from typing import Any, Callable
import flax.linen as nn
from ..nn_models import get_activation
from .utils_config_grad import get_alignment
from .utils_symmetry import reconstruct_full_metric
from .utils_configs_opt_schedules import get_optimizer, get_scheduler
from .utils_norm import minkowski_norm_sq, divergence_papuc_operator_canonical

def store_config(run_dir: str, config: dict | ConfigDict, file_name: str) -> None:
    """
    Store configuration dictionary to a human-readable YAML file.
    
    Args:
        run_dir (str): Directory path where the config file will be stored
        config (dict or ConfigDict): Configuration object to save
        file_name (str): Name of the config file (without extension)
        
    Note:
        Creates a .yml file in the specified directory. Handles both dict
        and ConfigDict objects by converting ConfigDict to dict if needed.
    """
    config_path = os.path.join(run_dir, f"{file_name}.yml")
    with open(config_path, 'w') as f:
        if isinstance(config, dict):
            yaml.dump(config, f, sort_keys=False)
        else:
            yaml.dump(config.to_dict(), f, sort_keys=False)

def load_config(run_dir: str) -> dict:
    """
    Load configuration dictionary from a YAML file.
    
    Args:
        run_dir (str): Directory path containing the config.yml file
        
    Returns:
        dict: Configuration dictionary loaded from the YAML file
        
    """
    config_path = os.path.join(run_dir, f"config.yml")
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def get_model_and_params_checkpoint(run_dir: str) -> tuple[nn.Module, dict]:
    """
    Load a trained model and its parameters from a checkpoint directory.
    Works only with orbax checkpoints and assumes only one checkpoint exists.
    
    Args:
        run_dir (str): Directory containing both config.yml and checkpoint folder
        
    Returns:
        tuple: (model, params) where:
            - model: Instantiated flax nn.Module object
            - params: Model parameters loaded from checkpoint
            
    Note:
        Automatically handles sharding files cleanup to ensure reproducibility
        over multiple user hardware configurations.

    """

    config = load_config(run_dir)

    options = ocp.CheckpointManagerOptions(
        enable_async_checkpointing=False,
        read_only=True,
    )

    checkpoint_path = os.path.join(run_dir, f"checkpoint")
    if os.path.exists(os.path.join(checkpoint_path, "0", "params", "_sharding")):
        os.remove(os.path.join(checkpoint_path, "0", "params", "_sharding"))
    mngr = ocp.CheckpointManager(checkpoint_path, options=options)

    checkpoint_dict = mngr.restore(
        mngr.latest_step(),
        args=ocp.args.Composite(
            params=ocp.args.StandardRestore()
        ),
    )

    params = checkpoint_dict['params']
    model = make_model(ConfigDict(config).architecture)

    return model, params

def store_checkpoint(run_dir: str, params: dict, opt_state: Any, epoch: int, last_lr: float) -> None:
    """
    Save training checkpoint to disk with model parameters and optimizer state.
    Additionally, store metadata like last epoch and last learning rate at the end of training.
    
    Args:
        params (dict): Model parameters to save
        opt_state (Any/PyTree): Optimizer state to save
        epoch (int): Last training epoch
        last_lr (float): Last learning rate at the end of training
        
    Note:
        Creates checkpoint directory if it doesn't exist. Saves parameters,
        optimizer state, and metadata (epoch, learning rate) separately.
    """
    checkpoint_dict = {
        'params' : params,
        'opt_state' : opt_state,
        'metadata': {
            'last_epoch': epoch,
            'last_lr': last_lr,
        }
    }
    checkpoint_path = os.path.join(run_dir, f"checkpoint")
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    
    # options = ocp.CheckpointManagerOptions(max_to_keep=1, create=True)
    options = ocp.CheckpointManagerOptions(
        enable_async_checkpointing=False
    )

    logging.info(f"Storing checkpoint at {checkpoint_path}")
    
    mngr = ocp.CheckpointManager(checkpoint_path,
                                 options=options)
    
    mngr.save(
        0,
        args=ocp.args.Composite(
            params=ocp.args.StandardSave(checkpoint_dict['params']),
            opt_state=ocp.args.StandardSave(checkpoint_dict['opt_state']),
            metadata=ocp.args.JsonSave(checkpoint_dict['metadata'])
        ),
    )



def load_checkpoint(checkpoint_path : str, opt_state_struct: Any) -> tuple[int, dict, Any]:
    """
    Load training checkpoint from disk. Usually to be used when starting
    training from a checkpoint. The optimizer state has a complex structure
    that needs to be provided to orbax for correct restoration. It requires
    only the optimizer state returned by the `init` method. Assumes the checkpoint
    directory contains a single orbax checkpoint with the latest step.
    
    Args:
        checkpoint_path (str): Path to checkpoint directory
        opt_state_struct: Structure template for optimizer state restoration
        
    Returns:
        tuple: (last_epoch, params, opt_state) where:
            - last_epoch (int): Last completed training epoch
            - params (dict): Restored model parameters
            - opt_state (Any/PyTree): Restored optimizer state
            
    Note:
        Automatically cleans up sharding files for both params and opt_state
        before loading.
    """

    if os.path.exists(os.path.join(checkpoint_path, "0", "params", "_sharding")):
        os.remove(os.path.join(checkpoint_path, "0", "params", "_sharding"))
    if os.path.exists(os.path.join(checkpoint_path, "0", "opt_state", "_sharding")):
        os.remove(os.path.join(checkpoint_path, "0", "opt_state", "_sharding"))

    options = ocp.CheckpointManagerOptions(
        enable_async_checkpointing=False
    )

    mngr = ocp.CheckpointManager(checkpoint_path, options=options)

    checkpoint_dict = mngr.restore(
        mngr.latest_step(),
        args=ocp.args.Composite(
            params=ocp.args.StandardRestore(),
            opt_state=ocp.args.StandardRestore(opt_state_struct),
            metadata=ocp.args.JsonRestore()
        ),
    )

    ## Return the saved state
    return checkpoint_dict['metadata']['last_epoch'], checkpoint_dict['params'], checkpoint_dict['opt_state']


def load_data(config: FrozenConfigDict | ConfigDict) -> tuple[dict, dict]:
    """
    Load training and validation data from disk based on configuration.
    
    Args:
        config (ConfigDict): Configuration containing data paths and settings
        
    Returns:
        tuple[dict, dict]: (training_data, validation_data) where each dict contains:
            - coords: Coordinate arrays
            - metric: Metric tensors
            - inv_vol_els: Invariant volume elements (if integration enabled)
            - metric_full: Full metric (for minkowski/papuc norms with distortion)
            - jacobian/hessian: Additional data based on loss configuration
            
    Note:
        Data loading is conditional based on:
        - Metric type (symmetric vs non-symmetric)
        - Training norm (mse, minkowski, papuc)
        - Integration flag
        - Loss function requirements (jacobian, hessian)
    """

    logging.info(f"Loading data from {config.data.data_dir}")

    data_dir = config.data.data_dir
    
    type = config.training.metric_type.split('_sym')[0]
    if config.training.metric_type.endswith("_sym"):
        metric_type_dir = os.path.join(data_dir, type, "symmetric")
    else:
        metric_type_dir = os.path.join(data_dir, type)

    training_dir = os.path.join(metric_type_dir, "training")
    validation_dir = os.path.join(metric_type_dir, "validation")
    
    
    logging.info(f"Loading metric:")


    training_data = {
        "coords": jnp.load(os.path.join(data_dir, "coords_train.npy")),
        "metric": jnp.load(os.path.join(training_dir, "metric.npy")),
    }

    logging.info(f"Metric: {training_data['metric'].shape}")

    validation_data = {}

    if 'wandb' in config:
        validation_data = {
            "coords": jnp.load(os.path.join(data_dir, "coords_validation.npy")),
            "metric": jnp.load(os.path.join(validation_dir, "metric.npy")),
        }

    if config.training.integration:
        logging.info(f"Loading invariant volume elements.")
        training_data.update({
            "inv_vol_els": jnp.load(os.path.join(data_dir, "inv_volume_measure_train.npy")),
        })
        validation_data.update({
            "inv_vol_els": jnp.load(os.path.join(data_dir, "inv_volume_measure_validation.npy")),
        })

    if config.training.norm == "minkowski" or config.training.norm == "papuc":
        if config.training.metric_type == "distortion":
            logging.info(f"Loading full metric for {config.training.norm} norm.")
            training_data.update({
                "metric_full": jnp.load(os.path.join(data_dir, "full_flatten", "training", "metric.npy")),
            })
            validation_data.update({
                "metric_full": jnp.load(os.path.join(data_dir, "full_flatten", "validation", "metric.npy")),
            })

    for k, v in config.data.losses.items():
        if v:
            logging.info(f"Loading {k.capitalize()} data for training.")
            training_data[k] = jnp.load(os.path.join(training_dir, f"{k}.npy"))
            logging.info(f"{k.capitalize()}: {training_data[k].shape}")

    if 'wandb' in config:
        logging.info(f"Loading Jacobian and Hessian for validation.")
        for file in os.listdir(validation_dir):
            if file in ['jacobian.npy', 'hessian.npy']:
                validation_data[file.split('.npy')[0]] = jnp.load(os.path.join(validation_dir, file))

    logging.info(f"Loading complete.")
    
    return training_data, validation_data


def make_model(config: FrozenConfigDict | ConfigDict) -> nn.Module:
    """
    Create and configure a flax neural network nn.Module based on configuration.

    Args:
        config: Model configuration containing:
            - model_name: Type of model (MLP, SIREN, WIRE, etc.)
            - output_dim: Number of output dimensions
            - hidden_dim: Hidden layer dimensions
            - num_layers: Number of layers
            - activation: Activation function name
            - extra_model_args: Additional model-specific arguments
            
    Returns:
        model (nn.Module): Configured flax nn.Module instance ready for training
        
    Note:
        Handles special case for WIRE and SIREN models which don't accept
        activation functions as an argument.
    """

    model = get_model(config.model_name)
    if config.extra_model_args is not None:
        if config.model_name == "WIRE" or config.model_name == "SIREN":
            model = model(output_dim=config.output_dim,
                          hidden_dim=config.hidden_dim,
                          num_layers=config.num_layers,
                          **config.extra_model_args)
        else:
            model = model(output_dim=config.output_dim,
                        hidden_dim=config.hidden_dim, 
                        num_layers=config.num_layers,
                        act=get_activation(config.activation),
                        **config.extra_model_args)
    else:
        model = model(output_dim=config.output_dim, 
                      hidden_dim=config.hidden_dim, 
                      num_layers=config.num_layers,
                      act=get_activation(config.activation),
    )   
        
    return model


def init_params(model: nn.Module, dim: int, rng: jax.random.key = None):
    """
    Initialize model parameters.
    
    Args:
        model (nn.Module): Neural network flax nn.Module instance
        dim (int): Input dimension for dummy data
        rng: JAX random key (defaults to key(0) if None)
        
    Returns:
        params (dict): Initialized model parameters
        
    Note:
        Uses dummy input of ones with specified dimension to trigger
        parameter initialization.
    """
    if rng is None:
        rng = jax.random.key(0)
    dummy_input = jnp.ones(dim)
    params = model.init(rng, dummy_input)
    return params 


def make_sym_callables(model: nn.Module):
    """
    Add symmetric metric computation functions to model as attributes.
    
    Args:
        model (nn.Module): Flax nn.Module instance to augment with callable functions
        
    Returns:
        None (modifies model in-place)
        
    Adds attributes:
        - para_metric_sym: Compute symmetric metric
        - para_metric_sym_jacobian: Compute Jacobian of symmetric metric
        - para_metric_sym_hessian: Compute Hessian of symmetric metric
        - v_* variants: Vectorized versions of above functions
        
    Note:
        Uses upper triangular indices for symmetric 4x4 matrices. Works only for 4x4 metrics.
    """
    i, j = jnp.triu_indices(4, k=0)
    model.para_metric_sym = lambda params, coords: model.apply(params, coords)
    model.para_metric_sym_jacobian = lambda params, coords: jnp.transpose(jax.jacfwd(model.para_metric_sym, argnums=1)(params, coords), [1, 0])
    model.para_metric_sym_hessian = lambda params, coords: jnp.transpose(jax.jacfwd(model.para_metric_sym_jacobian, argnums=1)(params, coords), [2, 0, 1])[i, j, :]

    model.v_para_metric_sym = jax.vmap(model.para_metric_sym, in_axes=[None, 0])
    model.v_para_metric_sym_jacobian = jax.vmap(model.para_metric_sym_jacobian, in_axes=[None, 0])
    model.v_para_metric_sym_hessian = jax.vmap(model.para_metric_sym_hessian, in_axes=[None, 0])

def make_callables(config: FrozenConfigDict | ConfigDict, model: nn.Module) -> None:
    """
    Add metric computation functions to model based on configuration.
    
    Args:
        config (FrozenConfigDict | ConfigDict): Configuration specifying output dimensions and metric type
        model (nn.Module): Model instance to augment with callable functions
        
    Returns:
        None (modifies model in-place)
        
    Adds attributes:
        - para_metric: Compute 4x4 metric tensor
        (if model's output dimension is 10 then it reconstructs the 4x4 metric, to be used for validation
        or debugging when testing with the `DifferentialGeometry` class)
        - para_metric_jacobian: Compute metric Jacobian
        - para_metric_hessian: Compute metric Hessian  
        - v_* variants: Vectorized versions for batch processing
        
    Note:
        Handles both full (16D) and symmetric (10D) output dimensions.
        Calls `make_sym_callables` for symmetric metric types.
    """
   
    if config.architecture.output_dim == 16:
        model.para_metric = lambda params, coords: model.apply(params, coords).reshape(4,4)
    elif config.architecture.output_dim == 10:
        model.para_metric = lambda params, coords: reconstruct_full_metric(model.apply(params, coords), 4).reshape(4,4) # Used for validation
    
    model.para_metric_jacobian = lambda params, coords: jnp.transpose(jax.jacfwd(model.para_metric, argnums=1)(params, coords), [2, 0, 1])
    model.para_metric_hessian = lambda params, coords: jnp.transpose(jax.jacfwd(model.para_metric_jacobian, argnums=1)(params, coords), [3, 0, 1, 2])

    model.v_para_metric = jax.vmap(model.para_metric, in_axes=[None, 0])
    model.v_para_metric_jacobian = jax.vmap(model.para_metric_jacobian, in_axes=[None, 0])
    model.v_para_metric_hessian = jax.vmap(model.para_metric_hessian, in_axes=[None, 0])

    if config.training.metric_type.endswith("_sym"):
        make_sym_callables(model)
    

def make_loss(config: dict, model: nn.Module) -> Callable:
    """
    Create loss functions based on configuration and training requirements.
    
    Args:
        config (dict): Training configuration specifying:
            - norm: Loss norm type (mse, minkowski, papuc)
            - integration: Whether to use invariant volume elements
            - metric_type: Type of metric (full, distortion, symmetric, etc.)
            - gradient_conflict: Gradient conflict resolution method
        model (nn.Module): Model with callable functions already attached
        
    Returns:
        callable: Loss function with signature (params, data, key) -> (loss, losses)
                 Or (params, data, key) -> (grads, loss, losses, alignment) for grad_norm
        
    Note:
        Returns different loss functions based on gradient_conflict setting:
        - None: Standard combined loss
        - "grad_norm": Normalized gradient direction loss
        
        Supports multiple supervision targets (metric, jacobian, hessian).
    """
    if config.training.norm=="mse":
        if config.training.integration:
            loss_fnc_metric = lambda preds, truths, inv_vol_els: jnp.dot(optax.l2_loss(preds, truths).mean(axis=[1,2]), inv_vol_els)
        else:
            loss_fnc_metric = lambda preds, truths: optax.l2_loss(preds, truths).mean()
    elif config.training.norm=="minkowski":
        if config.training.integration:
            loss_fnc_metric = lambda preds, truths, full_metric, inv_vol_els: jnp.dot(jax.vmap(minkowski_norm_sq, in_axes=[0, 0])(preds-truths, full_metric), inv_vol_els)
        else:
            loss_fnc_metric = lambda preds, truths, full_metric: jax.vmap(minkowski_norm_sq, in_axes=[0, 0])(preds-truths, full_metric).mean()
    elif config.training.norm=="papuc":
        if config.training.metric_type == "distortion":
            if config.training.integration:
                loss_fnc_metric = lambda preds, truths, full_metric, inv_vol_els, key: jnp.dot(jax.vmap(divergence_papuc_operator_canonical, in_axes=[0, 0, None, None])(preds + (full_metric - truths), full_metric, jnp.array([-1, 0, 0, 0]), key), inv_vol_els)
            else:
                loss_fnc_metric = lambda preds, truths, full_metric, key: jax.vmap(divergence_papuc_operator_canonical, in_axes=[0, 0, None, None])(preds + (full_metric - truths), full_metric, jnp.array([-1, 0, 0, 0]), key).mean()
        else:
            if config.training.integration:
                loss_fnc_metric = lambda preds, full_metric, inv_vol_els, key: jnp.dot(jax.vmap(divergence_papuc_operator_canonical, in_axes=[0, 0, None, None])(preds, full_metric, jnp.array([-1, 0, 0, 0]), key), inv_vol_els)
            else:
                loss_fnc_metric = lambda preds, full_metric, key: jax.vmap(divergence_papuc_operator_canonical, in_axes=[0, 0, None, None])(preds, full_metric, jnp.array([-1, 0, 0, 0]), key).mean()
    
    def loss_metric(params, data, key):
        """Compute the loss with supervision on the metric."""
        preds = model.v_para_metric(params, data["coords"]).reshape(-1, 4, 4)
        truths = data["metric"].reshape(-1, 4, 4)
        
        if config.training.norm == "mse":
            if config.training.integration:
                loss = loss_fnc_metric(preds, truths, data["inv_vol_els"])
            else:
                loss = loss_fnc_metric(preds, truths)
        elif config.training.norm == "minkowski":
            if config.training.metric_type.startswith("distortion"):
                full_metric = data["metric_full"].reshape(-1, 4, 4)
            else:
                full_metric = data["metric"].reshape(-1, 4, 4)
            if config.training.integration:
                loss = loss_fnc_metric(preds, truths, full_metric, data["inv_vol_els"])
            else:
                loss = loss_fnc_metric(preds, truths, full_metric)
        elif config.training.norm == "papuc":
            if config.training.metric_type == "distortion":
                if config.training.integration:
                    loss = loss_fnc_metric(preds, truths, data["metric_full"].reshape(-1, 4, 4), data["inv_vol_els"], key)
                else:
                    loss = loss_fnc_metric(preds, truths, data["metric_full"].reshape(-1, 4, 4), key)
            else:
                if config.training.integration:
                    loss = loss_fnc_metric(preds, truths, data["inv_vol_els"], key)
                else:
                    loss = loss_fnc_metric(preds, truths, key)

        return loss
    
    def loss_metric_sym(params, data):
        """Compute the loss with supervision on the metric, using symmetry."""
        preds_sym = model.v_para_metric_sym(params, data["coords"])
        truths = data["metric"]
        
        if config.training.integration:
            loss = jnp.dot(optax.l2_loss(preds_sym, truths).mean(axis=-1), data["inv_vol_els"]) 
        else:
            loss = optax.l2_loss(preds_sym, truths).mean()

        return loss

    def loss_jacobian(params, data):
        """Compute the loss with supervision on the Jacobian."""
        preds_jac = model.v_para_metric_jacobian(params, data["coords"]).reshape(-1, data["jacobian"].shape[-1])
        if config.training.integration:
            loss = jnp.dot(optax.l2_loss(preds_jac, data["jacobian"]).mean(axis=-1), data["inv_vol_els"])
        else:
            loss = optax.l2_loss(preds_jac, data["jacobian"]).mean()
        return loss
    
    def loss_jacobian_sym(params, data):
        """Compute the loss with supervision on the Jacobian, using symmetry."""
        preds_jac_sym = model.v_para_metric_sym_jacobian(params, data["coords"]).reshape(-1, data["jacobian"].shape[-1])
        if config.training.integration:
            loss = jnp.dot(optax.l2_loss(preds_jac_sym, data["jacobian"]).mean(axis=-1), data["inv_vol_els"])
        else:
            loss = optax.l2_loss(preds_jac_sym, data["jacobian"]).mean()
        return loss
    
    def loss_hessian(params, data):
        """Compute the loss with supervision on the Hessian."""
        preds_hes = model.v_para_metric_hessian(params, data["coords"]).reshape(-1, data["hessian"].shape[-1])
        if config.training.integration:
            loss = jnp.dot(optax.l2_loss(preds_hes, data["hessian"]).mean(axis=-1), data["inv_vol_els"])
        else:
            loss = optax.l2_loss(preds_hes, data["hessian"]).mean()
        return loss
    
    def loss_hessian_sym(params, data):
        """Compute the loss with supervision on the Hessian, using symmetry."""
        preds_hes_sym = model.v_para_metric_sym_hessian(params, data["coords"]).reshape(-1, data["hessian"].shape[-1])
        if config.training.integration:
            loss = jnp.dot(optax.l2_loss(preds_hes_sym, data["hessian"]).mean(axis=-1), data["inv_vol_els"])
        else:
            loss = optax.l2_loss(preds_hes_sym, data["hessian"]).mean()
        return loss
    
    if config.training.gradient_conflict == None:
        def loss_fn(params, data, key):
            """Compute the loss with optional supervision on the Jacobian and the Hessian."""

            total_loss = 0 ## total loss for backprop
            losses = {} ## individual losses for logging
            
            """Metric"""
            if config.training.metric_type.endswith("_sym"):
                losses["metric"] = loss_metric_sym(params, data)
            else:
                losses["metric"] = loss_metric(params, data, key)
            total_loss += losses["metric"]

            """Jacobian"""
            if config.data.losses.jacobian:
                if config.training.metric_type.endswith("_sym"):
                    losses["jacobian"] = loss_jacobian_sym(params, data)
                else:
                    losses["jacobian"] = loss_jacobian(params, data)
                total_loss += losses["jacobian"]
                            

            """Hessian"""
            if config.data.losses.hessian:
                if config.training.metric_type.endswith("_sym"):
                    losses["hessian"] = loss_hessian_sym(params, data)
                else:
                    losses["hessian"] = loss_hessian(params, data)
                total_loss += losses["hessian"]

            if config.optimizer.name=="kfac":
                if config.training.metric_type.endswith("_sym"):
                    preds_metric = model.v_para_metric_sym(params, data["coords"]).reshape(-1, 10)
                else:
                    preds_metric = model.v_para_metric(params, data["coords"]).reshape(-1, 16)
                kfac_jax.register_squared_error_loss(
                    preds_metric, 
                    data["metric"],
                    weight=1 / data["metric"].shape[-1]
                    )
                if config.data.losses.jacobian:
                    if config.training.metric_type.endswith("_sym"):
                        preds_jacobian = model.v_para_metric_sym_jacobian(params, data["coords"]).reshape(-1, data["jacobian"].shape[-1])
                    else:
                        preds_jacobian = model.v_para_metric_jacobian(params, data["coords"]).reshape(-1, data["jacobian"].shape[-1])
                    kfac_jax.register_squared_error_loss(
                        preds_jacobian, 
                        data["jacobian"],
                        weight=1 / data["jacobian"].shape[-1]
                        )
            
                if config.data.losses.hessian:
                    if config.training.metric_type.endswith("_sym"):
                        preds_hessian = model.v_para_metric_sym_hessian(params, data["coords"]).reshape(-1, data["hessian"].shape[-1])
                    else:
                        preds_hessian = model.v_para_metric_hessian(params, data["coords"]).reshape(-1, data["hessian"].shape[-1])
                    kfac_jax.register_squared_error_loss(
                        preds_hessian, 
                        data["hessian"],
                        weight=1 / data["hessian"].shape[-1]
                        )

            return total_loss, losses
        
        return loss_fn
    elif config.training.gradient_conflict == "grad_norm":
        def loss_grad_norm(params, data, key):
            """Compute the loss with normalized gradient direction."""
            loss = 0
            losses = {}
            grads = []
            if config.training.metric_type.endswith("_sym"):
                loss1, grads1 = jax.value_and_grad(loss_metric_sym)(params, data)
            else:
                loss1, grads1 = jax.value_and_grad(loss_metric)(params, data, key)
            grads.append(grads1)
            loss += loss1
            losses["metric"] = loss1
            if config.data.losses.jacobian:
                if config.training.metric_type.endswith("_sym"):
                    loss2, grads2 = jax.value_and_grad(loss_jacobian_sym)(params, data)
                else:
                    loss2, grads2 = jax.value_and_grad(loss_jacobian)(params, data)
                grads.append(grads2)
                loss += loss2
                losses["jacobian"] = loss2
            if config.data.losses.hessian:
                if config.training.metric_type.endswith("_sym"):
                    loss3, grads3 = jax.value_and_grad(loss_hessian_sym)(params, data)
                else:
                    loss3, grads3 = jax.value_and_grad(loss_hessian)(params, data)
                grads.append(grads3)
                loss += loss3
                losses["hessian"] = loss3
            
            grads1, unflatten = jax.flatten_util.ravel_pytree(grads[0])
            grads[0] = grads1
            for i in range(1, len(grads)):
                grads[i] = jax.flatten_util.ravel_pytree(grads[i])[0]

            grads = jnp.stack(grads, axis=0)
            alignment = get_alignment(grads)
            g = unflatten(jnp.sum(grads / jnp.linalg.norm(grads, axis=1, keepdims=True), axis=0))

            return g, loss, losses, alignment
        
        return loss_grad_norm

def make_train(config: FrozenConfigDict | ConfigDict, optimizer: Any, loss_fn: Callable) -> Callable:
    """
    Create training step function based on optimizer type and configuration.
    
    Args:
        config (FrozenConfigDict | ConfigDict): Training configuration
        optimizer (Any): Configured optimizer instance
        loss_fn: Loss function created by `make_loss`
        
    Returns:
        Callable: Training step function (jitted) with signature:
            (params, opt_state, data, rng) -> (params, opt_state, loss, losses[, alignment])
            
    Note:
        Creates different training steps for:
        - KFAC optimizer: Uses optimizer.step with special handling
        - LBFGS optimizer: Includes value_fn for line search
        - Standard optimizers: Uses optax update pattern
        - Gradient conflict resolution: Returns alignment information
    """
    if config.optimizer.name=="kfac":
        def train_step(params, opt_state, data, rng):
            params, opt_state, stats = optimizer.step(
                params,
                opt_state, 
                rng,
                batch=data,
                )
            return params, opt_state, stats['loss'], stats['aux']
        return train_step
    elif config.optimizer.name=="lbfgs":
        @jax.jit
        def train_step(params, opt_state, data, rng):
            (total_loss, losses), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, data, rng)
            updates, opt_state = optimizer.update(grads, opt_state, params, value=total_loss, grad=grads, value_fn= lambda params : loss_fn(params, data, rng)[0] )
            params = optax.apply_updates(params, updates)
            return params, opt_state, total_loss, losses
    else:
        if config.training.gradient_conflict is None:
            @jax.jit
            def train_step(params, opt_state, data, rng):
                (total_loss, losses), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, data, rng)
                updates, opt_state = optimizer.update(grads, opt_state, params)
                params = optax.apply_updates(params, updates)
                return params, opt_state, total_loss, losses
        else:
            @jax.jit
            def train_step(params, opt_state, data, rng):
                g, total_loss, losses, alignment = loss_fn(params, data, rng)
                updates, opt_state = optimizer.update(g, opt_state, params)
                params = optax.apply_updates(params, updates)
                return params, opt_state, total_loss, losses, alignment
    
    return train_step


def make_scheduler_optimizer(config: FrozenConfigDict | ConfigDict, loss_fn: Callable | None = None) -> tuple[Callable, Any]:
    """
    Create learning rate scheduler and optimizer instances.
    
    Args:
        config (FrozenConfigDict | ConfigDict): Optimizer configuration containing:
            - name: Optimizer type (adam, soap, kfac, lbfgs, etc.)
            - learning_rate: Base learning rate
            - lr_scheduler: Scheduler type
            - extra_optimizer_args: Additional optimizer arguments
            - extra_scheduler_args: Additional scheduler arguments
        loss_fn (Callable): Loss function (required for KFAC optimizer)
        
    Returns:
        tuple: (scheduler, optimizer) instances ready for training
               For KFAC/LBFGS: (dummy_scheduler, optimizer)
               
    Note:
        KFAC requires the loss function for initialization.
        KFAC and LBFGS don't use a learning rate schedule. They use adaptive learning rates.
    """
    optimizer = get_optimizer(config.optimizer.name) ## constructor
    optimizer_args = config.optimizer.extra_optimizer_args
    scheduler = get_scheduler(config.optimizer.lr_scheduler) ## constructor
    scheduler = scheduler(config.optimizer.learning_rate, **config.optimizer.extra_scheduler_args) ## instance
    if config.optimizer.name=="kfac":
        def kfac_loss_wrapper(params, data):
            return loss_fn(params, data, jax.random.key(0)) ## KFAC expects only params and data, not rng
        return lambda it : 0., optimizer(value_and_grad_func=jax.value_and_grad(kfac_loss_wrapper, has_aux=True), 
                                         **optimizer_args,
                                         value_func_has_aux=True,
                                         use_adaptive_learning_rate=True,
                                         use_adaptive_momentum=True,
                                         use_adaptive_damping=True)
    elif config.optimizer.name=="lbfgs":
        return lambda it : 0., optimizer(**optimizer_args)
    
    return scheduler, optimizer(scheduler, **optimizer_args) ## instances


def make_train_state(config: FrozenConfigDict | ConfigDict, model: nn.Module, optimizer: Any, data: dict, rng: jax.random.key = None) -> tuple[int, dict, Any]:
    """
    Initialize or restore training state (last epoch, parameters and optimizer state).
    
    Args:
        config (FrozenConfigDict | ConfigDict): Training configuration
        model: Flax nn.Module instance for parameter initialization
        optimizer: Optimizer for state initialization
        data: Training data dictionary to determine flax nn.Module parameter initialization
        rng: Random key for parameter initialization
        
    Returns:
        tuple: (start_epoch, params, opt_state) where:
            - start_epoch (int): Epoch to start/resume training from
            - params (dict): Model parameters (initialized or restored)
            - opt_state (Any/PyTree): Optimizer state (initialized or restored)
            
    Note:
        Behavior depends on config.checkpoint.load_dir:
        - None/empty: Initialize new training state
        - Valid path: Restore from checkpoint with options for:
            - Resetting optimizer state
            - Updating learning rate scheduler state and keeping optimizer statistics (if applicable, e.g. for Adam or Soap, but not for KFAC or LBFGS)
    """

    if config.checkpoint.load_dir:
        use_checkpoint = True
    else:
        use_checkpoint = False
    
    def init_opt_state(params: dict):
        if config.optimizer.name == "kfac":
            return optimizer.init(params, rng, data)
        else:
            return optimizer.init(params)
    
    if use_checkpoint is False:
        logging.info("Initializing a new run")
        start_epoch = 0
        params = init_params(model, data["coords"].shape[-1], rng)
        opt_state = init_opt_state(params)
    else:
        logging.info(f"Loading run from the checkpoint: {config.checkpoint.load_dir}")
        opt_name = config.optimizer.name
        test_params = init_params(model, data["coords"].shape[-1], rng)
        opt_state_struct = init_opt_state(test_params)
        start_epoch, params, opt_state = load_checkpoint(os.path.join(config.checkpoint.load_dir, f"checkpoint"), opt_state_struct)
        
        logging.info("Checking if the user wants to reset the stats of the optimizer...")
        if config.checkpoint.reset_optimizer:
            logging.info("Resetting the stats of the optimizer...")
            opt_state = init_opt_state(params)
        else:
            logging.info("Keeping the stats of the optimizer...")
            if opt_name != "kfac" and opt_name != "lbfgs":
                logging.info("Updating the learning rate scheduler...")
                opt_state = (*opt_state[:-1], optax.ScaleByScheduleState(0))
    
    return start_epoch, params, opt_state


def get_param_count(params: dict) -> int:
    """Counts the number of model parameters"""
    return sum(x.size for x in jax.tree.leaves(params))
