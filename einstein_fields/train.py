import jax
import jax.numpy as jnp
import logging 
import wandb
from tqdm.auto import trange
from .utils import (
    load_data,
    store_checkpoint,
    make_model,
    make_callables,
    make_loss,
    make_scheduler_optimizer,
    make_train_state,
    get_param_count,
    make_train,
    MiniBatchTensorDatasetDict
)
from .utils.utils_validate import validate
from ml_collections import FrozenConfigDict, ConfigDict


def train(config: FrozenConfigDict | ConfigDict, data_config: FrozenConfigDict | ConfigDict) -> None: 
    """
    Main neural field training function that handles complete training pipeline.
    
    Executes the full training loop for neural field models with support for:
    - Checkpoint loading and saving
    - Multiple loss functions (MSE, Minkowski, Papuc norms)
    - Various optimizers (Adam, SOAP, KFAC, LBFGS)
    - Gradient conflict resolution methods (only "grad_norm" supported)
    - Integration with Weights & Biases logging
    - Mini-batch training
    - Periodic validation during training
    
    Args:
        config (FrozenConfigDict | ConfigDict): Main training configuration containing:
            
        data_config (FrozenConfigDict | ConfigDict): Config from `data_generation` module.
    
    Training Pipeline:
        1. **Initialization Phase**:
           - Sets up JAX random number generator
           - Loads training and validation data from disk
           - Creates mini-batch data loader
           - Initializes neural network model
           - Configures model callable functions (metric, jacobian, hessian)
           
        2. **Setup Phase**:
           - Creates loss function based on configuration
           - Initializes optimizer and learning rate scheduler
           - Sets up training state (parameters, optimizer state)
           - Creates JIT-compiled training step function
           
        3. **Training Loop**:
           - Iterates through epochs with progress bar
           - For each epoch, processes all mini-batches
           - Accumulates losses across batches
           - Handles gradient conflict resolution if enabled
           - Logs metrics to Weights & Biases
           - Performs periodic validation
           
        4. **Finalization**:
           - Saves final checkpoint with trained parameters
           - Logs final training statistics
    
    Loss averaging:
        - With integration: Normalizes by sum of volume elements
        - Without integration: Normalizes by number of batches
    
    Gradient Conflict Handling:
        - None: Standard gradient descent with combined loss
        - "grad_norm": Normalized gradient direction method with alignment tracking
    
    Validation:
        - Runs validation every N epochs (configurable)
        - Computes comprehensive metrics on validation set
        - Logs validation results to Weights & Biases
    
    Checkpointing:
        - Supports loading from previous checkpoints
        - Saves final state with parameters, optimizer state, and metadata
        - Handles different optimizer types (KFAC, LBFGS, standard)
    
    Logging:
        - Per-epoch loss values and individual loss components
        - Model architecture details and parameter count
        - Gradient alignment metrics (if applicable)
        - Validation metrics at specified intervals
    
    Note:
        - Requires pre-processed data files in specified directory structure
        - Uses JAX for automatic differentiation and JIT compilation
        - Progress bar uses ncols=80 to prevent line breaking issues
        - Final learning rate is computed for checkpoint metadata
        
    Raises:
        FileNotFoundError: If data files are missing from configured paths
        ValueError: If configuration parameters are invalid or incompatible
    """

    ## RNG
    rng = jax.random.key(config.training.rng_seed)

    ## Data
    train_data, val_data = load_data(config)
    # config.data.config = ConfigDict(data_config)

    loader = MiniBatchTensorDatasetDict(train_data, config.training.num_batches)

    ## Model
    model = make_model(config.architecture)
    
    ## Callables
    make_callables(config, model)
    
    ## Loss
    loss_fn = make_loss(config, model)
    
    ## Optimizer and Scheduler
    scheduler, optimizer = make_scheduler_optimizer(config, loss_fn)

    if config.optimizer.name in ["kfac", "lbfgs"]:
        last_lr = 0.0  # These optimizers don't use external learning rates
    else:
        last_lr = scheduler(config.training.epochs * config.training.num_batches)
        if type(last_lr) is not float:
            last_lr = last_lr.item()
        
    ## Parameters and opt (fresh or checkpoint)  
    start_epoch, params, opt_state = make_train_state(config, model, optimizer, train_data, rng)
    
    ## Training step
    train_step = make_train(config, optimizer, loss_fn)
    
    ## Log some information    
    logging.info(f'Model: {config.architecture.model_name} with {config.architecture.num_layers} x {config.architecture.hidden_dim} hidden layers and {config.architecture.output_dim} outputs')
    logging.info(f'Number of parameters: {get_param_count(params)}')

    wandb.log({"nof_params": get_param_count(params)})


    ## Start training
    logging.info(f"Starting training with {config.training.num_batches} batch(es)")
    stop_epoch = start_epoch + config.training.epochs
    for epoch in (pbar:=trange(start_epoch, stop_epoch, ncols=80)): # ncols prevents a bug where tqdm breaks lines
        
        ## Batch training
        loss = 0
        if config.training.gradient_conflict is not None:
            alignment = 0
        losses = {k: 0 for k in train_data.keys() if k in ['metric', 'jacobian', 'hessian']}

        for _, data_batched in enumerate(loader):
            rng, subkey = jax.random.split(rng)
            if config.training.gradient_conflict is not None:
                params, opt_state, loss_batch, losses_batch, alignment_batch = train_step(params, opt_state, data_batched, subkey)
                alignment += alignment_batch
            else:
                params, opt_state, loss_batch, losses_batch = train_step(params, opt_state, data_batched, subkey)
            loss += loss_batch
            for k, v in losses_batch.items(): 
                losses[k] += v
        
        ## Mean over batch
        if config.training.integration:
            loss /= jnp.sum(train_data["inv_vol_els"])
        else:
            loss /= loader.num_batches

        for k, v in losses.items():
            if config.training.integration:
                losses[k] /= jnp.sum(train_data["inv_vol_els"])
            else:
                losses[k] /= loader.num_batches
        
        ## Logging
        if config.training.gradient_conflict is not None:
            wandb.log({"epoch": epoch, "loss": loss, "alignment" : alignment / loader.num_batches, **losses})
        else:
            wandb.log({"epoch": epoch, "loss": loss, **losses})
        pbar.set_description(f"{epoch}: \t {loss:.2e}")
        
        ## Validation to wandb
        if 'wandb' in config:
            if (epoch+1) % config.wandb.validate_every_n_epochs == 0 or epoch == start_epoch: ## start_epoch helps debugging
                validation_dict = validate(config, model, params, val_data, data_config, rng)
                wandb.log(validation_dict)

    logging.info(f"Loss final: {loss}, ... {losses}")


    store_checkpoint(config.run_dir, params, opt_state, stop_epoch, last_lr)