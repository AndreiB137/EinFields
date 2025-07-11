from ml_collections import ConfigDict
import yaml
import os
# import wandb

def valid_file(path):
    if not os.path.isfile(path):
        raise FileExistsError(f"'{path}' is not a valid file path")
    return path

def valid_dir(path):
    if not os.path.isdir(path):
        raise FileExistsError(f"'{path}' is not a valid directory")
    return path


def get_config(use_wandb : bool = True):
    config = ConfigDict(sort_keys=False)

    # --- wandb ---
    if use_wandb:
        config.wandb = wandb = ConfigDict(sort_keys=False)
        wandb.project = ""
        wandb.name = ""
        wandb.group = ""
        wandb.validate_every_n_epochs = 10
        wandb.validation_num_batches = 10

    # --- Logging ---

    config.log_dir = ""
    config.run_dir = ""

    # --- Checkpoint ---
    config.checkpoint = checkpoint = ConfigDict(sort_keys=False)
    checkpoint.load_dir = ""
    checkpoint.reset_optimizer = False

    # --- Data settings ---
    config.data = data = ConfigDict(sort_keys=False)

    data.data_dir = ""
    data.losses = ConfigDict(sort_keys=False)
    data.losses.jacobian = False
    data.losses.hessian = False

    # --- Model Architecture ---
    config.architecture = arch = ConfigDict(sort_keys=False)

    arch.model_name = "MLP"
    arch.hidden_dim = 16
    arch.output_dim = 10
    arch.num_layers = 4
    arch.activation = "silu"
    arch.extra_model_args = {}
    

    # --- Optimizer-Scheduler Hyperparameters ---
    config.optimizer = optim = ConfigDict(sort_keys=False)

    optim.name = "adam"
    optim.extra_optimizer_args = {}
    optim.learning_rate = 1e-3
    optim.lr_scheduler = "cosine_decay"
    optim.extra_scheduler_args = {}

    # --- Training Hyperparameters ---
    config.training = train = ConfigDict(sort_keys=False)

    train.epochs = 100
    train.num_batches = 80

    train.gradient_conflict = None
    
    train.rng_seed = 0

    train.norm = ""
    train.integration = False
    train.metric_type = ""

    return config

if __name__ == "__main__":
    config = get_config(use_wandb=True)
    with open("/Users/andrei/Documents/dataa/GRNFs-2/neural_compression/configs/config_train.yml", "w") as f:
        yaml.dump(config.to_dict(), f, sort_keys=False)

    