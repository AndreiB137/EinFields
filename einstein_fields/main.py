""" MIT License
# 
# Copyright (c) 2025 Andrei Bodnar (Dept of Physics and Astronomy, University of Manchester,United Kingdom), Sandeep S. Cranganore (Ellis Unit, LIT AI Lab, JKU Linz, Austria) and Arturs Berzins (Ellis Unit, LIT AI Lab, JKU Linz, Austria)
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""

import logging
import os, sys
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.75'
from ml_collections import ConfigDict, FrozenConfigDict
import argparse
import orbax.checkpoint as ocp
sys.path.append(os.path.abspath(f"/system/user/crangano/EinFields/"))
from einstein_fields.nn_models import list_activations, get_extra_model_cfg, model_key_dict
from einstein_fields.utils import (
    get_optimizer_args,
    get_scheduler_args,
    optimizer_dict_args,
    schedulers_dict_args,
    get_config,
    store_config,
    load_config
)
from einstein_fields.utils.config_training import (
    valid_file,
    valid_dir
)
from einstein_fields.train import train
import jax
import yaml
import wandb
import time
import absl.logging
from datetime import datetime

# Set ORBAX logging level to ERROR to avoid excessive output
absl.logging.set_verbosity(absl.logging.ERROR)

jax.config.update("jax_default_matmul_precision", "highest")

def parse_args():
    parser = argparse.ArgumentParser(description="Einfields training script", 
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # --- wandb ---

    wandb_group = parser.add_argument_group("Wandb")
    wandb_group.add_argument("--use_wandb", type=bool, default=False,
                        help="If set, uses wandb for logging metrics.")
    wandb_group.add_argument("--wandb_project", type=str, default=None,
                        help="Wandb project.")
    wandb_group.add_argument("--wandb_run", type=str, default=None,
                        help="Wandb run name.")
    wandb_group.add_argument("--wandb_group", type=str, default=None,
                        help="Wandb group name.")
    wandb_group.add_argument("--validate_every_n_epochs", type=int, default=10,
                        help="Number of epochs after which to validate the model.")
    wandb_group.add_argument("--validation_num_batches", type=int, default=10,
                        help="Number of batches to use for validation.")

    # --- Configuration and checkpoint ---

    logg_check_group = parser.add_argument_group("Configuration, Logging, checkpoint")
    logg_check_group.add_argument("--config_file", type=valid_file, default="/system/user/crangano/EinFields/einstein_fields/configs/schwarzschild/config.yml",
                        help="Use a pre-defined yaml config file. If set, the training will start from this config file. Ensure the config has the same structure as the default config file.")
    logg_check_group.add_argument("--log_dir", type=valid_dir, default=None,
                        help="Path to the directory where the logs will be stored.")
    logg_check_group.add_argument("--checkpoint", type=valid_dir, default=None,
                        help="Path to the checkpoint file to load. If set, training will resume from this checkpoint.")
    logg_check_group.add_argument("--reset_optimizer", action='store_true', default=False,
                                  help="If set, the optimizer will reset to its initial state. If not set, the optimizer will be loaded from the checkpoint with maybe a new scheduler.")

    # --- Data ---
    
    data_group = parser.add_argument_group("Data")
    data_group.add_argument("--data_dir", type=valid_dir, default=None,
                        help="Path to the data directory.")
    data_group.add_argument("--jacobian", type=bool, default=False, 
                        help="If set, Jacobian supervision is used.")
    data_group.add_argument("--hessian", type=bool, default=False,
                        help="If set, Hessian supervision is used.")

    # --- Architecture ---

    arch_group = parser.add_argument_group("Architecture")
    arch_group.add_argument("--arch_name", type=str, default=None, choices=list(model_key_dict.keys()),
                        help="Name of the architecture to use.")
    arch_group.add_argument("--hidden_dim", type=int, default=16,
                        help="Hidden dimension of the model.")
    arch_group.add_argument("--output_dim", type=int, default=10,
                        help="Output dimension of the model.")
    arch_group.add_argument("--num_layers", type=int, default=3,
                        help="Number of hidden layers of the model.")
    arch_group.add_argument("--activation", type=str, default=None, choices=list_activations(),
                        help="Activation function to use.")
    
    # --- Optimizer and scheduler ---

    optimizer_group = parser.add_argument_group("Optimizer and scheduler")
    optimizer_group.add_argument("--optimizer", type=str, default=None, choices=list(optimizer_dict_args.keys()),
                        help="Optimizer to use. Note kfac and lbfgs use an adaptive learning rate.")
    optimizer_group.add_argument("--learning_rate", type=float, default=1e-3,
                        help="Learning rate of the optimizer.")
    optimizer_group.add_argument("--scheduler", type=str, default=None, choices=list(schedulers_dict_args.keys()),
                        help="Scheduler to use.")
    
    # --- Training ---

    training_group = parser.add_argument_group("Training")
    training_group.add_argument("--epochs", type=int, default=10,
                        help="Number of epochs to train.")
    training_group.add_argument("--num_batches", type=int, default=80,
                        help="Number of batches to train.")
    training_group.add_argument("--rng_seed", type=int, default=0,
                        help="Random seed for training.")
    training_group.add_argument("--gradient_conflict", type=str, default=None, choices=["grad_norm"],
                        help="Weighting scheme for training. If None, no weighting scheme is used.")
    training_group.add_argument("--norm", type=str, default="mse", choices=["mse", "minkowski", "papuc"],
                        help="Choice of norm.")
    training_group.add_argument("--integration", type=bool, default=False, 
                        help="Use integration with invariant volume elements, namely dV * sqrt(g) for the loss function.")
    training_group.add_argument("--metric_type", type=str, default="full_flatten", choices=["distortion", "full_flatten", "distortion_sym", "full_flatten_sym"],
                                help="Type of metric to use for training. If 'distortion', uses the distortion metric. " \
                                "If 'full_flatten', uses the full flatten metric. " \
                                "If 'distortion_sym', uses the symmetric distortion metric. " \
                                "If 'full_flatten_sym', uses the symmetric full flatten metric.")
    

    return parser.parse_args()

def move_args_to_config(args, config):
    """Move the arguments from the command line to the config file."""
    ### WANDB ###
    
    if args.use_wandb:
        config.wandb.project = args.wandb_project
        config.wandb.name = args.wandb_name
        config.wandb.group = args.wandb_group
        config.wandb.validate_every_n_epochs = 10
        config.wandb.validation_num_batches = 10
    
    ### Logging ###
    config.log_dir = args.log_dir

    ### Data ###
    config.data.data_dir = args.data_dir
    config.data.losses.jacobian = args.jacobian
    config.data.losses.hessian = args.hessian

    ### Architecture ###
    config.architecture.model_name = args.arch_name
    config.architecture.hidden_dim = args.hidden_dim
    config.architecture.output_dim = args.output_dim
    config.architecture.num_layers = args.num_layers
    config.architecture.activation = args.activation
    config.architecture.extra_model_args = get_extra_model_cfg(args.arch_name)

    ### Optimizer and Scheduler ###
    config.optimizer.name = args.optimizer
    config.optimizer.learning_rate = args.learning_rate
    config.optimizer.extra_optimizer_args = get_optimizer_args(args.optimizer)
    config.optimizer.lr_scheduler = args.scheduler
    config.optimizer.extra_scheduler_args = get_scheduler_args(args.scheduler)

    ### Training ###
    config.training.epochs = args.epochs
    config.training.num_batches = args.num_batches
    config.training.rng_seed = args.rng_seed
    config.training.gradient_conflict = args.gradient_conflict
    config.training.norm = args.norm
    config.training.integration = args.integration
    config.training.metric_type = args.metric_type

    
def init_run(config) -> dict:

    ### WANDB ###
    if 'wandb' in config:
        wandb.init(project=config.wandb.project, name=config.wandb.name, group=config.wandb.group)
    else:
        wandb.init(mode="disabled")
    
    ### Logging ###
    # Get run
    run_id = wandb.run.id if wandb.run else datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
    run_dir = os.path.join(config.log_dir, run_id)

    os.makedirs(run_dir, exist_ok=True)
    config.run_dir = run_dir

def validate_config_structure(default_config, user_config, path=None):
    """
    Validate that user_config has the same structure as default_config.
    None values in default_config can accept any type in user_config.
    
    Returns:
        list: List of validation errors as (path, issue_type, details)
    """
    if path is None:
        path = []
    
    errors = []
    
    # Both must be dictionaries at this level
    if isinstance(default_config, ConfigDict) and isinstance(user_config, ConfigDict):
        # Check for missing required keys
        for key in default_config:
            if key not in user_config:
                errors.append((path + [key], 'MISSING_KEY', f"Required key '{key}' not found"))
            else:
                # Recursively validate nested structure
                errors.extend(validate_config_structure(
                    default_config[key], 
                    user_config[key], 
                    path + [key]
                ))
        
        # Check for unexpected extra keys (optional - remove if you want to allow extra keys)
        for key in user_config:
            if key not in default_config:
                errors.append((path + [key], 'EXTRA_KEY', f"Unexpected key '{key}' found"))
    
    elif isinstance(default_config, ConfigDict) and not isinstance(user_config, ConfigDict) and user_config is not None:
        errors.append((path, 'TYPE_MISMATCH', f"Expected ConfigDict for user config, got {type(user_config).__name__}"))
    
    elif not isinstance(default_config, ConfigDict) and isinstance(user_config, ConfigDict):
        errors.append((path, 'TYPE_MISMATCH', f"Expected ConfigDict for default config, got {type(default_config).__name__}"))

    else:
        # Both are non-dict values - check type compatibility
        # None in default accepts any type in user config
        if default_config is not None:
            if isinstance(default_config, int):
                if not isinstance(user_config, int):
                    errors.append((path, 'TYPE_MISMATCH', f"Expected int, got {type(user_config).__name__}"))
            elif type(default_config) != type(user_config) and user_config is not None:
                errors.append((path, 'TYPE_MISMATCH', 
                         f"Expected {type(default_config).__name__}, got {type(user_config).__name__}"))
    
    return errors

def format_validation_errors(errors):
    """Format validation errors for human-readable output"""
    
    if len(errors) == 0:
        return "Configuration structure is valid."
    
    formatted = ["Configuration validation errors:"]
    for path, error_type, details in errors:
        path_str = '.'.join(map(str, path)) if path else 'root'
        formatted.append(f"  • {path_str}: {error_type} - {details}")
    
    return '\n'.join(formatted)


def check_config(other):
    """"Check if the two configs have the same structure and types."""
    if 'wandb' in other:
        tmp = get_config(True)
    else:
        tmp = get_config(False)
    try:
        model_name = other.architecture.model_name
    except AttributeError as e:
        raise AttributeError("The provided config file is missing the 'architecture.model_name' attribute.") from e
    try:
        act_name = other.architecture.activation
    except AttributeError as e:
        raise AttributeError("The provided config file is missing the 'architecture.activation' attribute.") from e
    try:
        optimizer_name = other.optimizer.name
    except AttributeError as e:
        raise AttributeError("The provided config file is missing the 'optimizer.name' attribute.") from e
    try:
        scheduler_name = other.optimizer.lr_scheduler
    except AttributeError as e:
        raise AttributeError("The provided config file is missing the 'optimizer.lr_scheduler' attribute.") from e

    tmp.architecture.model_name = model_name
    tmp.architecture.extra_model_args = get_extra_model_cfg(model_name)
    tmp.architecture.activation = act_name
    tmp.optimizer.name = optimizer_name
    tmp.optimizer.extra_optimizer_args = get_optimizer_args(optimizer_name)
    tmp.optimizer.lr_scheduler = scheduler_name
    tmp.optimizer.extra_scheduler_args = get_scheduler_args(scheduler_name)

    errors = validate_config_structure(tmp, other)

    if len(errors) > 0:
        logging.error("The provided config file does not have the same structure as the default config file.")
        logging.error(format_validation_errors(errors))
        return False
    
    if other.optimizer.name in ["lbfgs", "kfac"] and other.training.gradient_conflict is not None:
        raise NotImplementedError(f"The optimizer {other.optimizer.name} does not support the gradient conflict weighting scheme. Please use a different optimizer.")
    
    if other.optimizer.name in ["lbfgs", "kfac"] and other.training.norm != "mse":
        raise NotImplementedError(f"The optimizer {other.optimizer.name} does not support norms other than 'mse'.")

    if other.optimizer.name == "kfac" and other.training.integration:
        raise NotImplementedError("The 'kfac' optimizer does not support integration with invariant volume elements. Please set 'integration' to False.")

    if not os.path.exists(other.data.data_dir):
        raise ValueError(f"The data directory {other.data.data_dir} does not exist. Please provide a valid data directory.")    

    if other.architecture.output_dim not in [16, 10]:
        raise NotImplementedError(f"The output dimension {other.architecture.output_dim} is not supported. Supported output dimensions are: [16, 10]") 

    if other.training.norm == 'minkowski' and (other.data.losses.jacobian or other.data.losses.hessian):
        raise NotImplementedError("The 'minkowski' norm does not support Jacobian or Hessian supervision. Please set 'jacobian' and 'hessian' to False.")
    
    if other.training.norm == 'papuc' and (other.data.losses.jacobian or other.data.losses.hessian):
        raise NotImplementedError("The 'papuc' norm does not support Jacobian or Hessian supervision. Please set 'jacobian' and 'hessian' to False.")

    if other.training.norm not in ['mse', 'minkowski', 'papuc']:
        raise NotImplementedError(f"The norm '{other.training.norm}' is not supported. Supported norms are: ['mse', 'minkowski', 'papuc']")
    
    contents = os.listdir(other.data.data_dir)

    if 'coords_train.npy' not in contents:
        raise ValueError(f"Data directory {other.data.data_dir} does not contain the required 'coords_train.npy' file. Please ensure the data file has the coordinates.")
    if 'coords_validation.npy' not in contents:
        raise ValueError(f"Data directory {other.data.data_dir} does not contain the required 'coords_validation.npy'. Please ensure the data file has the coordinates")
    if other.training.integration and ('inv_volume_measure_train.npy' not in contents or 'inv_volume_measure_validation.npy' not in contents):
        raise ValueError(f"Data directory {other.data.data_dir} does not contain the required invariant volume measures.")
    
    if other.training.norm == "minkowski":
        if "full_flatten" not in contents:
            raise ValueError(f"Data directory {other.data.data_dir} does not contain the required files for Minkowski norm. Please ensure the data file has the full flatten directory.")
    elif other.training.norm == "papuc":
        if "full_flatten" not in contents:
            raise ValueError(f"Data directory {other.data.data_dir} does not contain the required files for Papuc norm. Please ensure the data file has the full flatten directory.")

    if other.training.metric_type not in ['distortion', 'full_flatten', 'distortion_sym', 'full_flatten_sym', 'distortion_sym']:
        raise ValueError(f"The metric type '{other.training.metric_type}' is not supported. Supported metric types are: ['distortion', 'full_flatten', 'distortion_sym', 'full_flatten_sym']")
    
    if other.training.metric_type.endswith('_sym'):
        type = other.training.metric_type.split('_sym')[0]
        if 'symmetric' not in os.listdir(os.path.join(other.data.data_dir, type)):
            raise ValueError(f"The data directory {other.data.data_dir} does not contain the required symmetric files for the metric type '{other.training.metric_type}'. Please ensure the data file has the symmetric directory for the metric type.")
        
        if other.architecture.output_dim != 10:
            raise ValueError(f"The metric type '{other.training.metric_type}' is not supporting the output dimension {other.architecture.output_dim}. Please use an output of dimension 10.")
    
    if other.training.norm == 'minkowski' or other.training.norm == 'papuc':
        if other.training.metric_type.endswith('_sym'):
            raise ValueError(f"The metric type '{other.training.metric_type}' is not supported for the '{other.training.norm}' norm. Please use a non-symmetric metric type.")

    if other.training.gradient_conflict is not None and other.training.gradient_conflict not in ['grad_norm']:
        raise ValueError(f"The weighting scheme '{other.training.gradient_conflict}' is not supported. Supported schemes are: ['grad_norm']")

    if other.training.gradient_conflict is not None and (other.data.losses.jacobian == False and other.data.losses.hessian == False):
        raise ValueError("The gradient conflict weighting scheme is set, but Jacobian and Hessian supervision are not enabled. Please enable at least one of them.")

    if len(contents) == 0:
        raise ValueError(f"The data directory {other.data.data_dir} is empty. Please provide a valid data directory.")
    
    if len(contents) == 1:
        possible_types = []
        logging.info(f"The data directory {other.data.data_dir} contains only one file.")
        possible_types.append(contents[0])
        contents_sub = os.listdir(os.path.join(other.data.data_dir, contents[0]))
        if 'symmetric' in contents_sub:
            possible_types.append(contents[0] + "_sym")
        if other.training.metric_type not in possible_types:
            logging.info(f"The metric type {other.training.metric_type} is not in the possible types {possible_types} allowed for the file structure. Metric type will default to {possible_types[0]}.")
            other.training.metric_type = possible_types[0]

    if other.training.gradient_conflict is not None and (other.optimizer.name == "lbfgs" or other.optimizer.name == "kfac"):
        raise NotImplementedError(f"The optimizer {other.optimizer.name} does not support the gradient conflict weighting scheme. Please use a different optimizer.")

    problem_dir = os.path.dirname(os.path.dirname(other.data.data_dir))
    if not os.path.exists(os.path.join(problem_dir, "config.yml")):
        raise FileNotFoundError(f"The problem directory {problem_dir} does not contain a config.yml file. Please provide a valid problem directory.")

    return True

def main():

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
        force=True,
        handlers=[
            logging.StreamHandler()
        ]
    )

    args = parse_args()

    if args.checkpoint is not None and args.config_file is not None:
        logging.error("Both --checkpoint and --config_file flags are set. Please provide only one.")
        exit(1)
    logging.info("Starting main.py...")
    logging.info("Checking if a pre-defined config file is provided...")
    final_config = None
    if args.config_file is not None:
        if not os.path.exists(args.config_file):
            logging.error(f"The config file {args.config_file} does not exist. Please provide a valid config file.")
            exit(1)
        logging.info("Using a pre-defined config file.")
        logging.info(f"Using config file: {args.config_file}")
        tmp = {}
        with open(args.config_file, "r") as f:
            tmp = yaml.load(f, Loader=yaml.FullLoader)
        # print(tmp)
        # exit()
        # args.main_config = ConfigDict(tmp)
        logging.info("Checking if the config file has the same structure as the default config file...")
        if check_config(ConfigDict(tmp, sort_keys=False)):
            logging.info("The config file has the same structure as the default config file.")
            final_config = ConfigDict(tmp, sort_keys=False)
        else:
            logging.error("The config file is not allowed.")
            logging.info("Please check the config file and try again.")
            exit(1)
        init_run(final_config)
    elif args.checkpoint is not None:
        if not os.path.exists(args.checkpoint):
            logging.error(f"The checkpoint directory {args.checkpoint} does not exist. Please provide a valid checkpoint directory.")
            exit(1)
        logging.info(f"Using checkpoint: {args.checkpoint}")
        tmp = {}
        with open(os.path.join(args.checkpoint, "config.yml"), "r") as f:
            tmp = yaml.load(f, Loader=yaml.FullLoader)

        options = ocp.CheckpointManagerOptions(
            enable_async_checkpointing=False
        )

        mngr = ocp.CheckpointManager(os.path.join(args.checkpoint, "checkpoint"),
                                    options=options)
        checkpoint_dict = mngr.restore(
            0,
            args=ocp.args.Composite(
                metadata=ocp.args.JsonRestore(),
            )
        )
        # args.main_config = ConfigDict(tmp)
        logging.info("Checking if the config file has the same structure as the default config file...")
        if check_config(ConfigDict(tmp, sort_keys=False)):
            logging.info("The config file has the same structure as the default config file.")
            final_config = ConfigDict(tmp, sort_keys=False)
        else:
            logging.error("The config file is not allowed.")
            logging.info("Please check the config file and try again.")
            exit(1)

        final_config.checkpoint.load_dir = args.checkpoint
        final_config.optimizer.learning_rate = checkpoint_dict.metadata['last_lr']
        
        print(f"Additional info about the last checkpoint run: {checkpoint_dict.metadata}")

        ans = input("Do you wish to modify any configuration parameters? (y/n): ")
        init_run(final_config)
        if ans == "n":
            logging.info("No further changes will be applied. The config is ready for training.")
        elif ans == "y":
            tmp_path = os.path.join(final_config.run_dir, "config_tmp.yml")
            editable_config = {
                "training" : {
                    "epochs": final_config.training.epochs,
                    "num_batches": final_config.training.num_batches,
                    "rng_seed": final_config.training.rng_seed,
                    "gradient_conflict": final_config.training.gradient_conflict,
                    "norm": final_config.training.norm,
                    "integration": final_config.training.integration
                },
                "optimizer": final_config.optimizer.to_dict(),
                "checkpoint": {
                    "reset_optimizer": final_config.checkpoint.reset_optimizer
                },
                "data": {
                    "losses" : final_config.data.losses.to_dict()
                }
            }
            if 'wandb' in final_config:
                editable_config["wandb"] = {
                    "validate_every_n_epochs": final_config.wandb.validate_every_n_epochs,
                    "validation_num_batches": final_config.wandb.validation_num_batches
                }

            logging.info(f"Saving editable config parameters temporarily at {tmp_path}.")
            with open(tmp_path, "w") as f:
                yaml.dump(editable_config, f, sort_keys=False)
            logging.info("Please edit the config file and press Enter only when done.")
            logging.info("Don't forget to save changes.")
            _ = input()

            tmp_modified = {}
            with open(tmp_path, "r") as f:
                tmp_modified = yaml.load(f, Loader=yaml.FullLoader)
            
            tmp_modified = ConfigDict(tmp_modified, sort_keys=False)

            for key, val in tmp_modified["training"].items():
                final_config.training[key] = val
            final_config.optimizer = tmp_modified["optimizer"]
            final_config.checkpoint.reset_optimizer = tmp_modified["checkpoint"]["reset_optimizer"]
            final_config.data.losses = tmp_modified["data"]["losses"]
            if 'wandb' in tmp_modified:
                final_config.wandb.validate_every_n_epochs = tmp_modified["wandb"]["validate_every_n_epochs"]
                final_config.wandb.validation_num_batches = tmp_modified["wandb"]["validation_num_batches"]

            os.remove(tmp_path)

            logging.info("Checking if the final config has the same structure as the default config file...")
            if check_config(final_config):
                logging.info("The config file has the same structure as the default config file.")
            else:
                logging.error("The config file is not allowed.")
                logging.info("Please check the config file and try again.")
                exit(1)
        else:
            logging.info("Invalid input. Exiting.")
            exit(1)
    else:
        logging.info("Using the CLI arguments provided by the user.")
        # logging.info("Applying command line training arguments to main config...")
        logging.info("Checking for wandb...")
        if args.use_wandb:
            logging.info("Using wandb for logging.")
            final_config = get_config(use_wandb=True)
        else:
            logging.info("Not using wandb for logging.")
            final_config = get_config(use_wandb=False)
            logging.info("Entering debugging/testing mode.")
    
        if args.arch_name is None or args.activation is None or args.optimizer is None or args.scheduler is None or args.learning_rate is None:
            logging.error("The --arch_name, --activation, --optimizer, --scheduler and --learning_rate flags are required.")
            exit(1)

        move_args_to_config(args, final_config)

        init_run(final_config)

        logging.info("Finished applying CLI arguments to the config.")
        logging.info("The architecture, optimizer and scheduler might have extra hyperparameters.")
        logging.info("They are provided default values in the config.")

        ans = input("Do you wish to modify these extra arguments? (y/n): ")
        if ans == "n":
            logging.info("No further changes will be applied. The config is ready for training.")
        elif ans == "y":
            tmp_path = os.path.join(final_config.run_dir, "config_tmp.yml")
            logging.info(f"Saving the extra arguments config temporarly at {tmp_path}.")
            extra_args_config = {}
            extra_args_config["extra_model_args"] = final_config.architecture.extra_model_args.to_dict()
            extra_args_config["extra_optimizer_args"] = final_config.optimizer.extra_optimizer_args.to_dict()
            extra_args_config["extra_scheduler_args"] = final_config.optimizer.extra_scheduler_args.to_dict()
            with open(tmp_path, "w") as f:
                yaml.dump(extra_args_config, f, sort_keys=False)
            logging.info("Please edit the config file and press Enter only when done.")
            _ = input()
            tmp = {}
            with open(tmp_path, "r") as f:
                tmp = yaml.load(f, Loader=yaml.FullLoader)
            extra_args_config = ConfigDict(tmp, sort_keys=False)
            final_config.architecture.extra_model_args = extra_args_config.extra_model_args
            final_config.optimizer.extra_optimizer_args = extra_args_config.extra_optimizer_args
            final_config.optimizer.extra_scheduler_args = extra_args_config.extra_scheduler_args
            os.remove(tmp_path)
        else:
            logging.info("Invalid input. Exiting.")
            exit(1)
        
        logging.info("Checking if the final config has the same structure as the default config file...")
        if check_config(final_config):
            logging.info("The config file has the same structure as the default config file.")
        else:
            logging.error("The config file is not allowed.")
            logging.info("Please check the config file and try again.")
            exit(1)

    # Setup basicConfig to log to both console and file

    log_file = os.path.join(final_config.run_dir, "train.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logging.getLogger().addHandler(file_handler)


    ### Config ###

    problem_dir = os.path.dirname(os.path.dirname(final_config.data.data_dir))

    data_config = ConfigDict(load_config(problem_dir), sort_keys=False)

    wandb.config.update(final_config.to_dict())

    logging.info(f"Storing the config at {final_config.run_dir}.")
    store_config(final_config.run_dir, final_config, "config")

    logging.info("Config stored successfully.")

    logging.info("Final config:")

    logging.info(final_config)

    logging.info("Data config:")

    logging.info(data_config)

    logging.info("Freezing the config before training...")

    final_config = FrozenConfigDict(final_config)
    data_config = FrozenConfigDict(data_config)

    logging.info("Config freezed successfully.")

    ### JAX device ###
    devices = jax.local_devices()
    logging.info(f"Visible devices: {devices}")
    logging.info(f"JAX default backend: {jax.default_backend()}")

    t0 = time.time()

    logging.info("Initializing training...")
    train(final_config, data_config)
    logging.info(f"Training finished in {time.time() - t0} s")

    
if __name__ == "__main__":
    main()