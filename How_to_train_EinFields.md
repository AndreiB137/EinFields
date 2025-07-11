# EinFields Training Guide

## Overview

A training guideline for Einstein fields. It supports three distinct configuration modes to accommodate different use cases.

## Quick Start

### Method 1: Command Line Interface (tedious)

```bash
python main.py \
    --arch_name MLP \
    --activation silu \
    --optimizer adam \
    --scheduler constant \
    --learning_rate 0.001 \
    --data_dir /path/to/your/data \
    --epochs 100 \
    --use_wandb true \
    --wandb_project "your_project"
```

**Required CLI parameters**: `--arch_name`, `--activation`, `--optimizer`, `--scheduler`, `--data_dir`,
`--log_dir`

### Method 2: Config File (recommended)

```bash
python main.py --config_file path/to/your_config.yml
```

### Method 3: Resume from Checkpoint

```bash
python main.py --checkpoint /path/to/checkpoint_directory
```

## Configuration Modes Explained

### 1. CLI Mode (Interactive argparser configuration)

**Best for**: Learning the available hyperparameteres

**How it works**:
1. Specify core parameters via command line flags
2. Script creates a default configuration with your specified values
3. **Interactive prompt**: You can edit advanced parameters (extra model/optimizer/scheduler args)
4. Configuration is validated before training begins

**Required flags**:
- `--arch_name`: Model architecture (choices: see in `--help`)
- `--activation`: Activation function (choices: see in `--help`)
- `--optimizer`: Optimizer type (choices: see in `--help`)
- `--scheduler`: Learning rate scheduler (choices: see in `--help`)
- `--data_dir`: Path to your data directory
- `--log_dir`: Path to your log directory where logging info is saved

**Example with additional options**:
```bash
python main.py \
    --arch_name MLP \
    --activation silu \
    --optimizer adamw \
    --scheduler cosine_decay \
    --learning_rate 0.01 \
    --data_dir ./data/kerr_schild \
    --hidden_dim 64 \
    --num_layers 4 \
    --epochs 200 \
    --jacobian true \
    --hessian false \
    --norm mse \
    --metric_type full_flatten
```

### 2. Config File Mode (Complete Control)

**Best for**: Reproducible research, complex configurations, flexible modifications

**How it works**:
1. Create a YAML config file with complete configuration
2. Script validates structure against default config
3. Training starts immediately (no interactive prompts)

**Config file structure** (must match exactly):
```yaml
# Example: config.yml
wandb:  # Optional section (if you don't want wandb just remove this part from the YAML file)
  project: "EinFields_Training"
  name: "Testing Kerr-Schild"
  group: "Kerr"
  validate_every_n_epochs: 10
  validation_num_batches: 10

log_dir: "./logs"

data:
  data_dir: "./data/kerr_schild"
  losses:
    jacobian: true
    hessian: false

architecture:
  model_name: MLP # See /configs/config_arch.yml for extra_model_args
  hidden_dim: 64
  output_dim: 10
  num_layers: 4 # It always means the number of hidden layers (excluding input and output layers)
  activation: silu
  extra_model_args: {}  # Architecture-specific parameters

optimizer:
  name: "soap" # See /configs/config_optimizer.yml for extra_optimizer_args
  learning_rate: 0.01
  lr_scheduler: "cosine_decay"
  extra_optimizer_args:
    b1: 0.95
    b2: 0.95
    eps: 1.0e-08
    precondition_frequency: 1
    weight_decay: 0.0
  extra_scheduler_args: # See /configs/config_schedule.yml for extra_schedule_args
    decay_steps: 20000
    alpha: 1.e-5
    exponent : 1.0

training:
  epochs: 200
  num_batches: 100
  rng_seed: 0
  gradient_conflict: null  # or "grad_norm"
  norm: "mse"  # or "minkowski", "papuc"
  integration: false
  metric_type: "full_flatten"  # or "full_flatten_sym", "distortion", "distortion_sym", 
```

### 3. Checkpoint Mode (Resume Training)

**Best for**: Continuing interrupted training, fine-tuning, parameter adjustments

**How it works**:
1. Loads configuration and model state from checkpoint directory
2. **Interactive prompt**: Allows modification of training parameters
3. Can reset optimizer state or continue with saved state
4. Resumes training from the last epoch

**What you can modify when resuming**:
- Training parameters (epochs, learning rate, etc.)
- Optimizer settings (learning rate, scheduler, etc.)
- Data loss settings (jacobian, hessian supervision)
- Wandb validation settings
- Optimizer reset option

## Interactive Configuration Details

### CLI Mode Interactive Session
When using CLI mode, you'll be prompted:
```
Do you wish to modify these extra arguments? (y/n):
```

If you choose "y":
1. A temporary YAML file is created in your run directory
2. Edit the file with your preferred text editor
3. Save changes and press Enter to continue
4. The temporary file is automatically deleted

### Checkpoint Mode Interactive Session
When resuming from checkpoint:
```
Do you wish to modify any configuration parameters? (y/n):
```

If you choose "y", you can edit:
- Training epochs and batch settings
- Optimizer parameters and learning rate
- Data supervision settings
- Wandb validation frequency

## Available Options

### Architectures (`--arch_name`)
- `MLP`: Multi-layer perceptron
- Additional architectures available in `models/` directory

### Activations (`--activation`)
- `silu`, `tanh`, `sigmoid`, `gelu`, `telu`
- See `models/activations.py` for complete list

### Optimizers (`--optimizer`)
- `adam`, `adamw`: Standard adaptive optimizers
- `soap`: Second-order optimizer with Shampoo and Adam 
- `lbfgs`: Limited-memory BFGS (adaptive learning rate)
- `kfac`: K-FAC second-order optimizer (adaptive learning rate)

### Schedulers (`--scheduler`)
- `constant`: Fixed learning rate
- `cosine_decay`: Cosine annealing
- `exponential_decay`: Exponential decay
- Additional schedulers in `/configs/config_schedule.yml`

### Loss Norms (`--norm`)
- `mse`: Mean squared error (default)
- `minkowski`: Minkowski (not actually a norm, convergence not guaranteed)
- `papuc`: Papuc norm (not actually a norm, only in very specific settings and convergence is not guaranteed)

### Metric Types (`--metric_type`)
- `full_flatten`: Standard flattened quantities
- `distortion`: Distortion-based flattened quantities for the full metric
- `full_flatten_sym`: Symmetric part flattened quantities
- `distortion_sym`: Symmetric part distortion quantities for the full metric

## Data Directory Requirements


### Required Files
- `coords_train.npy`: Training coordinates
- `coords_validation.npy`: Validation coordinates

### Conditional Files
- **Integration mode** (`--integration true`):
  - `inv_volume_measure_train.npy`
  - `inv_volume_measure_validation.npy`

- **Minkowski/Papuc norms**:
  - `full_flatten/` directory with metric data

- **Symmetric metrics** (`*_sym` metric types):
  - `{metric_type}/symmetric/` directory structure
  - Requires `--output_dim 10`

### Directory Structure Example
```
folder_name/problem_name/
│ ├── coordinate_system
│   ├── no_scale/
    │   ├── coords_train.npy
    │   ├── coords_validation.npy
    │   ├── inv_volume_measure_train.npy            # If compute_volume_element = True
    │   ├── inv_volume_measure_validation.npy       # If compute_volume_element = True
    │   ├── full_flatten/
    │       ├── symmetric/                          # If store_symmetric = True
    │           ├── training
    │           ├── hessian.npy
    │           ├── jacobian.npy
    │           ├── metric.npy
    │       ├── validation
    │           ├── hessian.npy
    │           ├── jacobian.npy
    │           ├── metric.npy
    │       ├── training
    │           ├── hessian.npy
    │           ├── jacobian.npy
    │           ├── metric.npy
    │           ├── riemann_tensor.npy               # If store_GR_tensors = True
    │           ├── kretschmann.npy                  # If store_GR_tensors = True
    │       └── validation
    │           ├── hessian.npy
    │           ├── jacobian.npy
    │           ├── metric.npy
    │           ├── riemann_tensor.npy               # If store_GR_tensors = True
    │           ├── kretschmann.npy                  # If store_GR_tensors = True
    │   ├── distortion/                              # If store_distortion = True
    │       ├── same as for full_flatten
    ├── scale1/...
    └── scale2/...
│ ├── other_coordinate_systems                       # If other_coordinate_systems list is not empty
| └── config.yml
```

The same structure follows for `other_coordinate_systems` if provided. `other_coordinate_systems` is for the same metric but represented in different coordinates and evaluated at the same collocation points expressed in these coordinates.

The GR tensors will only be stored in `full_flatten`. The `no_scale` name is to distinguish from `scale` transformations (streching) which can be applied to the metric in its `coordinate system` and not in `other_coordinate_systems`. These will create file names `scale1`, `scale2` etc.

Check for `--data_dir` in config to be ./folder_name/problem_name/coordinate_system. Then finish with /no_scale or /scale1 depending on your usecase.

### Parent Directory Requirements
The parent directory of your data directory must contain:
- `config.yml`: Problem-specific configuration file

## Optimizer-Specific Limitations

### LBFGS and K-FAC Optimizers
These optimizers have restrictions:
- **No gradient conflict weighting**: Cannot use `--gradient_conflict`
- **MSE norm only**: Must use `--norm mse`
- **No integration**: K-FAC cannot use `--integration true`
- **Adaptive learning rate**: Specified learning rate may be adjusted automatically

### Norm-Specific Limitations

#### Minkowski and Papuc Norms
- **No supervision**: Cannot use `--jacobian true` or `--hessian true` (not implemented)
- **No symmetric metrics**: Cannot use `*_sym` metric types
- **Requires full_flatten data**: Data directory must contain `full_flatten/` subdirectory

## Validation and Error Handling

The script performs extensive validation:

1. **Structure validation**: Config files must match default structure exactly
2. **Parameter compatibility**: Checks optimizer/norm/metric combinations
3. **Data validation**: Verifies all required data files exist
4. **Path validation**: Ensures directories and files are accessible


## Output and Logging

### Directory Structure
Training creates the following structure:
```
log_dir/
└── {run_id}/
    ├── config.yml         # Final configuration used
    ├── train.log          # All loggings messages
    ├── checkpoint/        # Model checkpoint
    └── config_tmp.yml     # Temporary (for interactive config changes, deleted after use)
```

### Wandb Integration
When `--use_wandb true`:
- Metrics logged to Weights & Biases
- Run ID used as local directory name
- Config automatically uploaded to wandb 

### Checkpoint Management
- Resume capability with `--checkpoint`
- Option to reset optimizer state with `--reset_optimizer`

## Troubleshooting

### Memory Issues
If you encounter GPU memory errors:
- Reduce `--hidden_dim` or `--num_layers`
- Increase `--num_batches` or `--validation_num_batches` (if `wandb` is used)
- Adjust `XLA_PYTHON_CLIENT_MEM_FRACTION` environment variable (default: 0.75)

For additional help, check the validation error messages which provide specific guidance for fixing configuration issues.