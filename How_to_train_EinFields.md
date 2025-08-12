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

Note: 
- There is no need to specify `--run_dir` or even in the .yml config file because the script will automatically create it based on the current date and time.

- Validation during training is only possible if `wandb` is in the config file.

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
#### Experimental
- `minkowski`: Minkowski (not actually a norm, convergence not guaranteed)
- `papuc`: Papuc norm (not actually a norm, only in very specific settings and convergence is not guaranteed)

### Metric Types (`--metric_type`)
- `full_flatten`: Standard flattened quantities (metric of shape (16,), Jacobian of shape (64,), Hessian of shape (256,))
- `distortion`: Distortion-based flattened quantities for the full metric (metric of shape (16,), Jacobian of shape (64,), Hessian of shape (256,))
- `full_flatten_sym`: Symmetric part flattened quantities (metric of shape (10,), Jacobian of shape (40,), Hessian of shape (100,))
- `distortion_sym`: Symmetric part distortion quantities for the full metric (metric of shape (10,), Jacobian of shape (40,), Hessian of shape (100,))

Note: If metric is not of type `_sym`, then choosing `output_dim=10` for architecture is allowed, but before training starts it will reshape and reconstruct the (4,4) metric. On the other hand, for `_sym` type it will not reshape or reconstruct, and it will use the symmetric shapes.

## Data Generation


### Training requirements

#### Required Files
- `coords_train.npy`: Training coordinates
- `coords_validation.npy`: Validation coordinates

#### Conditional Files
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

The `no_scale` name is to distinguish from `scale` transformations (streching) which can be applied to the metric in its `coordinate system` and not in `other_coordinate_systems`. These will create file names `scale1`, `scale2` etc.

Check for `--data_dir` in training config to be `./folder_name/problem_name/coordinate_system` + `/no_scale` or `/scale1` etc. depending on your usecase.

#### Parent Directory Requirements
The parent directory of your data directory must contain:
- `config.yml`: Problem-specific configuration file

### Config parameters and metric type explained

#### Basic config

```yaml
"metric": "Kerr",
"metric_args": {
    "M": 1.0,
    "a": 0.7,
},
"coordinate_system":"kerr_schild_cartesian",
"other_coordinate_systems": other_coordinate_systems, 
"grid_shape": [1, 128, 128, 128],
"grid_range": [
    [0.0, 0.0],
    [-3., 3.],
    [-3., 3.],
    [0.1, 3.]], # Avoiding singularity at z=0
"endpoint": [True, True, True, True],
"store_quantities" : {
    "store_symmetric": True,
    "store_distortion": True,
    "store_GR_tensors": False,
},
"compute_volume_element": False,
"recompute_volume_elements": False, # Not implemented yet
"problem": "Kerr",
"data_dir": "/your_path/data_dir"
```

#### Details
- `metric` is checked if its present in `/data_lookup_tables.py`. So far only Schwarzschild, Kerr and GWs are available.
- `metric_args` should always contain `M` and maybe `a` if working with a rotating black hole. Remember that `G`=`c`=1. For gravitational waves, other metric_args are needed, see below. Its not mandatory to have only these arguments, the code will only use the needed ones. For example, if `M` or `a` is introduced in the dictionary it will have no effect and give no errors.

```yaml
"metric_args": {
    "polarization_amplitudes": (1e-6, 1e-6),
    "omega": 1.0,
}
```


- `coordinate_system` must be exactly named as in the available coordinate system for each metric which can be found in the `/data_lookup_tables.py`.
- `endpoint` is to control whether to include or not the right end of the the interval for each coordinate axis.
- if `store_symmetric=True`, it will store the symmetric content for both metrics full and `distortion` (if `store_distortion=True`).
- The GR tensors (if `store_GR_tensors=True`) will only be stored in `full_flatten`.
- `compute_volume_element` is if you want to replace mean squared error loss which gives equal weighting 1/n to a different weighting which depends on an invariant volume element dVsqrt(g), where dV are some numerically calculated volume elements based on the axes gradients. If one of the axes is sampled once, this will automatically fall to the subspace with at least 2 elements on the axis (computing a 2D, 3D volume weighting). This is to mimic the invariant integration under coordinate changes in general relativity. In our available analytical metrics this was not useful because when trained on distortion metrics these all have determinant zero.

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

### Jax compatibility with latest version

- Highly recommended to pip install -U the latest versions of all Jax dependent software in case of using Jax 0.6 or 0.7. Many things are getting depricated or removed from the API.

- In Jax 0.7 (and maybe a bit earlier) `kfac-jax` (or `tensorflow_probability`, which is used by it) gives AttributeError for `jax.interpreters.xla.pytype_aval_mappings`. Remove these lines in the code:
```python
  jax.interpreters.xla.pytype_aval_mappings[NumpyVariable] = (
      jax.interpreters.xla.pytype_aval_mappings[onp.ndarray])
```

For additional help, check the validation error messages which provide specific guidance for fixing configuration issues.