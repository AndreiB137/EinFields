from .mlp_pinn import MLP_PINN
from .PirateNet import PirateNet
from .mlp import MLP
from .siren import SIREN
from .wire import WIRE
from .activations import get_activation, list_activations

model_key_dict = {
    "MLP": MLP,
    "SIREN": SIREN,
    "WIRE": WIRE,
    "PirateNet": PirateNet,
    "MLP_PINN": MLP_PINN
}

def get_model(model_name : str):
    """
    Get the model class by name.

    Args:
        model_name (str): Name of the model.

    Returns:
        nn.Module: The model class.
    """
    if model_name not in model_key_dict:
        raise ValueError(f"Model `{model_name}` is not supported. Supported models are: {list(model_key_dict.keys())}")
    
    return model_key_dict[model_name]

def create_model_configs():
    """
    Create a dictionary of model configurations.

    Returns:
        dict: A dictionary of model configurations.
    """
    model_configs = {
        "MLP": {},
        "SIREN": {
            "omega_0": 3.
        },
        "WIRE": {
            "first_omega_0": 4.,
            "hidden_omega_0": 4.,
            "scale": 5.,
        },
        "PirateNet": {
            "nonlinearity": 0.0,
            "pi_init": None,
            "reparam": {
                "type": "weight_fact",
                "mean": 1.0,
                "stddev": 0.1,
            },
            "fourier_emb": {
                "embed_scale": 2.,
                "embed_dim": 256,
            },
        },
        "MLP_PINN": {
            "reparam": {
                "type": "weight_fact",
                "mean": 1.0,
                "stddev": 0.1,
            },
            "fourier_emb": {
                "embed_scale": 2.,
                "embed_dim": 256,
            },
        },

    }
    return model_configs

model_configs = create_model_configs()

def get_extra_model_cfg(model_name: str):
    """
    Get the extra model configuration for a given model name.

    Args:
        model_name (str): Name of the model.

    Returns:
        dict: The extra model configuration.
    """
    if model_name not in model_configs:
        raise ValueError(f"Model `{model_name}` is not supported. Available models are: {list(model_configs.keys())}")

    return model_configs[model_name]