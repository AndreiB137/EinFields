from soap_jax import soap
import optax
import kfac_jax

schedulers_dict_args = {
    "constant": {},
    "cosine_decay": {
        "decay_steps": 100,
        "alpha": 1e-3,
        "exponent" : 1.0,
    },
    "exponential_decay": {
        "transition_steps": 100,
        "decay_steps": 300,
        "decay_rate": 0.96,
    },
    "warmup_cosine_decay": {
        "peak_value": 1e-2,
        "warmup_steps": 100,
        "decay_steps": 100,
        "end_value": 0.,
        "exponent" : 1.0,
    },
    "warmup_exponential_decay": {
        "peak_value": 1e-2,
        "warmup_steps": 100,
        "transition_steps": 100,
        "decay_rate": 0.96,
        "transition_begin": 100,
    },
}

scheduler_dict = {
    "constant": optax.schedules.constant_schedule,
    "cosine_decay": optax.schedules.cosine_decay_schedule,
    "exponential_decay": optax.schedules.exponential_decay,
    "warmup_cosine_decay": optax.schedules.warmup_cosine_decay_schedule,
    "warmup_exponential_decay": optax.schedules.warmup_exponential_decay_schedule,
}

optimizer_dict_args = {
    "adam": {
        "b1": 0.9,
        "b2": 0.999,
        "eps": 1e-8,
    },
    "adamw": {
        "b1": 0.9,
        "b2": 0.999,
        "eps": 1e-8,
        "weight_decay": 1e-3,
    },
    "lars": {
        "momentum": 0.9,
        "eps": 0.,
        "weight_decay": 1e-3,
        "trust_coefficient": 0.001,
        "nesterov": False,
    },
    "lamb": {
        "b1": 0.9,
        "b2": 0.999,
        "eps": 1e-6,
        "eps_root": 0.,
        "weight_decay": 1e-3,
    },
    "soap": {
        "b1": 0.95,
        "b2": 0.95,
        "eps": 1e-8,
        "precondition_frequency": 1,
        "weight_decay": 0.,
    },
    "kfac": {
        "l2_reg": 0.,
        "initial_damping": 1.0,
        "multi_device": False,
        "num_burnin_steps": 0,
    },
    "lbfgs": {
        "memory_size": 10
    },
}

optimizer_dict = {
    "adam": optax.adam,
    "adamw": optax.adamw,
    "lars": optax.lars,
    "lamb": optax.lamb,
    "soap": soap,
    "kfac": kfac_jax.Optimizer,
    "lbfgs": optax.lbfgs,
}

def get_optimizer_args(name : str):
    """
    Get the optimizer argument dictionary by name.

    Args:
        name (str): Name of the optimizer.

    Returns:
        dict: The optimizer configuration.
    """
    if name not in optimizer_dict_args:
        raise ValueError(f"Optimizer `{name}` is not supported. Available optimizers are : {list(optimizer_dict_args.keys())}")

    return optimizer_dict_args[name]

def get_optimizer(name : str):
    """
    Get the optimizer class by name.

    Args:
        name (str): Name of the optimizer.

    Returns:
        optax: The optimizer class.
    """
    if name not in optimizer_dict_args:
        raise ValueError(f"Optimizer `{name}` is not supported. Available optimizers are : {list(optimizer_dict_args.keys())}")

    return optimizer_dict[name]

def get_scheduler(name : str):
    """
    Get the learning rate scheduler function by name.

    Args:
        name (str): Name of the scheduler.

    Returns:
        Callable: The learning rate scheduler.
    """
    if name not in scheduler_dict:
        raise ValueError(f"Scheduler `{name}` is not supported. Available schedulers are : {list(scheduler_dict.keys())}")

    return scheduler_dict[name]

def get_scheduler_args(name : str):
    """
    Get the learning rate scheduler argument dictionary by name.

    Args:
        name (str): Name of the scheduler.

    Returns:
        dict: The scheduler configuration.
    """
    if name not in schedulers_dict_args:
        raise ValueError(f"Scheduler `{name}` is not supported. Available schedulers are : {list(schedulers_dict_args.keys())}")

    return schedulers_dict_args[name]