"""
Contextual utilities for environment and experiment setup.
"""

import gymnasium
from multiworld.core.image_env import ImageEnv


def get_gym_env(env_id, env_class=None, env_kwargs=None, **kwargs):
    """
    Get a gym environment, either by ID or by class.
    
    Args:
        env_id: Environment ID string (if using gymnasium registry)
        env_class: Environment class (if creating directly)
        env_kwargs: Environment kwargs
        **kwargs: Additional kwargs
    
    Returns:
        Environment instance
    """
    if env_id is not None:
        # Create from gym registry
        env = gymnasium.make(env_id)
    elif env_class is not None:
        # Create from class
        if env_kwargs is None:
            env_kwargs = {}
        env = env_class(**env_kwargs)
    else:
        raise ValueError("Must provide either env_id or env_class")
    
    return env
