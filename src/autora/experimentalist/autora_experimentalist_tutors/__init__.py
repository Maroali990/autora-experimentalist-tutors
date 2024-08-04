"""
Example Experimentalist
"""
import numpy as np
import pandas as pd

from typing import Union, List

from autora.experimentalist.model_disagreement import model_disagreement_sample
from autora.experimentalist.random import random_sample
from autora.experimentalist.mixture import mixture_sample


def sample(
        conditions: Union[pd.DataFrame, np.ndarray],
        models: List,
        reference_conditions: Union[pd.DataFrame, np.ndarray],
        num_samples: int = 1,
        num_cycles: int = 20) -> pd.DataFrame:
    """
    Add a description of the sampler here.

    Args:
        conditions: The pool to sample from.
            Attention: `conditions` is a field of the standard state
        models: The sampler might use output from the theorist.
            Attention: `models` is a field of the standard state
        reference_conditions: The sampler might use reference conditons
        num_samples: number of experimental conditions to select

    Returns:
        Sampled pool of experimental conditions

    *Optional*
    Examples:
        These examples add documentation and also work as tests
        >>> example_sampler([1, 2, 3, 4])
        1
        >>> example_sampler(range(3, 10))
        3

    """
    if num_samples is None:
        num_samples = conditions.shape[0]


       
    # roll = np.random.rand(1)
    # if roll > 0.7:
    #     new_conditions = model_disagreement_sample(conditions, models = models, num_samples=num_samples )
    #     print('random')
    # else:
    #     new_conditions = random_sample(conditions, models = models, num_samples=num_samples )
    #     print('model disagreement')

    #random_state_1 = np.random.randint(1, 10000)
    #random_state_2 = np.random.randint(1, 10000)
    conditions_random = random_sample(conditions, models = models, num_samples=num_samples)
    conditions_model_disagreement = model_disagreement_sample(conditions, models=models, num_samples=num_samples)
    rand_value = np.random.rand()
    if rand_value < 0.5:
        conditions = conditions_random
    else:
        conditions = conditions_model_disagreement
            


    

    new_conditions = conditions

    return new_conditions[:num_samples]

def mix_sample(
        conditions: Union[pd.DataFrame, np.ndarray],
        models: List,
        reference_conditions: Union[pd.DataFrame, np.ndarray],
        num_samples: int = 1,
        num_cycles: int = 20) -> pd.DataFrame:
    """
    Add a description of the sampler here.

    Args:
        conditions: The pool to sample from.
            Attention: `conditions` is a field of the standard state
        models: The sampler might use output from the theorist.
            Attention: `models` is a field of the standard state
        reference_conditions: The sampler might use reference conditons
        num_samples: number of experimental conditions to select

    Returns:
        Sampled pool of experimental conditions

    *Optional*
    Examples:
        These examples add documentation and also work as tests
        >>> example_sampler([1, 2, 3, 4])
        1
        >>> example_sampler(range(3, 10))
        3

    """
    if num_samples is None:
        num_samples = conditions.shape[0]

    params = {
    "random": {"conditions": conditions, "models": models, "num_samples":num_samples},
    "model disagreement": {"conditions": conditions, "models": models, "num_samples":num_samples},
    "sample": {"conditions": conditions, "models": models, "num_samples": num_samples}
    }
            
    conditions = mixture_sample(conditions, 
                                temperature=0.7, 
                                samplers=[[random_sample, "random", [0.3]], [model_disagreement_sample, "model disagreement", [0.7]], [sample, "sample", [0.5]]],
                                params = params,
                                num_samples = num_samples
                                          )


    

    new_conditions = conditions

    return new_conditions[:num_samples]