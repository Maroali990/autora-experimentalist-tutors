"""
Example Experimentalist
"""

import numpy as np
import pandas as pd

candidate_points = None


def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1-point2) ** 2))


def max_min_distance_selection(candidate_points, selected_points):
    max_min_distance = -1
    next_point = None

    for candidate in candidate_points:
        min_distance = np.min([euclidean_distance(candidate, selected) for selected in selected_points])
        if min_distance > max_min_distance:
            max_min_distance = min_distance
            next_point = candidate

    return next_point


def resample_condition_space(condition_space, values_per_dim):
    resampled_space = []
    for col in condition_space.columns:
        resampled_space.append(np.linspace(condition_space[col].min(), condition_space[col].max(), values_per_dim))
    return np.array(np.meshgrid(*resampled_space)).T.reshape(-1, len(condition_space.columns))


def max_distance_sampler(data, condition_space, n_samples, values_per_dim=10):
    """
    Picks the next set of experimental conditions by maximizing the
    distance to all previously recorded conditions.

    Args:
        data (DataFrame):             Experimental Data gathered so far
        condition_space (DataFrame):  n-dimensional grid of experimental conditions
                      where n corresponds to the number of independent variables in an experiment.
                      Columns are named after independent variables. Can be acquired via
                      calling autora experimentalist grid_pool on independent variables
        n_samples (int):              Number of samples to return per step
        values_per_dim (int):         Resamples the condition_space
                    such that each dimension is discretized to the specified number of values.
    Returns:
        conditions (DataFrame):       DataFrame featuring next proposed experimental
                                      conditions
    Example:
    >> data = pd.DataFrame({
    >> 'var1': [1, 3, 5],
    >> 'var2': [2, 4, 6],
    >> })
    >> condition_space = pd.DataFrame({
         'var1': np.linspace(0, 10, 100),
         'var2': np.linspace(0, 10, 100)
         })
    >> n_samples = 5
    >> values_per_dim = 10
    >> new_conditions = max_distance_sampler(data, condition_space, n_samples, values_per_dim)
    >> print(new_conditions)
    """
    # Class level variable, since autora_experimentalists do not seem to be instance based
    global candidate_points

    # Define previous conditions based on the recorded values of independent
    #  variables in data
    previous_conditions = data[condition_space.columns].values

    # Resample the condition space to create a dense grid of candidate points.
    if candidate_points is None:
        candidate_points = resample_condition_space(condition_space, values_per_dim)

    new_conditions = []
    for _ in range(n_samples):
        # Find new condition based on distance to previous conditions
        next_condition = max_min_distance_selection(candidate_points, previous_conditions)
        new_conditions.append(next_condition)

        # Insert new condition into previous_conditions
        previous_conditions = np.vstack([previous_conditions, next_condition])

        # Remove the selected condition from candidate points
        candidate_points = np.array([point for point in candidate_points if not np.array_equal(point, next_condition)])

    return pd.DataFrame(new_conditions, columns=condition_space.columns)

