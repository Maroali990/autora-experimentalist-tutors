"""
This is the implementation of a max min-distance sampler.
It chooses the next sample based on the maximal minimal difference to a comparison set of points.
If outperforms random sampling for low sampling steps on synthetic experimental data.
It is a custom implementation of the "novelty" approach from autorRA's novelty experimentalist.
Unlike autoRA's novelty experimentalist, this one only offers the calculation of the Euclidean distance and always
utilizes the maximal _min_ distance to choose the next point. Once a suitable condition is chosen, it is possible
to add a small stochastic element to it by sampling uniformly from a n-dimensional blob surrounding that point.
Blob size is chosen such that there is no overlap for candidate point and the condition space is optimally covered.
Furthermore, dynamic resampling of the condition space to improve performance is provided."""

import numpy as np
import pandas as pd

"""
candidate_points are all points in the conditions space that could potentially be proposed.
It is a result of resampling the entirety of condition space. Must be reinitialized before each experiment.
"""
# don't like using globals but the challenge set up requires it.
candidate_points = None
"""The size of the uniform random blobs surrounding selected candidate points. (Suggested conditions)"""
blob_scale = np.array([0])


def uniform_blob(center, num_samples=1):
    """
    Generates samples from an n-dimensional uniform probability distribution centered at 'center'.

    Args:
        center (np.ndarray): The center of the uniform blob. A point in n-dimensional space
        num_samples (int):   Number of samples to generate.

    Returns:
        np.ndarray: Generated samples.
    """
    global blob_scale

    n_dims = len(center)
    # Compute blob low and high points and clip them to the boundaries of candidate_points
    dim = len(center)
    low = np.maximum(center-blob_scale, np.zeros(dim))
    high = np.minimum(center+blob_scale, np.ones(dim) * range_scale)

    return np.random.uniform(low=low, high=high, size=(num_samples, dim))


def compute_optimal_blob_size(space):
    """

    Args:
        space (Union[ndarray,DataFrame]) : A condition space. For this use cases expected to be candidate_points

    Returns:
        The optimal size for blobs centered around candidate_points to cover the entire condition space
        without overlap
    """

    # Compute a range such that uniform blobs of candidate points do not overlap but all together
    # cover the entire condition_space
    range_scale = []
    n_dims = len(candidate_points.shape)

    for col in space.columns:
        col_range = candidate_points[col].max()-candidate_points[col].min()
        range_scale.append(col_range / n_dims)

    return np.array(range_scale)


def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1-point2) ** 2))


def max_min_distance_selection(candidate_points, sampled_points):
    """
    Computes all distances between candidate_points and sampled_points.
    The minimal distance for a candidate_point is the shortest distance
    to any of the sampled_points. This algorithm selects the candidate_point
    which has the greatest minimal distance.

    candidate_points (DataFrame): Points in space that could potentially be chosen
                In autora context that would grid_pooled independent variable values
    sampled_points (DataFrame):  Points in space which have already been sampled
    """
    max_min_distance = -1
    next_point = None

    for candidate in candidate_points:
        min_distance = np.min([euclidean_distance(candidate, sampled) for sampled in sampled_points])
        if min_distance > max_min_distance:
            max_min_distance = min_distance
            next_point = candidate

    return next_point


def resample_condition_space(condition_space, values_per_dim):
    """
    Resamples the condition space to create a dense grid of candidate points. Can be used to increase or decrease
    the search space. For high dimensional spaces low number of values per dimension are recommended.

    Args:
        condition_space (DataFrame): A DataFrame representing the n-dimensional grid of experimental conditions,
                                     where each column corresponds to an independent variable.
        values_per_dim (int):        The number of discrete values each independent variable can assume.

    Returns:
        np.ndarray: A 2D array where each row represents a point in the resampled condition space,
                    and each column corresponds to an independent variable.
    """
    resampled_space = []
    for col in condition_space.columns:
        resampled_space.append(np.linspace(condition_space[col].min(), condition_space[col].max(), values_per_dim))
    return np.array(np.meshgrid(*resampled_space)).T.reshape(-1, len(condition_space.columns))


def check_resampling_required(condition_space, candidate_points, values_per_dim):
    """
    Determines whether the condition_space must be resampled again. E.g. because the experiment or other
    condition_space related parameters have changed.

    Args:
        condition_space (DataFrame):  * specified in `max_min_distance_sampler`
        candidate_points (DataFrame): The current resampled variant of the condition space.
        values_per_dim (DataFrame):   * specified in `max_min_distance_sampler`
    Returns:
        bool True if resampling is required
    """
    # 1. If there are no candidate_points
    if candidate_points is None:
        return True
    # 2. If candidate_points.shape[0] != values_per_dim**n where n is the number of columns in condition_space
    if candidate_points.shape[0] != values_per_dim ** len(condition_space.columns):
        return True
    # 3. If independent variable names have changed
    if not (candidate_points.columns == condition_space.columns):
        return True
    # 4. If the boundaries of the condition_space have changed
    for col in condition_space.columns:
        if condition_space[col].min() != candidate_points[col].min() or \
           condition_space[col].max() != candidate_points[col].max():
            return True

    # No resampling required
    return False


def max_min_distance_sampler(data,
                             condition_space,
                             n_samples,
                             values_per_dim=10,
                             stochastic=False):
    """
    Picks the next set of experimental conditions by maximizing the
    distance to all previously recorded conditions. Default distance measure
    is Euclidean.

    Args:
        data (DataFrame):             Experimental Data gathered so far
        condition_space (DataFrame):  n-dimensional grid of experimental conditions
                      where n corresponds to the number of independent variables in an experiment.
                      Columns are named after independent variables. Can be acquired via
                      calling autora experimentalist grid_pool on independent variables
        n_samples (int):              Number of samples to return per step
        values_per_dim (int):         The number of discrete values each independent variable can assume
                                      after resampling.
        stochastic (bool):            If true, on top of max min-distance selection a small stochastic
                                      displacement is added.

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
    # TODO: Discuss use case and change in the future.
    global candidate_points
    global blob_scale

    # Decide whether candidate_points must be resampled
    if check_resampling_required(condition_space, candidate_points, values_per_dim):
        # Resample the condition space to create a dense grid of candidate points.
        candidate_points = resample_condition_space(condition_space, values_per_dim)
        compute_optimal_blob_size(candidate_points)

    # Identify which conditions have already been sampled before.
    previous_conditions = data[condition_space.columns].values

    new_conditions = []
    for _ in range(n_samples):
        # Find new condition based on max_min_distance to previous conditions
        next_condition = max_min_distance_selection(candidate_points, previous_conditions)

        if stochastic:
            next_condition = uniform_blob(next_condition)

        new_conditions.append(next_condition)

        # Insert new condition into previous_conditions
        previous_conditions = np.vstack([previous_conditions, next_condition])

        # Remove the selected condition from candidate points
        candidate_points = np.array([point for point in candidate_points if not np.array_equal(point, next_condition)])

    return pd.DataFrame(new_conditions, columns=condition_space.columns)

