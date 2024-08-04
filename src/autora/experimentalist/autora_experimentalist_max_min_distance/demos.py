
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from . import max_min_distance_sampler


def sampling_example_2d():
    ## Plot sampling order of new conditions

    # Define the ranges for 2 independent variables
    condition_space = pd.DataFrame({
        'iv1': np.linspace(0, 10, 100),
        'iv2': np.linspace(0, 10, 100),
    })

    # Pretend some data has been sampled already
    data = pd.DataFrame({
        'iv1': [1, 3, 5],
        'iv2': [2, 4, 6]
    })

    # Define Sampling Conditions
    n_samples = 10
    values_per_dim = 10

    # Sample new Conditions
    nc = max_min_distance_sampler(data, condition_space,
                              n_samples, values_per_dim)

    # Plot Results
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.scatter(data.iv1, data.iv2, label="Already Observed Conditions")
    ax.scatter(nc.iv1, nc.iv2, label="Iteratively Proposed Conditions")

    # Add numbers to each point
    for i in range(len(nc)):
        v1 = nc.iloc()[i].iv1
        v2 = nc.iloc()[i].iv2
        ax.text(v1, v2, str(i+1), fontsize=15, ha='left', va='top')

    ax.set_ylabel("Indep.Var. 1")
    ax.set_xlabel("Indep.Var. 2")

    ax.legend()
    ax.set_title("Eucl. Distance- Based Coverage Sampling")
    return



def sampling_example_3d():
    # Define the ranges for 3 independent variables
    condition_space = pd.DataFrame({
        'iv1': np.linspace(0, 10, 100),
        'iv2': np.linspace(0, 10, 100),
        'iv3': np.linspace(0, 10, 100)
    })

    # Pretend some data has been sampled already
    data = pd.DataFrame({
        'iv1': [1, 3, 5, 6, 0],
        'iv2': [2, 4, 6, 7, 1],
        'iv3': [7, 8, 9, 9, 3]
    })

    # Define Sampling Conditions
    n_samples = 10
    values_per_dim = 10

    # Sample new Conditions
    nc = max_min_distance_sampler(data, condition_space,
                              n_samples, values_per_dim)

    # Plot Results (3D)
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(data.iv1, data.iv2, data.iv3, label="Already Observed Conditions")
    ax.scatter(nc.iv1, nc.iv2, nc.iv3, label="Iteratively Proposed Conditions")

    # Add numbers to each point
    for i in range(len(nc)):
        v1 = nc.iloc()[i].iv1
        v2 = nc.iloc()[i].iv2
        v3 = nc.iloc()[i].iv3
        ax.text(v1, v2, v3, str(i+1), fontsize=15, ha='left', va='top')

    ax.set_ylabel("Indep.Var. 1")
    ax.set_xlabel("Indep.Var. 2")
    ax.set_zlabel("Indep.Var. 3")
    ax.legend()
    ax.set_title("Eucl. Distance- Based Coverage Sampling")