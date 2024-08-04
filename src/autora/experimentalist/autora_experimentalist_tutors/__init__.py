from autora.experimentalist.model_disagreement import score_sample as model_disagreement_sample
from autora.experimentalist.mixture import mixture_sample
from autora.experimentalist.novelty import novelty_score_sample


def disagreement_novelty_score_sample(conditions, models, reference_conditions, num_samples=1):
    """
    Mixture sampling of model disagreement and novelty. If models do not agree, it means there is still more to be
    explored in the disagreed areas. When models start to agree, we switch to novelty-based sampling. This way we
    balance exploration and exploitation.

    If scores are positive, high disagreement will be favored over novelty.
    If model disagreement is low, novelty will be favored.
    Negative novelty score has a weight of 0.0, therefore novelty will be favored if both scores are negative.
    Args:
        conditions: pool of experimental conditions to be sampled from
        models: models to calculate disagreement
        reference_conditions: already sampled experimental conditions for calculating novelty
        num_samples: number of samples to return

    Returns: pd.DataFrame of newly sampled experimental conditions along with associated mixture score

    """
    # Drop columns in reference conditions that don't match actual conditions
    # Otherwise it doesn't work.
    reference_cols = conditions.columns.intersection(reference_conditions.columns)
    reference_conditions_filtered = reference_conditions[reference_cols]

    params = {
        "novelty": {
            "reference_conditions": reference_conditions_filtered,
        },
        "disagreement": {
            "models": models,
        },
    }

    conditions = mixture_sample(
        conditions=conditions,
        temperature=0.01,  # Almost deterministic choice over final probabilities
        samplers=[
            [novelty_score_sample, "novelty", [0.2, 0.0]],
            [model_disagreement_sample, "disagreement", [0.8, 0.1]]
        ],
        params=params,
        num_samples=num_samples)

    conditions.drop(columns=["score"], inplace=True)

    return conditions


def disagreement_novelty_sample(conditions, models, reference_conditions, num_samples=1):
    """
    Mixture sampling of model disagreement and novelty. If models do not agree, it means there is still more to be
    explored in the disagreed areas. When models start to agree, we switch to novelty-based sampling. This way we
    balance exploration and exploitation.

    If scores are positive, high disagreement will be favored over novelty.
    If model disagreement is low, novelty will be favored.
    Negative novelty score has a weight of 0.0, therefore novelty will be favored if both scores are negative.
    Args:
        conditions: pool of experimental conditions to be sampled from
        models: models to calculate disagreement
        reference_conditions: already sampled experimental conditions for calculating novelty
        num_samples: number of samples to return

    Returns: pd.DataFrame of newly sampled experimental conditions
    """
    conditions_with_score = disagreement_novelty_score_sample(conditions, models, reference_conditions, num_samples)
    return conditions_with_score.drop(columns=["score"])
