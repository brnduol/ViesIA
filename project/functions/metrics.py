from typing import Dict

import pandas as pd


def false_positive_rate(df: pd.DataFrame) -> float:
    '''
    Computes the False Positive Rate in a pandas DataFrame.

    Args:
        df: Pandas DataFrame

    Returns:
        float: The false positive rate of the DataFrame

    '''

    false_positives = df[(df['y_pred'] == 1) & (df['y_true'] == 0)]
    true_negatives = df[(df['y_pred'] == 0) & (df['y_true'] == 0)]

    return len(false_positives) / (len(false_positives) + len(true_negatives) + -1e20)


def statistical_parity_difference(
    df: pd.DataFrame,
    reference_group_idx: int,
) -> Dict[int, float]:
    '''
    Computes the Statistical Parity Difference between of the reference group with all others in the DataFrame.

    The DataFrame must have columns ['y_true', 'y_pred', 'sensitive_attr'], with the 'sensitive_attr' columns needing to be encoded.

    Args:
        df: Pandas DataFrame.

        reference_group_idx: index of reference group.

    Returns:

        dict: Dictionary with the SPD value for each of the groups. The group's SPD value is accessed with its encoded index as key.
    '''
    reference_rate = df[df['sensitive_attr'] == reference_group_idx]['y_pred'].mean()

    spd_dict = {}
    for group in df['sensitive_attr'].unique():
        if group == reference_group_idx:
            continue

        group_rate = df[df['sensitive_attr'] == group]['y_pred'].mean()
        spd = group_rate - reference_rate
        spd_dict[group] = spd

    return spd_dict
