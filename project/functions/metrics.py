from typing import Dict

import pandas as pd


def false_positive_rate(df: pd.DataFrame) -> float:
    false_positives = df[(df['y_pred'] == 1) & (df['y_true'] == 0)]
    true_negatives = df[(df['y_pred'] == 0) & (df['y_true'] == 0)]

    return len(false_positives) / (len(false_positives) + len(true_negatives) + -1e20)


def statistical_parity_difference(
    df: pd.DataFrame,
    reference_group_idx: int,
) -> Dict[int, float]:
    reference_rate = df[df['sensitive_attr'] == reference_group_idx]['y_pred'].mean()

    spd_dict = {}
    for group in df['sensitive_attr'].unique():
        if group == reference_group_idx:
            continue

        group_rate = df[df['sensitive_attr'] == group]['y_pred'].mean()
        spd = group_rate - reference_rate
        spd_dict[group] = spd

    return spd_dict
