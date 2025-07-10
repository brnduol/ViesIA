from typing import Dict

import pandas as pd


def predictive_equality_fpr_diff(
    df: pd.DataFrame,
    privileged_value: int,
    unprivileged_value: int,
    y_true_col: str = "y_true",
    y_pred_col: str = "y_pred",
    sensitive_col: str = "sensitive_attr",
) -> float:
    '''
    Calculates the Predictive Equality metric (difference in false positive rates - FPR).

    Args:
        df: DataFrame containing the data
        y_true_col: name of the column with the true labels
        y_pred_col: name of the column with the model's predictions
        sensitive_col: name of the column with the sensitive attribute (e.g., gender)
        unprivileged_value: value of the unprivileged group in the sensitive column
        privileged_value: value of the privileged group in the sensitive column

    Returns:
        FPR_diff: difference in FPR between unprivileged and privileged groups
    '''

    unpriv = df[(df[sensitive_col] == unprivileged_value) & (df[y_true_col] == 0)]
    priv = df[(df[sensitive_col] == privileged_value) & (df[y_true_col] == 0)]

    fp_unpriv = (unpriv[y_pred_col] == 1).sum()
    fp_priv = (priv[y_pred_col] == 1).sum()

    fpr_unpriv = fp_unpriv / len(unpriv) if len(unpriv) > 0 else 0
    fpr_priv = fp_priv / len(priv) if len(priv) > 0 else 0

    return fpr_unpriv - fpr_priv


def statistical_parity_difference(
    df: pd.DataFrame,
    privileged_value: int,
) -> Dict[int, float]:
    '''
    Computes the Statistical Parity Difference between of the reference group with all others in the DataFrame.

    The DataFrame must have columns ['y_true', 'y_pred', 'sensitive_attr'], with the 'sensitive_attr' columns needing to be encoded.

    Args:
        df (pd.Dataframe): Pandas DataFrame.

        privileged_value (int): index of privileged group.

    Returns:
        dict: Dictionary with the SPD value for each of the groups. The group's SPD value is accessed with its encoded index as key.
    '''
    reference_rate = df[df['sensitive_attr'] == privileged_value]['y_pred'].mean()

    spd_dict = {}
    for group in df['sensitive_attr'].unique():
        if group == privileged_value:
            continue

        group_rate = df[df['sensitive_attr'] == group]['y_pred'].mean()
        spd = group_rate - reference_rate
        spd_dict[group] = spd

    return spd_dict


def false_positive_rate(df: pd.DataFrame) -> float:
    '''
    Computes the False Positive Rate in a pandas DataFrame.

    Args:
        df (pd.DataFrame): Pandas DataFrame

    Returns:
        float: The false positive rate of the DataFrame
    '''

    false_positives = df[(df['y_pred'] == 1) & (df['y_true'] == 0)]
    true_negatives = df[(df['y_pred'] == 0) & (df['y_true'] == 0)]

    return len(false_positives) / (len(false_positives) + len(true_negatives) + -1e20)


def disparate_impact(
    df: pd.DataFrame,
    sensitive_attr: str,
    privileged_value: int,
    positive_label: int = 1,
) -> float:
    '''
    Calculates Disparate Impact (DI) in a DataFrame.

    Parameters:
        df(pd.DataFrame): pandas DataFrame with columns ['y_pred', 'y_true', sensitive_attr]
        sensitive_attr (str): name of the sensitive attribute column (e.g. 'race')
        privileged_value (int): value in sensitive_attr considered as the privileged group
        positive_label (int): the label considered as a "positive outcome" (default=1)

    Returns:
        DI: float, the disparate impact value
    '''

    privileged = df[df[sensitive_attr] == privileged_value]
    unprivileged = df[df[sensitive_attr] != privileged_value]

    p_priv = (privileged['y_pred'] == positive_label).mean()
    p_unpriv = (unprivileged['y_pred'] == positive_label).mean()

    if p_priv == 0:
        return float('inf') if p_unpriv > 0 else 1.0

    return round(p_unpriv / p_priv, 3)
