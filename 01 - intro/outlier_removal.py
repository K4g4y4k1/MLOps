import pandas as pd

def remove_outliers_iqr(df, column, lower_quantile=0.25, upper_quantile=0.75, factor=1.5):
    """
    Removes outliers from a single column in a DataFrame using the IQR method.

    Parameters:
        df (pd.DataFrame): Input DataFrame
        column (str): Column to filter
        lower_quantile (float): Lower quantile (default 0.25)
        upper_quantile (float): Upper quantile (default 0.75)
        factor (float): Multiplier for IQR (default 1.5)

    Returns:
        pd.DataFrame: Filtered DataFrame without outliers in that column
    """
    Q1 = df[column].quantile(lower_quantile)
    Q3 = df[column].quantile(upper_quantile)
    IQR = Q3 - Q1
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR

    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return filtered_df