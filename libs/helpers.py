import numpy as np
import pandas as pd


def approximate_value(series, search_time):
    """
    Approximate the value at a specific search_time in a time series.

    Parameters:
    - series: Pandas Series with a DateTimeIndex.
    - search_time: The time for which to approximate the value.

    Returns:
    - The approximated value at search_time.
    """
    # Ensure search_time is a Pandas Timestamp
    search_time = pd.to_datetime(search_time)

    # If the search_time is already in the series, return the corresponding value
    if search_time in series.index:
        return series.loc[search_time]

    # Create a temporary Series by appending search_time with NaN, interpolate, and extract the value
    temp_series = series.append(pd.Series([np.nan], index=[search_time])).sort_index()
    temp_series.interpolate(method='time', inplace=True)

    return temp_series.loc[search_time]


def main():
    pass


if __name__ == "__main__":
    main()
    pass
