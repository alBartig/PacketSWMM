import numpy as np
import pandas as pd
import random
from libs.helpers import approximate_value

s_per_hr = 3600


def _snag(fun, time, flows):
    velocity = approximate_value(flows, time)
    probability = fun(velocity)
    return random.random() < probability


def get_snagging_function(fun=None, flows=None, wipes=1):
    """
    generates function that takes time and flows as input and calculates randomly how many of the wipes have snagged
    Args:
        fun:
        flows:
        wipes:

    Returns:

    """
    if fun is None:
        fun = lambda v: 0.5

    if flows is None:
        def snagging_function(time, flows):
            snags = sum([_snag(fun, time, flows) for _ in range(wipes)])
            return snags
    else:
        def snagging_function(time):
            snags = sum([_snag(fun, time, flows) for _ in range(wipes)])
            return snags

    return snagging_function


def process_pile(new_arrivals, init_pile, start_time, end_time, pile_decay_rate, snag):
    pile = init_pile
    s_arrivals = pd.concat([new_arrivals, pd.Series([start_time, end_time],
                                                    index=['start', 'end'])]).rename("arrivals").sort_values()
    df_process = pd.DataFrame(s_arrivals)
    df_process["diff_times"] = df_process["arrivals"].diff().dt.total_seconds() / s_per_hr
    for arrival_time, diff_time in df_process[["arrivals", "diff_times"]].values:
        pile = pile*np.exp(-pile_decay_rate*diff_time) + snag(arrival_time)
    return pile


def main():
    pass


if __name__ == "__main__":
    main()
    pass
