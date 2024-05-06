import logging
from collections import deque

from helpers.log_config import setup_logging
setup_logging()

import numpy as np
from pyswmm import Simulation, Nodes, Links
import pandas as pd
from swmm_api import read_inp_file, read_out_file
from swmm_api.output_file.extract import SwmmOutExtractWarning
from libs.routing import Router, approximate_value
from tqdm.auto import tqdm
import random
import warnings
from libs.wipes import get_snagging_function, process_pile

warnings.filterwarnings("ignore", category=SwmmOutExtractWarning)
logger = logging.getLogger("sim_builder")

sample_blockage_dict = dict(
    # model input and output files
    # model_path = r"..\sample_data\sample_model.inp",
    # out_path = r"..\sample_data\sample_model.out",
    model_path = r"..\sample_data\simple_model.inp",
    out_path = r"..\sample_data\simple_model.out",
    # population data
    pop_path = r"../sample_data/pop_data.csv",
    # pattern for distribution of defecation events, total sum of 1.5 defecations per day
    defpat = pd.Series(1.5 * np.array([1.4, 0.3, 0.1, 0.0, 0.3, 1.7, 9.1, 21, 13, 9, 6.9, 4.9,
                                         1.9, 3.6, 2.5, 2, 2.9, 2.3, 4.1, 4.0, 2.7, 2.1, 2.2, 2.0])/100),
    # number of wipes sent per flush
    wipes_per_flush = 5,
    # snagging probability by velocity
    probability_by_velocity = lambda v: 0.5,
    # blockage / target setting depending on pile size
    blockage_from_pile = lambda pile: 1 - pile / (2+pile),
    target_node = "MH3295504178",
    pile_decay_rate = 0.1,  # 1/hr
    update_freq = "10H"
)


def sim_blockage(model_path, out_path, pop_path, defpat, wipes_per_flush, probability_by_velocity, blockage_from_pile, pile_decay_rate, update_freq, target_node):
    df_population = pd.read_csv(pop_path)
    logger.info
    router = Router()
    router.get_network_from_inpfile(model_path)
    router.seed_population = df_population
    router.seed_pattern = defpat
    succ_node = list(router.g_network.successors(target_node))[0]
    target_link = router.g_network.edges[target_node, succ_node]["label"]
    rtable = router.generate_empty_routingtable(target=target_node)
    pile = 0

    def update_simulation(rtable, pile, current_time, previous_time, super_previous_time):
        # get new flows
        router.get_flows_from_outfile(out_path, super_previous_time, sim.current_time)
        # create snagging function with updated flows
        snag = get_snagging_function(probability_by_velocity,
                                     flows=router.df_flows[target_link], wipes=wipes_per_flush)

        # prepare and route routing table
        df_packages = router.generate_packets(previous_time, current_time)
        df_packages = df_packages[list(set(rtable.columns) & set(df_packages.columns))]
        rtable = pd.concat([rtable, df_packages], axis=0)
        rtable = router.route_table(rtable)
        # extract arrivals at target node
        new_arrivals = rtable[target_node].dropna().sort_values()
        logger.info(f"{len(new_arrivals)} new arrivals at {current_time}")
        # process new arrivals at target node
        pile = process_pile(new_arrivals, pile, previous_time, sim.current_time, pile_decay_rate, snag)
        link_blockage.target_setting = blockage_from_pile(pile)
        logger.info(f"link {link_blockage} target setting set to {link_blockage.target_setting:.2f}")
        # generate starting routing table for next update step
        rtable = router.from_old_routingtable(rtable)
        return rtable, pile

    with Simulation(model_path) as sim:
        updates = pd.date_range(sim.start_time, sim.end_time, freq=update_freq, inclusive="neither")
        updates = updates.union([sim.end_time])
        update_queue = deque(updates)
        previous_time = sim.start_time
        super_previous_time = sim.start_time
        link_blockage = Links(sim)[target_link]
        # prepare progress bar
        progress_bar = tqdm(total=len(updates)-1)
        logger.info("Simulation started")
        for _ in tqdm(sim):
            if sim.current_time >= update_queue[0]:
                update_queue.popleft()
                logger.info(f"Simulation updated at {sim.current_time}")
                # update progress bar
                progress_bar.update(1)
                # update simulation
                rtable, pile = update_simulation(rtable, pile, sim.current_time, previous_time, super_previous_time)
                # prepare next update
                super_previous_time = previous_time
                previous_time = sim.current_time
                next_time = update_queue[0]
                pass
            pass
        # final update simulation, only to calculate pile in the end
        rtable, pile = update_simulation(rtable, pile, sim.end_time, previous_time, super_previous_time)


def main():
    sim_blockage(**sample_blockage_dict)


if __name__ == "__main__":
    main()
    pass
