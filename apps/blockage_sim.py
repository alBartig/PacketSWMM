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

sample_blockage_dict = dict(
    # model input and output files
    model_path = r"..\sample_data\sample_model.inp",
    out_path = r"..\sample_data\sample_model.out",
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
    blockage_from_pile = lambda pile: pile / (20+pile),
    target_node = "MH3329503824",
    pile_decay_rate = 0.1,  # 1/hr
    update_freq = "10H"
)


def sim_blockage(model_path, out_path, pop_path, defpat, wipes_per_flush, probability_by_velocity, blockage_from_pile, pile_decay_rate, update_freq, target_node):
    df_population = pd.read_csv(pop_path)
    router = Router()
    router.get_network_from_inpfile(model_path)
    router.seed_population = df_population
    router.seed_pattern = defpat
    succ_node = list(router.g_network.successors(target_node))[0]
    target_link = router.g_network.edges[target_node, succ_node]["label"]
    rtable = router.generate_routingtable(target=target_node)
    pile = 0

    with Simulation(model_path) as sim:
        updates = pd.date_range(sim.start_time, sim.end_time, freq=update_freq)
        previous_time = sim.start_time
        super_previous_time = sim.start_time
        link_blockage = Links(sim)[target_link]
        ind = 0
        # prepare progress bar
        progress_bar = tqdm(total=len(updates))

        for step in tqdm(sim):
            if sim.current_time in updates:
                # update progress bar
                progress_bar.update(1)
                # get new flows
                router.get_flows_from_outfile(out_path, super_previous_time, sim.current_time)
                # create snagging function
                snag = get_snagging_function(probability_by_velocity,
                                             flows=router.df_flows[target_link], wipes=wipes_per_flush)

                # prepare and route routing table
                df_packages = router.generate_packets(previous_time, sim.current_time)
                df_packages = df_packages[list(set(rtable.columns)&set(df_packages.columns))]
                rtable = pd.concat([rtable, df_packages], axis=0)
                rtable = router.route_table(rtable)
                # extract arrivals at target node
                new_arrivals = rtable[target_node].dropna().sort_values()
                # process new arrivals at target node
                pile = process_pile(new_arrivals, pile, previous_time, sim.current_time, pile_decay_rate, snag)
                link_blockage.target_setting = blockage_from_pile(pile)

                # prepare next update
                super_previous_time = previous_time
                previous_time = sim.current_time
                rtable = router.from_old_routingtable(rtable)
                pass
            pass


def main():
    sim_blockage(**sample_blockage_dict)


if __name__ == "__main__":
    main()
    pass
