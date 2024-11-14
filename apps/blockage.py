from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import random
from swmmRouting.routing import Router
from swmmRouting.seeding import Seeder
from swmm_api.input_file import read_inp_file
from swmm_api.output_file import read_out_file
from swmm_api.external_files.dat_timeseries import write_swmm_timeseries_data, read_swmm_timeseries_data
from swmm_api.run_swmm.run_epaswmm import swmm5_run_epa, run_swmm_custom, get_swmm_command_line_auto
from swmm_api.output_file import VARIABLES as swmm_vars
from swmm_api.output_file import OBJECTS as swmm_objs
import warnings
from swmm_api.output_file.extract import SwmmOutExtractWarning
from datetime import datetime
import logging

warnings.filterwarnings("ignore", category=SwmmOutExtractWarning)

# Create or get a named logger
logger = logging.getLogger("blockage_logger")
# Set the logging level for this logger
logger.setLevel(logging.INFO)
# Create a handler (e.g., a StreamHandler to print to console)
console_handler = logging.StreamHandler()
# Set level for the handler
console_handler.setLevel(logging.INFO)
# Create a formatter
formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
# Add the formatter to the handler
console_handler.setFormatter(formatter)
# Add the handler to the logger
logger.addHandler(console_handler)

def dissipation_rate(X, b1, b2, b3, b4):
    wipes, flow_rate, time = X
    return (b1 * wipes + b2) * np.exp(b3 * time) * flow_rate**b4

def fitted_dissipation_rate(X):
    """
    Calculates dissipation rate
    Args:
        X: [wipes, flow_rate, age]

    Returns:
        dissipation rate (float)
    """
    params = [-0.31000456, 16.45328033, -0.38209651,  0.22813606]
    return dissipation_rate(X, *params)

def next_blockage(onion, flow, accumulation, current_time, timestep=1, bulk_approach=False):
    """

    Args:
        onion (list): Array of onion-layers consisting of [size in number of wipes, flushtime]
        flow (float): Flow rate in l/s
        accumulation (float): Number of new arriving wipes in next timestep,
        current_time (timestamp): current time at simulation step
        timestep (float): Timestep length in hours
        bulk_approach (bool): If True will use bulk approach instead of onion approach

    Returns:
        sizes, flushtimes
    """
    sizes = onion["size"]
    flushtimes = onion["flushtime"]
    ages = (current_time - flushtimes) / np.timedelta64(1, "h")
    avg_ages = np.mean(ages) * np.ones_like(ages)
    # calculate new layer sizes after timestep
    if bulk_approach:
        try:
            sizes = sizes * np.exp(-fitted_dissipation_rate([sizes, flow, avg_ages])*timestep)
        except:
            print("error")
            pass
    else:
        sizes = sizes * np.exp(-fitted_dissipation_rate([sizes, flow, ages])*timestep)
    # add second oldest layer to oldest layer and shift the rest one field
    new_sizes = sizes.copy()
    new_sizes[-1] += new_sizes[-2]
    new_sizes[1:-1] = sizes[:-2]
    new_sizes[0] = accumulation
    # shift the flushtime window for one timestep
    new_flushtimes = flushtimes + np.timedelta64(timestep, "h")
    return new_sizes, new_flushtimes

data = {
    "BP": [30, 25, 20, 15, 10, 0, 30, 25, 20, 15, 0, 30, 25, 20, 15, 10, 0],
    "Q": [2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 10, 10, 10, 10, 10, 27, 27, 27, 27, 27, 27],
    "S": [0.1041, 0.1065, 0.1100, 0.1299, 0.1303, 1.0000, 0.276, 0.278, 0.295, 0.31, 1, 0.506, 0.520, 0.550, 0.560, 0.596, 1.000]
}
df_settings = pd.DataFrame(data)
# Fit the StandardScaler on the original training data
scaler = StandardScaler()
scaler.fit(df_settings[['BP', 'Q']])

# Define the model function
def orifice_model(BP, Q, b0, b1, c1):
    c = c1 * Q
    # return np.maximum((1 - c) * np.exp(b0 * BP + b1) + c, 0)
    return np.maximum(0, (1 - c) * np.exp(b0 * BP + b1) + c)

def orifice_model_fitted(BP, Q):
    # Parameters from the trained model
    params = [-0.55, -1.02, 0.24]
    return orifice_model(BP, Q, *params)

def snagging_probability(q):
    # q: flow rate in l/s
    return (-1.0489*q + 97.143) / 100

def snag(flowrate):
    # calculation of whether wipe snags
    q = flowrate
    probability = snagging_probability(q)
    return random.random() < probability

def accum_probability(q):
    # q: flow rate in l/s
    return (-0.2006*q + 68.174) / 100

def accumulate(flowrate):
    # calculation of whether wipe accumulates
    q = flowrate
    probability = accum_probability(q)
    return random.random() < probability

def calc_blockage_series(df, max_age=48, bulk=False):
    # prepare arrays
    df["orifice"] = 1
    accumulation, flowrates, orifice = df[["accumulation", "flow", "orifice"]].dropna().values.T
    flush_times = df[["accumulation", "flow", "orifice"]].dropna().index.to_numpy()
    # Create an empty structured array
    onion = np.zeros((len(flush_times), max_age), dtype=[('size', 'f4'), ('flushtime', 'datetime64[h]')])
    # Fill the 'timestamp_field' with a range of timestamps: current_time - i*1h
    onion[0]['flushtime'] = np.arange(df.index[0], df.index[0] - max_age * np.timedelta64(1, 'h'),
                                   -np.timedelta64(1, 'h'))
    # calculate pile for each timestep
    for i in range(1, len(flush_times)-1, 1):
        onion[i]["size"], onion[i]["flushtime"] = next_blockage(onion[i-1], flowrates[i-1],
                                                                accumulation[i], flush_times[i], bulk_approach=bulk)
    blockage = [sum(layer["size"]) for layer in onion]

    # calculate orifice setting
    blockage_scaled, flowrates_scaled = scaler.transform(np.c_[np.minimum(blockage, 999), np.minimum(flowrates, 999)]).T

    orifice = orifice_model_fitted(blockage_scaled, flowrates_scaled)
    df["orifice"] = orifice
    df["blockage"] = blockage
    return df


class BlockageSimulation:
    def __init__(self, settings):
        self.model_path = settings.get("model_path")
        self.out_path = settings.get("out_path")
        self.pop_path = settings.get("pop_path")
        if type(settings.get("orifice_path")) == Path:
            self.orifice_path = settings.get("orifice_path")
        else:
            self.orifice_path = Path(settings.get("orifice_path"))
        self.result_path = settings.get("result_path")
        self.defpat = settings.get("defpat")
        inp = read_inp_file(self.model_path)
        self.start = datetime.combine(inp.OPTIONS.get("REPORT_START_DATE"), inp.OPTIONS.get("REPORT_START_TIME"))
        self.end = datetime.combine(inp.OPTIONS.get("END_DATE"), inp.OPTIONS.get("END_TIME"))
        self.target_node = settings.get("target_node")
        self.target_link = settings.get("target_link")
        self.target_orifice = settings.get("target_orifice")
        return

    def read_flowrates(self, outpath=None, target="orifice"):
        if outpath is None:
            outpath = self.out_path
        if target == "orifice":
            target = self.target_orifice
        elif target == "link":
            target = self.target_link
        with read_out_file(outpath) as out:
            # import outfile as dataframe
            df = out.get_part(kind=swmm_objs.LINK, variable=swmm_vars.LINK.FLOW)
            s_flows = df.loc[self.start:self.end, target].resample("1h").mean()
            s_flows.rename("flow", inplace=True)
        return s_flows

    def read_nodedepths(self):
        with read_out_file(self.out_path) as out:
            # import outfile as dataframe
            df = out.get_part(kind=swmm_objs.NODE, variable=swmm_vars.NODE.DEPTH)
            s_depth = df.loc[self.start:self.end, self.target_node].resample("1h").mean()
            s_depth.rename("depth", inplace=True)
        return s_depth

    def blank_model_run(self):
        logger.info("creating empty orifice series")
        # create orifice settings file with full capacity
        index = pd.date_range(self.start, self.end, freq="1h")
        s_orifice = pd.Series(np.ones(len(index)), index=index)
        s_orifice.rename("orifice", inplace=True)
        s_blockage = s_orifice.copy().rename("blockage")
        s_blockage[:] = 0.0
        write_swmm_timeseries_data(s_orifice, self.orifice_path)
        # initial model run
        logger.info(f"running hydraulic model")
        fn_rpt, fn_out = swmm5_run_epa(self.model_path)
        self.out_path = fn_out
        s_flows = self.read_flowrates()
        s_depth = self.read_nodedepths()
        with pd.HDFStore(self.result_path, mode="w") as store:
            store["/iteration_0"] = pd.concat([s_orifice, s_blockage, s_flows, s_depth], axis=1)
        return

    def initial_routing(self):
        # prepare seeder
        df_population = pd.read_csv(self.pop_path)
        self.seeder = Seeder(df_seeding_population=df_population, hourly_probability=self.defpat)
        seed_table = self.seeder.generate_seeds(start=self.start, end=self.end)

        # prepare router
        self.router = Router()
        self.router.get_network_from_inpfile(self.model_path)
        self.router.get_flows_from_outfile(path_out=self.out_path, start=self.start, end=self.end)

        # create seeds and route table
        rtable = self.router.from_seeding_table(seeding_table=seed_table, target=self.target_node)
        routed = self.router.route_table(rtable)

        # get arrivals from target node and aggregate to 1 hour timeslots
        arrivals = routed[self.target_node].dropna().sort_values()
        hourly_arrivals = arrivals.dt.floor("H")
        s_arrivals = hourly_arrivals.groupby(hourly_arrivals).size()
        s_arrivals = s_arrivals[s_arrivals.index > self.start]
        s_arrivals.rename("arrivals", inplace=True)
        return s_arrivals

    def sample_accumulation(self, df):
        # sample snagging and accumulating wipes from arrivals
        accumulation = np.zeros(len(df))
        pile = 0
        for i, (idx, values) in enumerate(df.iterrows()):
            n_arrivals = (values["arrivals"]).round(0).astype(int)
            flowrate = values["flow"]
            accumulations = 0
            # if no pile exists, calculate snagging
            while (pile == 0) & (n_arrivals > 0):
                snagged = snag(flowrate)
                pile += snagged
                accumulations += snagged
                n_arrivals -= 1
            # calculate accumulation with remaining arriving wipes
            accumulations += sum([accumulate(flowrate) for _ in range(n_arrivals)])
            accumulation[i] = accumulations
        df["accumulation"] = accumulation
        return df

    def run_iterations(self, df, n_iterations, router, bulk):
        # iteratively calculate orifice settings and blockage series
        for i in range(1, n_iterations+1):
            # calculate blockage series from accumulating wipes and flows
            logger.info(f"starting iteration {i}")
            logger.info(f"calculating orifice settings")
            df = calc_blockage_series(df[["arrivals", "flow", "accumulation"]].copy(), bulk=bulk)
            # write orifice settings to .dat file
            write_swmm_timeseries_data(df["orifice"].round(3), self.orifice_path, drop_zeros=False)
            write_swmm_timeseries_data(df["orifice"].round(3), self.orifice_path.parent / f"or_setting_iteration{i}.dat",
                                       drop_zeros=False)
            # run swmm model for updated hydraulic results
            logger.info(f"running hydraulic model")
            if type(self.model_path) != Path:
                self.model_path = Path(self.model_path)
            fn_rpt = self.model_path.parent / f"rptfile_iteration{i}.rpt"
            fn_out = self.model_path.parent / f"outfile_iteration{i}.out"
            cmd = get_swmm_command_line_auto(self.model_path,
                                             fn_rpt,
                                             fn_out)
            run_swmm_custom(cmd)
            logger.info(f"Model run finished:\n"
                        f"New rpt-file at {fn_rpt}\n"
                        f"New out-file at {fn_out}")
            self.out_path = fn_out
            # read new hydraulic results
            df["flow"] = 0.0
            df["flow"] = self.read_flowrates(fn_out)
            df["depth"] = self.read_nodedepths()
            # write results
            logger.info(f"writing results")
            with pd.HDFStore(self.result_path) as store:
                store[f"iteration_{i}"] = df[["orifice", "flow", "blockage", "depth"]]
            try:
                logger.info(f"iteration {i} complete")
            except:
                print("error")
                pass
        return df

    def run_simulation(self, bulk):
        # run blank
        self.blank_model_run()
        # route initial wipes through network
        arrivals = self.initial_routing()
        # get flow rates from target link
        s_flowrates = self.read_flowrates()
        s_nodedepths = self.read_nodedepths()
        # concatenate flowrates and arrivals to common dataframe
        df = pd.concat([arrivals, s_flowrates, s_nodedepths], axis=1)
        df["arrivals"].fillna(0, inplace=True)
        # sample snagging and accumulating wipes from arriving
        df = self.sample_accumulation(df)
        # iteratively adapt orifice settings to hydraulic simulation
        df = self.run_iterations(df, n_iterations=3, router=self.router, bulk=bulk)
        return


def run_sample_sim():
    df_defpat = pd.Series(0.1 * np.array([1.4, 0.3, 0.1, 0.0, 0.3, 1.7, 9.1, 21, 13, 9, 6.9, 4.9,
                                          1.9, 3.6, 2.5, 2, 2.9, 2.3, 4.1, 4.0, 2.7, 2.1, 2.2, 2.0]) / 100)
    result_path = r"C:\Users\albert\PycharmProjects\PacketSWMM\sample_data\katys_model\blockage_results250.hd5"
    settings = dict(model_path=r"C:\Users\albert\PycharmProjects\PacketSWMM\sample_data\katys_model\Sample.inp",
                    pop_path=r"C:\Users\albert\PycharmProjects\PacketSWMM\sample_data\katys_model\Pop_data.csv",
                    out_path=r"C:\Users\albert\PycharmProjects\PacketSWMM\sample_data\katys_model\Sample.out",
                    orifice_path=r"C:\Users\albert\PycharmProjects\PacketSWMM\sample_data\katys_model\orifice_settings.dat",
                    result_path=result_path,
                    defpat=df_defpat,
                    target_node="MH4699705183",
                    target_link="MH4699705183.1",
                    target_orifice="O1")

    blockage_sim = BlockageSimulation(settings)
    bulk=False
    blockage_sim.run_simulation(bulk=bulk)


def main():
    run_sample_sim()

if __name__ == "__main__":
    main()
    pass
