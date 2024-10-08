import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import random
from swmmRouting.routing import Router
from swmmRouting.seeding import Seeder
from swmm_api.input_file import read_inp_file
from swmm_api.output_file import read_out_file
from swmm_api.external_files.dat_timeseries import write_swmm_timeseries_data, read_swmm_timeseries_data
from swmm_api.run_swmm.run_epaswmm import swmm5_run_epa
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


def dissipation_rate(X, b1, b2, b3, b4, b5):
    wipes, flow_rate, time = X
    return (b1 * wipes + b2 * flow_rate + b3 * wipes * flow_rate + b4) * np.exp(b5 * time)

def fitted_dissipation_rate(X):
    """
    Calculates dissipation rate
    Args:
        X: [wipes, flow_rate, age]

    Returns:
        dissipation rate (float)
    """
    params = [ 0.10194453,  1.23100065, -0.04354149, 10.65112679, -0.38916378]
    return dissipation_rate(X, *params)

def next_blockage(onion, flow, accumulation, current_time, timestep=1):
    """

    Args:
        onion (list): Array of onion-layers consisting of [size in number of wipes, flushtime]
        flow (float): Flow rate in l/s
        accumulation (float): Number of new arriving wipes in next timestep,
        current_time (timestamp): current time at simulation step
        timestep (float): Timestep length in hours

    Returns:
        sizes, flushtimes
    """
    sizes = onion["size"]
    flushtimes = onion["flushtime"]
    ages = (current_time - flushtimes) / np.timedelta64(1, "h")
    # calculate new layer sizes after timestep
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

def calc_blockage_series(df, max_age=48):
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
                                                                accumulation[i], flush_times[i])
    blockage = [sum(layer["size"]) for layer in onion]

    # calculate orifice setting
    blockage_scaled, flowrates_scaled = scaler.transform(np.c_[blockage, flowrates]).T
    orifice = orifice_model_fitted(blockage_scaled, flowrates_scaled)
    df["orifice"] = orifice
    df["blockage"] = blockage
    return df


class BlockageSimulation:
    def __init__(self, settings):
        self.model_path = settings.get("model_path")
        self.out_path = settings.get("out_path")
        self.pop_path = settings.get("pop_path")
        self.orifice_path = settings.get("orifice_path")
        self.result_path = settings.get("result_path")
        self.defpat = settings.get("defpat")
        inp = read_inp_file(self.model_path)
        self.start = datetime.combine(inp.OPTIONS.get("REPORT_START_DATE"), inp.OPTIONS.get("REPORT_START_TIME"))
        self.end = datetime.combine(inp.OPTIONS.get("END_DATE"), inp.OPTIONS.get("END_TIME"))
        self.target_node = settings.get("target_node")
        self.target_link = settings.get("target_link")
        return

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
        with read_out_file(fn_out) as out:
            # import outfile as dataframe
            df = out.get_part(kind=swmm_objs.LINK, variable=swmm_vars.LINK.VELOCITY)
            s_flows = df.loc[self.start:self.end, self.target_link].resample("1h").mean()
            s_flows.rename("flow", inplace=True)
        with pd.HDFStore(self.result_path) as store:
            store["/iteration_0"] = pd.concat([s_orifice, s_blockage, s_flows], axis=1)
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

    def run_iterations(self, df, n_iterations, router):
        # iteratively calculate orifice settings and blockage series
        for i in range(1, n_iterations+1):
            # calculate blockage series from accumulating wipes and flows
            logger.info(f"starting iteration {i}")
            logger.info(f"calculating orifice settings")
            df = calc_blockage_series(df[["arrivals", "flow", "accumulation"]].copy())
            # write orifice settings to .dat file
            write_swmm_timeseries_data(df["orifice"],
                                       r"C:\Users\albert\PycharmProjects\PacketSWMM\sample_data\orifice_settings.dat")
            # run swmm model for updated hydraulic results
            logger.info(f"running hydraulic model")
            fn_rpt, fn_out = swmm5_run_epa(self.model_path)
            self.out_path = fn_out
            # read new hydraulic results
            router.get_flows_from_outfile(path_out=self.out_path, start=self.start, end=self.end)
            df["flow"] = router.df_flows[self.target_link].resample("1h").mean()
            logger.info(f"writing results")
            with pd.HDFStore(self.result_path) as store:
                store[f"iteration_{i}"] = df[["orifice", "flow", "blockage"]]
            logger.info(f"iteration {i} complete")
        return df

    def run_simulation(self):
        # run blank
        self.blank_model_run()
        # route initial wipes through network
        arrivals = self.initial_routing()
        # get flow rates from target link
        s_flowrates = self.router.df_flows[self.target_link].resample("1h").mean().rename("flow")
        # concatenate flowrates and arrivals to common dataframe
        df = pd.concat([arrivals, s_flowrates], axis=1)
        df["arrivals"].fillna(0, inplace=True)
        # sample snagging and accumulating wipes from arriving
        df = self.sample_accumulation(df)
        # iteratively adapt orifice settings to hydraulic simulation
        df = self.run_iterations(df, n_iterations=3, router=self.router)
        return


def run_sample_sim():
    df_defpat = pd.Series(0.9 * np.array([1.4, 0.3, 0.1, 0.0, 0.3, 1.7, 9.1, 21, 13, 9, 6.9, 4.9,
                                           1.9, 3.6, 2.5, 2, 2.9, 2.3, 4.1, 4.0, 2.7, 2.1, 2.2, 2.0]) / 100)

    settings = dict(model_path = r"..\sample_data\sample_model.inp",
                    pop_path = r"../sample_data/pop_data.csv",
                    out_path = r"..\sample_data\sample_model.out",
                    orifice_path = r"../sample_data/orifice_settings.dat",
                    result_path = r"../sample_data/blockage_results.hd5",
                    defpat = df_defpat,
                    target_node = "MH3295504178",
                    target_link = "MH3295504178.1")

    blockage_sim = BlockageSimulation(settings)
    blockage_sim.run_simulation()


def run_original():
    from src.swmmRouting.routing import Router
    from src.swmmRouting.seeding import Seeder
    from swmm_api.external_files.dat_timeseries import write_swmm_timeseries_data, read_swmm_timeseries_data
    from swmm_api.run_swmm.run_epaswmm import swmm5_run_epa

    # settings for packet router
    model_path = r"..\sample_data\sample_model.inp"
    pop_path = r"../sample_data/pop_data.csv"
    orifice_path = r"../sample_data/orifice_settings.dat"

    # cleaning orifice setting data
    s = read_swmm_timeseries_data(r"C:\Users\albert\PycharmProjects\PacketSWMM\sample_data\orifice_settings.dat")
    s.loc[:] = 0
    write_swmm_timeseries_data(s, orifice_path, drop_zeros=False)

    # run model initial run
    out_path = r"..\sample_data\sample_model.out"
    # rpt_path, out_path = swmm5_run_epa(model_path)

    target_node = "MH3295504178"
    start = pd.to_datetime("01.04.2015 00:00", format="%d.%m.%Y %H:%M")
    end = pd.to_datetime("01.06.2015 00:00", format="%d.%m.%Y %H:%M")

    # pattern for distribution of defecation events, total sum of 1.5 defecations per day
    # df_defpat = pd.Series(1.5 * np.array([1.4, 0.3, 0.1, 0.0, 0.3, 1.7, 9.1, 21, 13, 9, 6.9, 4.9,
    #                                      1.9, 3.6, 2.5, 2, 2.9, 2.3, 4.1, 4.0, 2.7, 2.1, 2.2, 2.0])/100)
    df_defpat = pd.Series(0.9 * np.array([1.4, 0.3, 0.1, 0.0, 0.3, 1.7, 9.1, 21, 13, 9, 6.9, 4.9,
                                           1.9, 3.6, 2.5, 2, 2.9, 2.3, 4.1, 4.0, 2.7, 2.1, 2.2, 2.0]) / 100)
    # number of wipes sent per flush

    # prepare seeder
    df_population = pd.read_csv(pop_path)
    seeder = Seeder(df_seeding_population=df_population, hourly_probability=df_defpat)
    seed_table = seeder.generate_seeds(start=start, end=end)

    # prepare router
    router = Router()
    router.get_network_from_inpfile(model_path)
    router.get_flows_from_outfile(path_out=out_path, start=start, end=end)

    # create seeds and route table
    rtable = router.from_seeding_table(seeding_table=seed_table, target=target_node)
    routed = router.route_table(rtable)

    # get arrivals from target node and aggregate to 1 hour timeslots
    arrivals = routed[target_node].dropna().sort_values()
    hourly_arrivals = arrivals.dt.floor("H")
    s_arrivals = hourly_arrivals.groupby(hourly_arrivals).size()
    s_arrivals = s_arrivals[s_arrivals.index > start]
    s_arrivals.rename("arrivals", inplace=True)

    # get flow rates from target link
    s_flowrates = router.df_flows["MH3295504178.1"].resample("1h").mean().rename("flow")
    # concatenate flowrates and arrivals to common dataframe
    df = pd.concat([s_arrivals, s_flowrates], axis=1)
    df["arrivals"].fillna(0, inplace=True)

    # sample snagging and accumulating wipes from arrivals
    accumulation = np.zeros(len(df))
    pile = 0
    wipes_per_flush = 0.1
    for i, (idx, values) in enumerate(df.iterrows()):
        n_arrivals = (wipes_per_flush * values["arrivals"]).round(0).astype(int)
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

    # iteratively calculate orifice settings and blockage series
    for i in range(3):
        df = calc_blockage_series(df)
        write_swmm_timeseries_data(df["orifice"],
                                   r"C:\Users\albert\PycharmProjects\PacketSWMM\sample_data\orifice_settings.dat")
        fn_rpt, fn_out = swmm5_run_epa(model_path)
        router.get_flows_from_outfile(path_out=out_path, start=start, end=end)
        df["flow"] = router.df_flows["MH3295504178.1"].resample("1h").mean()
    return


def main():
    run_sample_sim()

if __name__ == "__main__":
    main()
    pass
