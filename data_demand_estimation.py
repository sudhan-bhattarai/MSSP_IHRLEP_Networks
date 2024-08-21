# Libraries and other modules of the project

import os
import math
import json
import itertools
import numpy as np
import pandas as pd
from shapely.geometry import Point, LineString
import importlib
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx

import helper
import commands
import data_forecast_error_scenarios
from data_forecast_error_scenarios import createErrorSamples
import data_initial_inputs

# Helper functions for demand estimation
"""
## Demand computation at a period given the hurricane attributes and the expected landfall period

Function 'compute_demand' in the next cell is used to compute the demand vector at any period 't' given the realization of the hurricane attributes.

**Important inputs:**

- lead_time = 6 represents the number of periods before the landfall of the first demand realization. In other words, we may observe some demand > 0 at 6 periods before the landfall (expected) period. We do not observe any demand (demand = 0) if the expected landfall period ($t_s$) at a period 't' is more than 6 periods from 't', i.e., $\text{demand}_t = 0 \text{ if } t_s - t > 6$.

#### Demand by hurricane intensity (or category)

In the following cell, function 'demandByWs' computes the demand contribution given the hurricane category ($cat$). We use the following relation to define the demand-by-intensity function:
$$
\text{demand-by-intensity} =
\begin{cases}
    0.0 & \text{ if } cat = 0 \text{, i.e., wind speed $\leq$ 73 } \\
    0.2 & \text{ if } cat = 1 \text{, i.e., 74 $\leq$ wind speed $\leq$ 95 }  \\
    0.4 & \text{ if } cat = 2 \text{, i.e., 96 $\leq$ wind speed $\leq$ 110 }  \\
    0.6 & \text{ if } cat = 3 \text{, i.e., 111 $\leq$ wind speed $\leq$ 129 }  \\
    0.8 & \text{ if } cat = 4 \text{, i.e., 130 $\leq$ wind speed $\leq$ 156 }  \\
    1.0 & \text{ if } cat = 5 \text{, i.e., 157 $\leq$ wind speed }  \\
\end{cases}
$$
"""


def demand_deterministic_landfall(data, hurr_pos, hurr_cat, t):
    """Compute demand for the given Hurricane position, category,
    and the time period in deterministic landfall case.

    hurr_pos : tuple
        (long, lat) as the position of hurricane track.
    hurr_cat: int
        Hurricane category = 0, 1, 2, 3, 4, or 5.
    t : int
        time period = 0, 1, ...,9, or 10.
    """
    DF = np.zeros([data.I])
    # First demand is realized 4 periods before landfall.
    if t > 4:
        dY = [0] * data.I
        dX = [0] * data.I
        for i in range(data.I):
            proj_point = data.study_line.interpolate(
                data.study_line.project(Point(hurr_pos))
                )
            dY[i] = math.sqrt(
                ((proj_point.x - data.DP_GIS[i, 0]) * data.Xmiles)**2 +
                ((proj_point.y - data.DP_GIS[i, 1]) * data.Ymiles)**2
                )
            dX[i] = math.sqrt(
                ((proj_point.x - hurr_pos[0]) * data.Xmiles)**2 +
                ((proj_point.y - hurr_pos[1]) * data.Ymiles)**2
                )
            if data.demand_zone.contains(Point(hurr_pos)):
                DF_by_int = helper.demandByWS(hurr_cat)
                DF_by_time = 1 - dX[i] / data.x_max
                DF_by_track = 1 - dY[i] / data.study_region_dist
                DF[i] = DF_by_track * DF_by_time * DF_by_int
    return DF.round(3).tolist()


def demand_random_landfall(data, hurr_pos, hurr_cat, t, ts):
    """Compute demand for the given Hurricane position, category,
    the time period, and expected landfall period in random landfall case.

    hurr_pos : tuple
        (long, lat) as the position of hurricane track.
    hurr_cat: int
        Hurricane category = 0, 1, 2, 3, 4, or 5.
    t : int
        Time period = 0, 1, ...,T_max.
    ts : int
        Expected landfall period = t, ..., T_max
    """
    if t > ts:
        print(f"error! Ets{ts} t{t}  pos ({hurr_pos}) Cat({hurr_cat})")
        exit(0)
    DF = np.zeros([data.I])
    lead_time = 6
    if ts - t <= lead_time:
        dY = list(map(lambda y: y * data.Ymiles,
                      list(abs(data.DP_GIS[:, 1] - hurr_pos[1]))))
        dX = list(map(lambda x: x * data.Xmiles,
                      list(abs(data.DP_GIS[:, 0] - hurr_pos[0]))))
        dist = np.array([math.sqrt(dY[i] ** 2 + dX[i] ** 2)
                         for i in range(data.I)])
        if np.max(dist) <= data.x_max:
            for i in range(data.I):
                DF_by_time = 1 - (ts - t) / lead_time
                DF_by_int = helper.demandByWS(hurr_cat)
                DF_by_track = 1 - dist[i] / data.x_max
                DF[i] = DF_by_track * DF_by_time * DF_by_int
    return DF.round(3).tolist()

def create_hurricane_scen(data, args, err):
    """ Create Hurricane scenarios (track and intensity) using the
    point forecast and the forecast errors.

    data: Class
        Data struct with initial inputs
    err: dict = {"err_type": {"scen_index": list(err for all periods)}}
        Contains forecast error data for all error types, for a
        number of scenarions and all periods
    """
    T = data.T
    S = len(err["track"])
    scen_T = {}
    scen_I = {}
    scen_Cat = np.zeros([S, T]).astype("int")
    for s in range(S):
        # Add intensity forecast errors directly to the point forecast.
        scen_I[s] = data.FORE.WS[:T] + err["intensity"][s][:T]
        # Compute hurricane category from intensity (wind speed).
        scen_Cat[s, :] = list(map(lambda x: helper.get_hurr_cat(x), scen_I[s]))
        if args['landfall'] == 'd':
            # For track, convert forecast errors in miles to (long, lat) degrees, then apply rotation, and add to the point forecast
            scen_T[s] = np.column_stack((list(x + data.X_rotate * e/data.Xmiles for x, e in zip(data.FORE.X[:T], err["track"][s][:T])),
                                         list(y + data.Y_rotate * e/data.Ymiles for y, e in zip(data.FORE.Y[:T], err["track"][s][:T]))))
        else:
            errors = list(zip(err["along"][s][:], err["cross"][s][:]))
            scen_T[s] = np.array(list(map(lambda x: helper.transform_gis_random_landfall(data, t=x[0], xi=x[1]), list(enumerate(errors)))))
    return scen_T, scen_Cat

"""### Plot the sample paths created"""

def plot_scenarios(state, sample_paths):
    gdf_list = []
    for s, sample in sample_paths.items():
        gdf = gpd.GeoDataFrame({'scen': [s], 'geometry': [LineString(sample)]})
        gdf_list.append(gdf)
    gdf = pd.concat(gdf_list)
    fig, ax = plt.subplots()
    state_gdf = gpd.GeoDataFrame({'geometry': [state]})
    CRS = "EPSG:4326"
    state_gdf.crs = CRS
    state_gdf = state_gdf.to_crs(epsg=int(CRS[-4:]))
    state_gdf.plot(ax=ax)
    gdf.plot(ax=ax)
    # ctx.add_basemap(ax, zoom=7, crs=CRS)
    plt.show()

def sample_paths_to_demand(data, args, err, S, tprime=0, FE_CLASS=None):
    """Given the coimplate sample paths of error realizations, compute the respective demand.
    Demand is a function of hurricane track and intensity scenarios rather than errors.
    Hurricane scenarios are created in intermediate step using the function (create_hurricane_scen).

    data : class
        Class of initial inputs including numeric inputs and GIS data.
    args : dict
        Arguments of the particular instance for which demand is being computed.
    err : dict = {"err_type": {"scen_index": list(err for all periods)}}
        Contains forecast error data for all error types, for a
        number of scenarions and all periods
    S : int
        Number of sample paths to create. err dict may contain a huge number of samples but
        we only create demand for S samples.
    tprime : int
        Starting period. Useful when creating in-samples for SLAM for rolling horizon at t' > 0.
    FE_CLASS : class
        Python class that contains all data related to forecast errors.
        # TODO: merge FE_CLASS to data as have a single input.
    """
    # Initials
    T = data.T
    ts = [T - 1] * S
    # Expected landfall time at every period
    Ets = np.array([[T-1 for t in range(T)] for s in range(S)])
    DF = [{} for _ in range(S)]
    scen_T, scen_Cat = create_hurricane_scen(data, args, err)
    # plot_scenarios(data.SC, scen_T)
    for s in range(S):
        if tprime == 0:
            DF[s][tprime] = [0.0] * data.I
        for t in range(tprime+1, T):
            if args['landfall'] == 'd':
                # Map hurricane track and category to the demand
                DF[s][t] = demand_deterministic_landfall(data=data, hurr_pos=scen_T[s][t, :], hurr_cat=scen_Cat[s, t], t=t)
            else:  # random landfall
                # Check if t is the landfall time
                if data.US.intersects(LineString(scen_T[s][t-1:t+1])) or t == T-1:
                    ts[s] = t
                    Ets[s, t] = t
                # Compute Expected ts if t is not t_s
                else:
                    along_err, cross_err = list(createErrorSamples(data=FE_CLASS, err=ERR, S=data.S, t=t, oos_err_at_t=err[ERR][s][t])
                                                for ERR in ["along", "cross"])
                    if args['fix_along_err'] is True:
                        for scen in range(along_err.shape[0]):
                            along_err[scen, :] = 0.0
                    ts_temp = [T-1] * data.S
                    for s_ in range(data.S):
                        errors_temp = list(zip(along_err[s_, :], cross_err[s_, :]))
                        scen_T_temp = list(map(lambda p: helper.transform_gis_random_landfall(data, p[0], p[1]), enumerate(errors_temp)))
                        for t_, P in enumerate(scen_T_temp):
                            if t_ > t:
                                if data.US.contains(Point(P)):
                                    ts_temp[s_] = t_
                                    break
                    Ets[s, t] = np.ceil(np.average(ts_temp))
                # Compute demand
                DF[s][t] = demand_random_landfall(data=data, hurr_pos=scen_T[s][t, :], hurr_cat=scen_Cat[s, t], t=t, ts=Ets[s, t])
                print("s", s, "t", t, "Ets", Ets[s, t], "DF", DF[s][t], "WS", scen_Cat[s, t])
                if ts[s] == t:
                    break
    if args['landfall'] == 'd':
        ts = [T-1] * S
    scen_Cat = scen_Cat.tolist()
    scen_Cat = list(scen_Cat[s][:ts[s]+1] for s in range(S))
    return DF, scen_T, scen_Cat, ts, Ets

"""# Create demand for random landfall
In the following class 'Demand', we take python classes for forecast_error and parameters and compute demand data. This class has methods to create out-sample demand, in-sample scenarios for static 2SSP model, demand of Markov Chain (MC) model for MSSP, transition probabilities between MC states, test-samples generated from MC model, and more.
"""

class Demand:
    """This class contains methods to create all kind of demand data including out-of-sample demands,
    Markov chain demand, Markov chain transition probability and more. This class takes inputs of args of
    the instance, forecast error class that contains all information about forecast error under all
    scenarios and markov chain states, and logistics network information with other intial numeric inputs.
    """

    def __init__(self, args, forecast_err_class, params_class):
        self.args = args
        self.FE = forecast_err_class
        self.PARAM = params_class
        self.ST_cross = 5
        self.ST_along = 5
        self.ST_track = self.args['ST_track']

    def discretize_intensity_scen(self, mode='r'):
        """Discretization of intensity error samples.

        mode : char
            mode = 'r' to read samples and probability matrix.
            mode = 'w' to write (create) samples and probability matrix.
        """
        path = self.PARAM.DIR[1] + "p_int.csv"
        if not os.path.exists(path):
            mode = 'w'
        if mode == 'w':
            T = self.PARAM.T
            samples = []
            intensity_scen = []
            for s in range(self.args['n_oos']):
                intensity_scen.append(self.PARAM.FORE.WS[:T] + self.FE.oos_err["intensity"][s][:T])
                samples.append(list(map(lambda i: helper.get_hurr_cat(i), intensity_scen[s])))
            State_space = [0, 1, 2, 3, 4, 5]
            S = len(State_space)
            COUNT = np.zeros([S, S]).astype('int')
            P = np.zeros([S, S])
            N = len(samples)
            for s in range(N):
                ts = len(samples[s])
                for t in range(ts - 1):
                    i, j = samples[s][t], samples[s][t + 1]
                    COUNT[i, j] += 1
            for st in range(S):
                if sum(COUNT[st, :]) > 0:
                    P[st, :] = COUNT[st, :] / sum(COUNT[st, :])
            pd.DataFrame(P).round(3).to_csv(self.PARAM.DIR[1] + "p_int.csv", header=False, index=False)
        setattr(self, 'pi_int', pd.read_csv(path, header=None).to_numpy())

    def characterize_absorbing_state(self, point):
        """Mark given Shapely Point geometry (long, lat) as 'absorbing' or 'transient' using US map.
        To convert a (x, y) coordinate to Shapely point, use: from shapely.geometry import Point; Point([x, y]).
        """
        if self.PARAM.US.contains(point):  # Point insde the US map
            return True
        else:
            for poly in self.PARAM.US.geoms:
                nearest_point = poly.exterior.interpolate(poly.exterior.project(point))
                # Convert long/lat difference to miles
                dx = abs(nearest_point.x - point.x)*self.PARAM.Xmiles
                dy = abs(nearest_point.y - point.y)*self.PARAM.Ymiles
                # Absorbing within 1 miles of US boarder
                if math.sqrt(dx**2 + dy**2) <= self.PARAM.landfall_tol:
                    return True
                else:
                    return False

    def create_sample_path_demand(self, mode, kind, S):
        """Generate out-sample or in-sample demand scenarios at t=0.

        mode : 'r' or 'w'
            'r' to read or 'w' to write (create).
        kind : 'oos' or 'insample'
            'oos' for out-of-sample or 'insample'.
        S : int
            Number of samples to generate.
        """
        if self.args['hurricane'] == 'Florence' and self.args['landfall'] == 'r':
            path = self.PARAM.DIR[2] + '{}_{}_fix_along_{}.json'.format(
                kind, self.args['landfall'], self.args['fix_along_err'],
            )
        else:
            path = self.PARAM.DIR[2] + '{}_{}.json'.format(kind, self.args['landfall'])
        names = ['ts', 'cat_scen', 'demand']  #, 'expected_ts']
        if mode == 'r' and not os.path.exists(path):
            mode = 'w'
        if mode == "w":
            err = self.FE.oos_err if kind == "oos" else {err_name: createErrorSamples(data=self.FE, err=err_name, S=S, t=0, oos_err_at_t=0)
                                                         for err_name in self.FE.err_type}
            if self.args['fix_along_err'] is True:
                err['along'] = np.zeros(np.array(err['along']).shape)
            DF, track_scen, cat_scen, ts, Ets = sample_paths_to_demand(self.PARAM, self.args, err, S, 0, self.FE)
            ts, cat_scen, DF = dict(enumerate(ts)), dict(enumerate(cat_scen)), dict(enumerate(DF))
            export = {name: val for name, val in zip(names, [ts, cat_scen, DF])}
            with open(path, 'w') as file:
                json.dump(export, file, indent=4)
        with open(path, 'r') as file:
            _dict = json.load(file)
            _dict = helper.json_import_conversion(_dict)
            for name in names:
                setattr(self, name + '_' + kind, _dict[name])

    def markov_chain_state_space(self):
        err_a = self.FE.err_tree['along'][self.ST_along]  # list of discrete 'along' errors
        if self.args['fix_along_err'] is True:
            err_a = {a: np.zeros(len(lst)) for a, lst in err_a.items()}
        err_c = self.FE.err_tree['cross'][self.ST_cross]  # list of discrete 'cross' errors
        if self.args['landfall'] == 'd':
            err_t = self.FE.err_tree['track'][self.ST_track]  # list of discrete 'track' errors
        cat_t0 = helper.get_hurr_cat(float(self.PARAM.FORE['WS'][0]))
        cat = [[cat_t0]] + [list(range(6))]*(self.PARAM.T-1)
        # Construct State Space for MC tree (combine three errors)
        state_space = [[] for t in range(self.PARAM.T)]
        for t in range(self.PARAM.T):
            if self.args['landfall'] == 'd':
                state_space[t] = list(itertools.product(err_t[t], cat[t]))
            elif self.args['fix_along_err'] is True:
                state_space[t] = list(itertools.product(err_c[t], cat[t]))
            else:
                state_space[t] = list(itertools.product(err_a[t], err_c[t], cat[t]))
        ST = [len(state_space[t]) for t in range(self.PARAM.T)]
        return state_space, ST

    def markov_chain_transition_prob(self, state_space):
        """Compute the transition probability and expected landfall period for all states at all periods
        using discretized forecast errors, hurricane categories, individual transition probability matrices,
        and Point forecast.
        """
        self.discretize_intensity_scen(mode='r')
        Pi = {t: {s: {s_: 0.0 for s_ in state_space[t+1]} for s in state_space[t]} for t in range(self.PARAM.T-1)}
        Absorb = {t: {state: True for state in state_space[t]} for t in range(self.PARAM.T)}
        Ets = {t: {state: self.PARAM.T-1 for state in state_space[t]} for t in range(self.PARAM.T)}
        for t in reversed(range(self.PARAM.T - 1)):
            for s1 in state_space[t]:
                if self.args['landfall'] == 'd':
                    absorb = False
                elif self.args['fix_along_err'] is True:
                    scen = helper.transform_gis_random_landfall(self.PARAM, t, [0, s1[0]])
                    absorb = self.characterize_absorbing_state(Point(scen))
                else:
                    scen = helper.transform_gis_random_landfall(self.PARAM, t, [s1[0], s1[1]])
                    absorb = self.characterize_absorbing_state(Point(scen))
                if absorb:
                    P = {s2: 0.0 for s2 in state_space[t + 1]}
                    ts = t
                else:
                    P = {}
                    for s2 in state_space[t + 1]:
                        if self.args['landfall'] == 'd':
                            prob_i = self.pi_int[s1[1]][s2[1]]
                            prob_t = self.FE.pi_tree["track"][self.ST_track][t][s1[0]][s2[0]]
                        elif self.args['fix_along_err'] is True:
                            prob_i = self.pi_int[s1[1]][s2[1]]
                            prob_t = self.FE.pi_tree["cross"][self.ST_cross][t][s1[0]][s2[0]]
                        else:
                            prob_i = self.pi_int[s1[2]][s2[2]]
                            prob_a = self.FE.pi_tree["along"][self.ST_along][t][s1[0]][s2[0]]
                            prob_c = self.FE.pi_tree["cross"][self.ST_cross][t][s1[1]][s2[1]]
                            prob_t = prob_a * prob_c
                        P[s2] = round(prob_t * prob_i, 3)
                    P = {s2: pi2 / sum(P.values()) for s2, pi2 in P.items()}
                    ts = sum(P[s2] * Ets[t+1][s2] for s2 in state_space[t+1])
                Pi[t][s1] = P
                Ets[t][s1] = ts
                Absorb[t][s1] = absorb
        Ets = {t: {state: round(val) for state, val in Ets[t].items()} for t in range(self.PARAM.T)}
        return Pi, Ets, Absorb

    def markov_chain_demand(self, state_space, ts_mssp):
        """Map the errors into demand for each state.
        """
        demand_mssp = {t: {state: list() for state in state_space[t]} for t in range(self.PARAM.T)}
        for t in range(self.PARAM.T):
            for state in state_space[t]:
                if self.args['landfall'] == 'd':
                    pos = [self.PARAM.FORE.X[t] + self.PARAM.X_rotate * state[0]/self.PARAM.Xmiles,
                           self.PARAM.FORE.Y[t] + self.PARAM.Y_rotate * state[0]/self.PARAM.Ymiles]
                    hurr_cat = state[1]
                    demand_mssp[t][state] = demand_deterministic_landfall(data=self.PARAM, hurr_pos=pos, hurr_cat=hurr_cat, t=t)
                else:
                    pos = helper.transform_gis_random_landfall(
                        self.PARAM, t, ([0, state[0]] if self.args['fix_along_err'] is True else state[:2])
                    )
                    hurr_cat = state[1] if self.args['fix_along_err'] is True else state[2]
                    demand_mssp[t][state] = demand_random_landfall(data=self.PARAM, hurr_pos=pos, hurr_cat=hurr_cat, t=t, ts=ts_mssp[t][state])
        return demand_mssp

    def samples_from_markov_chain(self, S):
        """Create or read test samples generated from MC Tree.

        mode='r' to read existing data;
        mode='w' to write (create) new samples
        """
        self.T = self.PARAM.T
        self.HURR = self.args['hurricane']
        sample_dict = {}
        for s in range(S):
            sample_dict[s] = [self.state_space[0][0]]
            for t in range(1, self.PARAM.T):
                s_ = sample_dict[s][-1]
                indices = range(len(self.state_space[t]))
                i = np.random.choice(indices, p=list(self.pi_mssp[t-1][s_].values()))
                state_sampled = self.state_space[t][i]
                sample_dict[s].append(state_sampled)
                if self.absorb_mssp[t][state_sampled]:
                    break
        return sample_dict

    def get_all_markov_chain_data(self, mode='r'):
        """Generate all data related to demand estimation of the Markov chain.
        """
        self.state_space, self.ST_MSSP = self.markov_chain_state_space()
        names = ['ts_mssp', 'absorb_mssp', 'demand_mssp', 'pi_mssp', 'test_samples_from_tree',
                 'in_sample_from_tree_2ssp']
        if self.args['hurricane'] == 'Florence' and self.args['landfall'] == 'r':
            paths = list(self.PARAM.DIR[2] + '{}_{}_fix_along_{}.json'.format(
                name, self.args['landfall'], self.args['fix_along_err'],
                ) for name in names)
            if self.args['fix_along_err'] is True:
                for i, path in enumerate(paths):
                    name = path.split('.')[0]
                    paths[i] = name + '_ST_cross_{}'.format(self.ST_cross) + '.json'
        elif self.args['landfall'] == 'd':
            paths = list(self.PARAM.DIR[2] + '{}_{}_ST_track_{}.json'.format(
                name, self.args['landfall'], self.ST_track,
                ) for name in names)
        else:
            paths = list(self.PARAM.DIR[2] + '{}_{}.json'.format(
                name, self.args['landfall'],
                ) for name in names)
        if mode == 'r' and max(list(map(lambda p: os.path.exists(p), paths))) is False:
            mode = 'w'
        if mode == 'w':
            self.pi_mssp, self.ts_mssp, self.absorb_mssp = self.markov_chain_transition_prob(state_space=self.state_space)
            self.demand_mssp = self.markov_chain_demand(state_space=self.state_space, ts_mssp=self.ts_mssp)
            self.test_samples_from_tree = self.samples_from_markov_chain(S=self.args['n_oos'])
            self.in_sample_from_tree_2ssp = self.samples_from_markov_chain(S=self.PARAM.S)
        for name, path in zip(names, paths):
            with open(path, mode) as json_file:
                if mode == 'w':
                    file = getattr(self, name)
                    file = helper.json_export_conversion(file, name)
                    json.dump(file, json_file, indent=4)
                else:
                    file = json.load(json_file)
                    setattr(self, name, helper.json_import_conversion(file))

    def create_all_demand_data(self, markov_chain=True, p_int=False):
        """Create all data related to demand estimation.
        """
        self.create_sample_path_demand(mode='w', kind="oos", S=self.args['n_oos'])
        self.create_sample_path_demand(mode='w', kind="in_sample", S=self.PARAM.S)
        if p_int:
            self.discretize_intensity_scen(mode='w')
        if markov_chain:
            self.discretize_intensity_scen(mode='r')
            self.get_all_markov_chain_data(mode='w')

    def read_demand_data(self):
        """Read already created demand data (everything that is related to demand)
        """
        self.create_sample_path_demand(mode="read", kind="oos", S=None)
        self.create_sample_path_demand(mode="read", kind="in_sample", S=None)
        self.discretize_intensity_scen(mode='r')
        self.get_all_markov_chain_data(mode='r')

"""# Create data

## Read arguments of the instance
Note: 'commands.csv' has a list of commands and the respective values. One should choose the right values to create demand for that setting.
Not all commandas are used in demand estimation hence can be ignored. The args can be modified on the command_defaults.csv. Note: command library has to be reloaded to import the changes to command_defaults.csv. All args that are used in demand estimation are:
"""

if __name__ == '__main__':
    importlib.reload(commands)  # rea the updated command from .csv file.
    args = commands.get_commands()[-1]  # get the respective arguments of the instance configuration.
    args['fix_along_err'] = bool(args['fix_along_err'])
    pd.DataFrame(args, index=['value']).transpose().rename_axis('arg')

"""## Read numeric inputs, logistrics parameters, and forecast error data"""

if __name__ == '__main__':
    # Read initial numeric inputs and GIS data.
    NumericInputs = data_initial_inputs.Input(args=args)
    NumericInputs.get_all_inputs()
    numeric_args = vars(NumericInputs)
    numeric_args.pop('args')

    # Read logistics data which is needed to compute demand.
    LogisticsParams = data_initial_inputs.LogisticsParameters(args=args, input_args=numeric_args)
    LogisticsParams.logistics_network()

    # Read forecast error data
    FE = data_forecast_error_scenarios.ForecastError(args=numeric_args)
    FE.readFEData(oos_only=False, args=args)

"""## Create demand data for the given configuration of instance in 'args'"""

if __name__ == '__main__':
    DE = Demand(args, FE, LogisticsParams)
    DE.create_all_demand_data()

"""## Read the created data"""

if __name__ == '__main__':
    DE = Demand(args, FE, LogisticsParams)
    DE.read_demand_data()

