import os
import numpy as np
import pandas as pd
import geopandas as gpd
import pgeocode
from shapely.geometry import Point, LineString, Polygon
from shapely.ops import unary_union

import helper
import data_process_network_data as process_data
from data_demand_estimation import Demand
from data_forecast_error_scenarios import ForecastError

np.random.seed(seed=111)
query_GIS = pgeocode.Nominatim('us').query_postal_code
CRS = "EPSG:4326"
US = gpd.read_file(r'Data/us_GIS.json')
US = US[(US.NAME != "Hawaii") &
        (US.NAME != "Alaska") &
        (US.NAME != "Puerto Rico")]
GDF = gpd.GeoDataFrame


class Input:
    def __init__(self, args):
        path_numeric_input = r"Data/numeric_inputs.csv"
        path_prob_size = r"Data/problem_size_opt.csv"
        self.args = args

        """-------All user defined numeric inputs from .csv file.--------"""
        df_numeric_input = pd.read_csv(path_numeric_input, index_col=0)
        for i in df_numeric_input.index:
            val = df_numeric_input.loc[i, 'Value']
            if val % 1 == 0:
                setattr(self, i, int(val))
            else:
                setattr(self, i, float(val))
        # make changes to data according to args
        if self.args['landfall'] == 'r':
            self.T = self.T_max

        """-----------------Network size-----------------------------------"""
        ian = self.args['hurricane'] == 'Ian'
        create = self.args['task'] == 'create_data'
        if ian and create:
            setattr(self, 'I', 10)
            setattr(self, 'J', 10)
            setattr(self, 'K', 1)
        else:
            df_temp = pd.read_csv(path_prob_size, index_col=0)
            setattr(self, 'I',  # Number of demand points
                    int(df_temp.loc[self.args["instance"], "I"]))
            setattr(self, 'J',  # Number of shelters
                    int(df_temp.loc[self.args["instance"], "J"]))
            setattr(self, 'K', 1)  # MDC
        """---------------------------More inputs--------------------------"""
        self.p = [1/self.S for s in range(self.S)]  # probabilities of scen.

    def directories(self):
        """Path to specific folders.
        1: path to data common to both Hurricanes
        2: path to data specific to given Hurricane and instance
        3: path to forecast error data (common to both Hurricanes)
        4: path to results to given Hurricane and instance
        """
        self.DIR = {
            1: "Data/{}/".format(self.args['hurricane']),
            2: "Data/{}".format(self.args['hurricane']) + (
                "/Instances/" if self.args['hurricane'] == "Ian" else
                "/Instance_I{}_J{}/").format(self.I, self.J),
            # Path to particular instance of the given hurr and instance
            3: "Data/Forecast error/",
            4: "Results/{}/instance{}/ff{}_gf{}_pf{}_pcost{}/".format(
                self.args['hurricane'],
                self.args['instance'],
                self.args['ffact'],
                self.args['gfact'],
                self.args['pfact'],
                self.args['purchase_cost']
                ),
            }
        # --> added on 08-20-2024
        if self.args['landfall'] == 'd':
            self.DIR[4] = self.DIR[4] + \
                'ST_track_{}/'.format(self.args['ST_track'])
        # <--
        # Make paths if not already available
        for key, _dir in self.DIR.items():
            os.makedirs(_dir) if not os.path.isdir(_dir) else None

    def forecast(self):
        """Read the forecast of track and intensity for the hurricane."""
        self.FORE = pd.read_csv(
            self.DIR[1] + "{}_forecast.csv".format(self.args['hurricane']),
            index_col=0,
            ).iloc[:self.T]
        self.FORE = self.FORE.rename(
            columns={"Longitude": "X",
                     "Latitude": "Y",
                     "Intensity": "WS"}
            )

    def gis_data(self):
        us = US.to_crs(CRS)
        self.US = unary_union(us["geometry"])
        geom_file = self.DIR[1] + "geoms.geojson"
        if not os.path.isfile(geom_file):
            if self.args['hurrricane'] == 'Ian':
                process_data.demand_zone_ian()
            else:
                process_data.demand_zone_florence(args=vars(self))
        gdf = gpd.read_file(geom_file)
        for i, obj in enumerate(gdf["object"].values):
            setattr(self, obj, gdf["geometry"].values[i])
        if self.args['hurricane'] == "Florence":
            shift = pd.read_csv(self.DIR[1] + "geom_params.csv")
            X_rotate, Y_rotate, x_max, study_region_dist = shift.values[0]
            self.X_rotate = X_rotate
            self.Y_rotate = Y_rotate
            self.x_max = x_max
            self.study_region_dist = study_region_dist
            self.study_line = gdf.loc[
                gdf['object'] == 'study_region'
                ]['geometry'].values[0]
        else:
            geom_path = r"Data/Ian/forecast_slope_angles.csv"
            df = pd.read_csv(geom_path, index_col=0)
            self.fore_angle = df.to_numpy().ravel()

    def get_all_inputs(self):
        self.directories()
        self.forecast()
        self.gis_data()


class LogisticsParameters:
    def __init__(self, args, input_args):
        self.args = args
        for v, val in input_args.items():
            setattr(self, v, val)

    def logistics_network(self):
        # Cluster DPs and SPs if the facility data is not availale
        # for given instance
        if self.args['hurricane'] == "Florence":
            if not os.path.isfile(self.DIR[2]+"DP.csv"):
                process_data.clusterFlorenceCenters(self)
        else:
            if not os.path.isfile(self.DIR[1]+"DP.csv"):
                process_data.clusterIanCenters(self.DIR[1])
        path = self.DIR[1] if self.args['hurricane'] == "Ian" else self.DIR[2]
        SP = pd.read_csv(path + "SP.csv", index_col=0)
        DP = pd.read_csv(path + "DP.csv", index_col=0)
        MDC = pd.read_csv(self.DIR[1] + "{}_MDC.csv".format(
            self.args['hurricane']
            ), index_col=0)
        self.DP_GIS = DP[["X", "Y"]].to_numpy()
        self.SP_GIS = SP[["X", "Y"]].to_numpy()
        self.DC_GIS = MDC[["X", "Y"]].to_numpy()
        self.q_J = SP[["CAPACITY"]].to_numpy().ravel()
        if self.args['hurricane'] == "Florence":
            self.DP_POP = DP["Pop5%"].to_numpy().ravel()
        else:
            self.DP_POP = DP["Demand"].to_numpy().ravel()
        self.K = int(MDC.shape[0])
        self.INIT = {
            "lJ": [0] * self.J,
            "lK": [0] * self.K,
            "zJ": [0] * self.J,
            "eJ": [0] * self.J,
            "eI": self.DP_POP,
            }

    def transportation_cost(self, mode='read'):
        """
        Create/read transportation cost parameters.
        mode = "create" to create new data; "read": to read existing data
        """
        names = ["c_E_IJ", "c_R_JJ", "c_R_KJ"]
        for name in names:
            DIR = self.DIR[2] + f"{name}.csv"
            if mode == 'read' and not os.path.exists(DIR):
                mode = "create"
                break
        if mode == "create":
            c_E_IJ = self.alpha * helper.dist_matrix(
                self.DP_GIS, self.SP_GIS,
                )
            c_R_JJ = self.beta * helper.dist_matrix(
                self.SP_GIS, self.SP_GIS,
                )
            c_R_KJ = self.beta * helper.dist_matrix(
                self.DC_GIS, self.SP_GIS,
                )
            for key, val in locals().items():
                # avoid unnecessary variables
                if key not in ["self", "mode", "names", "DIR", "name"]:
                    df = pd.DataFrame(val)
                    df.to_csv(
                        self.DIR[2]+"{}.csv".format(key),
                        header=False, index=False
                        )
        # Read data
        for name in names:
            setattr(self, name, pd.read_csv(
                self.DIR[2] + f"{name}.csv", header=None,
                ).to_numpy())

    def amplify_params(self, factors):
        return dict(
            c_P_K=np.array(
                [factors["purchase_cost"] for k in range(self.K)]
                ),
            c_invR_K=np.array(
                [factors["purchase_cost"] * self.INVR_FACT
                 for k in range(self.K)]
                ),
            c_invR_J=np.array(
                [factors["purchase_cost"] * self.INVR_FACT
                 for j in range(self.J)]
                ),
            c_invE_J=np.array(
                [factors["purchase_cost"] * self.INVE_FACT
                 for j in range(self.J)]
                ),
            c_H_J=np.array(
                [- factors["purchase_cost"] * self.HFact
                 for j in range(self.J)]
                ),
            c_G_J=np.array(
                [np.average(self.c_R_KJ[:, j]) * factors['gfact']
                 + factors["purchase_cost"]
                 for j in range(self.J)]
                ),
            c_PE=np.array(
                [factors["pfact"] for i in range(self.I)]
                ),
            c_F_J=np.array(
                [self.q_J[j] * factors['ffact'] for j in range(self.J)]
                ),
            c_F_J_var=np.array(
                [self.q_J[j] * factors['ffact'] * self.FVarFact
                 for j in range(self.J)]
                ),
            )

    def tunable_params(self, mode='read'):
        # Create directory for the given keyword arguments
        names = ['c_F_J', 'c_F_J_var', 'c_P_K', 'c_invR_K', 'c_invR_J',
                 'c_invE_J', 'c_H_J', 'c_G_J', 'c_PE']
        if mode == 'read':
            if max(list(map(
                    lambda name: os.path.exists(self.DIR[2] + f'{name}.csv'),
                    names
                    ))) is False:
                mode = 'create'
        if mode == 'create':
            params = self.amplify_params(
                factors={'purchase_cost': 1.0,
                         'gfact': 1.0,
                         'pfact': 1.0,
                         'ffact': 1.0}
                )
            for key, item in params.items():
                df = pd.DataFrame(item)
                df.to_csv(self.DIR[2] + f"{key}.csv",
                          header=False,
                          index=False,)
        # Read data
        amplified_params = self.amplify_params(self.args)
        for param_name, param_val in amplified_params.items():
            setattr(self, param_name, param_val)

    def get_params(self, mode):
        self.logistics_network()
        self.transportation_cost(mode=mode)
        self.tunable_params(mode=mode)

