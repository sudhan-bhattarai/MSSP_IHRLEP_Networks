import matplotlib.pyplot as plt
import geopandas as gpd
import numpy as np
from shapely.geometry import Point, LineString
import contextily as ctx
import itertools

import helper


class Plot:
    def __init__(self, data, args):
        self.data = data
        self.args = args
        sc = args["hurricane"] == "Florence"
        self.scen_args = {
            "color": "r",
            "markersize": 12,
            "color": "r",
            "linewidth": 0.5,
            }
        self.fig, self.ax = plt.subplots(figsize=(10, 5))
        # State map
        self.CRS = "EPSG:4326"
        state = gpd.GeoDataFrame({
                "geometry": [data.SC if sc else data.FL]
                })
        state.crs = self.CRS
        state = state.to_crs(
            epsg=int(self.CRS[-4:])
            )
        state.plot(
            ax=self.ax,
            edgecolor='black',
            alpha=0.4,
            cmap='OrRd',
            label="South Carolina" if sc else "Florida",
            )

    def plotUS(self):
        US = gpd.GeoDataFrame({'geometry': [self.data.US]})
        US.plot(ax=self.ax, edgecolor='black', alpha=0.4, cmap='OrRd',
                legend=False)

    def forecast(self, t=None):
        fore = self.data.FORE[["X", "Y"]].to_numpy().tolist()
        gdf_fore = gpd.GeoDataFrame({'geometry': [LineString(fore)]},
                                    crs=self.CRS)
        gdf_fore.plot(ax=self.ax, color="k", linewidth=2,
                      label="Track forecast")
        fore = [fore[t]] if isinstance(t, int) else fore
        gdf_point = gpd.GeoDataFrame({'geometry': [Point(f) for f in fore]},
                                     crs=self.CRS)
        gdf_point.plot(ax=self.ax, marker="x", color="k", markersize=20)

    def facility(self):
        scale = 50 if self.args['plot_opt'] in [2, 3] else 200
        gdf = []
        fac = ["DP", "SP", "DC"]
        labels = ["DP", "SP", "MDC"]
        colors = ["m", "b", "g"]
        markers = ["H", "^", "s"]
        size = [self.data.DP_POP / max(self.data.DP_POP) * scale,
                self.data.q_J / max(self.data.q_J) * scale,
                [scale]]
        for f in fac:
            gdf.append(gpd.GeoDataFrame({
                "geometry": [Point(x) for x in getattr(self.data, f"{f}_GIS")]
                }, crs=self.CRS))
        for i, df in enumerate(gdf):
            df.plot(ax=self.ax, label=labels[i], color=colors[i],
                    marker=markers[i], markersize=size[i])

    def sp_activation(self, model='MSSP'):
        import json
        if model == '2SSP':
            name = '2ssp_bc.json'
        else:
            name = 'sddp_bb.json'
        with open(self.data.DIR[4] + name, 'r') as file:
            result = json.load(file)
            if model == '2SSP':
                zJ = list(zj_t[0] for zj_t in result['zJ'])
            else:
                zJ = result['zJ']
        alpha = []
        for z in zJ:
            alpha.append(1 if z > 0 else 0.1)
        fac = ["DP", "SP", "DC"]
        labels = ["DP", "SP", "MDC"]
        colors = ["m", "b", "g"]
        markers = ["H", "^", "s"]
        size = [self.data.DP_POP / max(self.data.DP_POP)*100,
                self.data.q_J / max(self.data.q_J)*100,
                [100]]
        for i, f in enumerate(fac):
            if f != 'SP':
                df = gpd.GeoDataFrame({
                    "geometry": list(Point(x) for x in getattr(
                        self.data, "{}_GIS".format(f)
                        ))
                    }, crs=self.CRS)
                df.plot(ax=self.ax, label=labels[i], color=colors[i],
                        marker=markers[i], markersize=size[i])
            else:
                SP_GIS = self.data.SP_GIS.tolist()
                sp1 = list(
                    gis for j, gis in enumerate(SP_GIS)
                    if zJ[j] > 0
                    )
                sp0 = [gis for gis in SP_GIS if gis not in sp1]
                size1 = [self.data.q_J[j] / max(self.data.q_J)*200
                         for j in range(self.data.J) if zJ[j] > 0]
                size0 = [self.data.q_J[j] / max(self.data.q_J)*200
                         for j in range(self.data.J) if zJ[j] == 0]
                sp1_df = gpd.GeoDataFrame({
                    "geometry": list(map(lambda x: Point(x), sp1))
                    }, crs=self.CRS)
                sp0_df = gpd.GeoDataFrame({
                    "geometry": list(map(lambda x: Point(x), sp0))
                    }, crs=self.CRS)
                sp1_df.plot(ax=self.ax, label='Active SP',
                            color=colors[i],
                            marker=markers[i], markersize=size1)
                sp0_df.plot(ax=self.ax, label='Inactive SP',
                            color='white', edgecolor=colors[i],
                            marker=markers[i], markersize=size0)
        plt.title(f'SP activation decision of {model} model')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ctx.add_basemap(self.ax, zoom=9, crs=self.CRS)
        plt.savefig(self.data.DIR[4] +
                    f'sp_activation_{model}.PNG',
                    dpi=200,
                    bbox_inches='tight')

    def inv_level(self, model='MSSP'):
        import json
        import pandas as pd
        with open(self.data.DIR[4] + '2ssp_bc.json', 'r') as file:
            result = json.load(file)
            zJ = list(zj_t[0] for zj_t in result['zJ'])
        alpha = []
        for z in zJ:
            alpha.append(1 if z > 0 else 0.1)
        df = pd.read_excel(
            self.data.DIR[4] + 'disaster_logistic_workshop_results_2024.xlsx'
            )
        fac = ["DP", "SP", "DC"]
        labels = ["DP", "SP", "MDC"]
        colors = ["m", "b", "g"]
        markers = ["H", "^", "s"]
        size = [self.data.DP_POP / 164881*300,
                self.data.q_J / max(self.data.q_J)*100,
                [100]]
        for i, f in enumerate(fac):
            if f != 'SP':
                df = gpd.GeoDataFrame({
                    "geometry": list(Point(x) for x in getattr(
                        self.data, "{}_GIS".format(f)
                        ))
                    }, crs=self.CRS)
                df.plot(
                    ax=self.ax,
                    label=labels[i],
                    color=colors[i],
                    marker=markers[i],
                    markersize=size[i],
                    )
            else:
                SP_GIS = self.data.SP_GIS.tolist()
                sp1 = list(
                    gis for j, gis in enumerate(SP_GIS)
                    if zJ[j] > 0
                    )
                sp0 = [gis for gis in SP_GIS if gis not in sp1]
                size1 = [self.data.q_J[j] / max(self.data.q_J)*200
                         for j in range(self.data.J) if zJ[j] > 0]
                size0 = [self.data.q_J[j] / max(self.data.q_J)*200
                         for j in range(self.data.J) if zJ[j] == 0]
                sp1_df = gpd.GeoDataFrame({
                    "geometry": list(map(lambda x: Point(x), sp1))
                    }, crs=self.CRS)
                sp0_df = gpd.GeoDataFrame({
                    "geometry": list(map(lambda x: Point(x), sp0))
                    }, crs=self.CRS)
                sp1_df.plot(ax=self.ax, label='Active SP',
                            color=colors[i],
                            marker=markers[i], markersize=size1)
                sp0_df.plot(ax=self.ax, label='Inactive SP',
                            color='white', edgecolor=colors[i],
                            marker=markers[i], markersize=size0)
        plt.title(f'SP activation decision of {model} model')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ctx.add_basemap(self.ax, zoom=9, crs=self.CRS)
        plt.savefig(self.data.DIR[4] +
                    f'sp_activation_{model}.PNG',
                    dpi=200,
                    bbox_inches='tight')

    def scen_random_landfall(self):
        scen = []
        transient_state = []
        absorbing_state = []
        for s in self.args["scen_lst"]:
            errors = list(
                zip(self.data.oos_err["along"][s][:self.data.ts_oos[s] + 1],
                    self.data.oos_err["cross"][s][:self.data.ts_oos[s] + 1])
                )
            sample_path = list(map(
                lambda x: helper.transform_gis_random_landfall(
                    self.data, t=x[0], xi=x[1]
                    ), list(enumerate(errors))
                    ))
            for t, state in enumerate(sample_path):
                if t == self.data.ts_oos[s]:
                    intersection = self.data.US.intersection(
                        LineString(sample_path[-2:])
                        )
                    if intersection.is_empty:
                        STATE = state
                    else:
                        if isinstance(intersection, Point):
                            STATE = [intersection.x, intersection.y]
                        elif intersection.type in [
                                'LineString', 'MultiLineString'
                                ]:
                            if intersection.type == 'MultiLineString':
                                line_coords = []
                                # Iterate through each component LineString
                                # and append its coordinates to the list
                                for ls in intersection.geoms:
                                    line_coords.extend(list(ls.coords))
                                # Create a new LineString using
                                #  the concatenated coordinates
                                intersection = LineString(line_coords)
                            intersection_points = list(intersection.coords)
                            nearest = np.argmin(
                                [helper.distMiles(point, sample_path[-2])
                                 for point in intersection_points
                                 ])
                            STATE = intersection_points[nearest]
                        else:
                            print("error")
                            exit(0)
                    absorbing_state.append(Point(STATE))
                else:
                    transient_state.append(Point(state))
            scen.append(LineString(sample_path[:-1] + [STATE]))
        gdf_scen = gpd.GeoDataFrame({"s": self.args["scen_lst"],
                                     "geometry": scen}, crs=self.CRS)
        gdf_states_t = gpd.GeoDataFrame({"geometry": transient_state},
                                        crs=self.CRS)
        gdf_states_a = gpd.GeoDataFrame({"geometry": absorbing_state},
                                        crs=self.CRS)
        gdf_scen.plot(ax=self.ax, **self.scen_args, label="Track scenarios")
        gdf_states_t.plot(ax=self.ax, **self.scen_args, facecolors="none",
                          label="Transient states")
        gdf_states_a.plot(
            ax=self.ax, **self.scen_args, label="Absorbing states"
            )

    def scen_deterministic_landfall(self):
        scen = []
        T = self.data.T
        for s in self.args["scen_lst"]:
            err = self.data.oos_err['track'][s][:self.data.ts_oos[s] + 1]
            if abs(err[-1]) > 200:
                continue
            scen.append(np.column_stack((
                list(x + self.data.X_rotate * e/self.data.Xmiles for x, e in
                     zip(self.data.FORE.X[:T], err)
                     ),
                list(y + self.data.Y_rotate * e/self.data.Ymiles for y, e in
                     zip(self.data.FORE.Y[:T], err)
                     )
                ))
            )
        gdf_scen = gpd.GeoDataFrame(
            {'s': range(len(scen)),
             'geometry': list(map(lambda pos: LineString(pos), scen)),
             }, crs=self.CRS
            )
        gdf_scen.crs = self.CRS
        gdf_scen.plot(ax=self.ax, **self.scen_args, label="Track scenarios")

    def mc_deterministic_landfall(self):
        T = self.data.T
        absorb = []
        transient = []
        for t in range(T):
            err = self.data.err_tree['track'][t]
            err = list(e for e in err if abs(e) < 200)
            if len(err) >= 4:
                err = [e for i, e in enumerate(err) if i % 2 == 0]
            X = list(self.data.FORE.X[t] +
                     self.data.X_rotate * e/self.data.Xmiles for e in
                     err)
            Y = list(self.data.FORE.Y[t] +
                     self.data.Y_rotate * e/self.data.Ymiles for e in
                     err)
            ss = list(Point(p) for p in zip(X, Y))
            if t < T - 1:
                transient = transient + ss
            else:
                absorb = ss
        gdf_0 = gpd.GeoDataFrame({"geometry": absorb}, crs=self.CRS)
        gdf_1 = gpd.GeoDataFrame({"geometry": transient}, crs=self.CRS)
        gdf_0.plot(ax=self.ax,
                   marker="o",
                   label=f"Absorbing states",
                   edgecolor="r",
                   facecolor="r")
        gdf_1.plot(ax=self.ax,
                   marker="o",
                   label=f"Transient states",
                   edgecolor="r",
                   facecolor="none")

    def mc_random_landfall(self):
        t = self.args["t"]
        errors = itertools.product(self.data.err_tree["along"][t],
                                   self.data.err_tree["cross"][t])
        scen = list(map(lambda x: helper.transform_gis_random_landfall(
            self.data, t=t, xi=x
            ), errors))
        scen = list(map(lambda x: Point(x), scen))
        absorb = [s for s in scen if self.data.US.contains(s)]
        transient = list(filter(lambda s: s not in absorb, scen))
        gdf_0 = gpd.GeoDataFrame({"geometry": absorb}, crs=self.CRS)
        gdf_1 = gpd.GeoDataFrame({"geometry": transient}, crs=self.CRS)
        gdf_0.plot(ax=self.ax,
                   marker="o",
                   label=f"Absorbing states at t={t}",
                   edgecolor="r",
                   facecolor="r")
        gdf_1.plot(ax=self.ax,
                   marker="o",
                   label=f"Transient states at t={t}",
                   edgecolor="r",
                   facecolor="none")

    def plot(self):
        s = 1
        num_scen = 100
        plot_args = {
            'usa': 0,
            't': int(self.data.ts_oos[s]),
            'scen_lst': range(num_scen),
            }
        self.args.update(plot_args)
        if self.args["usa"] not in [2]:
            self.plotUS()

        if self.args['plot_opt'] == 1:
            self.facility()

        if self.args['plot_opt'] not in [1, 4]:
            self.forecast(
                t=self.args['t'] if self.args['plot_opt'] == 3 and
                self.args['landfall'] == 'r' else None
                )
            if self.args['landfall'] == 'd':
                self.scen_deterministic_landfall()
                if self.args['plot_opt'] == 3:
                    self.mc_deterministic_landfall()
            else:
                self.scen_random_landfall()
                if self.args['plot_opt'] == 3:
                    self.mc_random_landfall()
        if self.args['landfall'] == 'r':
            self.ax.legend(loc=1, fontsize=8)
            self.ax.legend(
                loc='center left',
                fontsize=14,
                bbox_to_anchor=(1, 0.5),
                )
        else:
            self.ax.legend(loc=1)
        ctx.add_basemap(self.ax, zoom=9, crs=self.CRS)
        if self.args['plot_opt'] in [1, 2]:
            path = self.data.DIR[1]
        else:
            path = self.data.DIR[2]
        plt.savefig(
            path +
            "{}_plot_option_{}_{}.PNG".format(
                self.args["hurricane"],
                self.args['plot_opt'],
                self.args['landfall']
                ),
            dpi=300,
            bbox_inches="tight",
            )
