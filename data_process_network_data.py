# Libraries
from geopy.geocoders import Nominatim
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
import math
from sklearn.cluster import KMeans
import numpy as np
from shapely.ops import unary_union
from shapely.geometry import Point, LineString, Polygon
import pgeocode
import csv
# Other modules of the project
import helper


query_GIS = pgeocode.Nominatim('us').query_postal_code
CRS = "EPSG:4326"
US = gpd.read_file(r'Data/us_GIS.json').query(
    'NAME not in ["Hawaii", "Alaska", "Puerto Rico"]'
    )


def florenceDPs():
    from geopy.exc import GeocoderTimedOut
    geolocator = Nominatim(user_agent="my_geocoder")
    geolocator.timeout = 10  # Set a longer timeout (in seconds)
    gdf = gpd.read_file("Data/Florence/geoms.geojson")
    gdf = gdf.set_index("object")
    dt_GIS = gdf.squeeze().to_dict()
    all_ZIPs = pd.read_csv("Data/Florence/SC_ZIP.csv", index_col=0)
    dt_float = pd.read_csv("Data/User Input/float.csv", index_col=0)
    dt_float = dt_float.squeeze().to_dict()
    Y = []
    X = []
    DP_ZIP = []
    for ZIP in all_ZIPs["Zip Code"].tolist():
        try:
            location = geolocator.geocode(
                {"postalcode": ZIP, "country": "United States"},
                exactly_one=True,
            )
        except GeocoderTimedOut:
            continue
        x, y = location.longitude, location.latitude
        projected_point = dt_GIS["study_region"].interpolate(
            dt_GIS["study_region"].project(Point([x, y]))
            )
        x_proj, y_proj = projected_point.x, projected_point.y
        dx = abs((x - x_proj) * dt_float["Xmiles"])
        dy = abs((y - y_proj) * dt_float["Ymiles"])
        dist = math.sqrt(dx ** 2 + dy ** 2)
        if dist <= dt_float["DP_DIST_TOL"]:
            DP_ZIP.append(ZIP)
            X.append(x_proj)
            Y.append(y_proj)
    df = all_ZIPs.loc[all_ZIPs["Zip Code"].isin(DP_ZIP)]
    df["X"] = X
    df["Y"] = Y
    df["Pop11%"] = list(map(lambda a: math.ceil(a), df["Population"] * 0.11))
    df["Pop5%"] = list(map(lambda a: math.ceil(a), df["Population"] * 0.05))
    df = df.reset_index()
    df.rename_axis("DP").to_csv(
        r"Data/Florence/DP_all_ZIPs.csv", header=True, index=True
        )
    return df


def clusterFlorenceCenters(data):
    dt_SP = pd.read_csv("Data/Florence/SC_counties.csv", index_col=0)
    dt_DP = pd.read_csv("Data/Florence/DP_all_ZIPs.csv", index_col=0)

    def cluster(DF, N, names):
        POS = DF[["X", "Y"]].to_numpy().astype("float")
        kmeans = KMeans(n_clusters=N, random_state=0).fit(POS)
        DF[names[0]] = kmeans.labels_
        df = pd.DataFrame(DF.groupby(names[0])[names[1]].sum())
        df[["X", "Y"]] = kmeans.cluster_centers_
        return df
    cluster(dt_DP, data.I, ["I", "Pop5%"]).to_csv(data.DIR[2] + "DP.csv")
    cluster(dt_SP, data.J, ["J", "CAPACITY"]).to_csv(data.DIR[2] + "SP.csv")


def clusterIanCenters(DIR, J=1):
    FL = US[(US.NAME == "Florida")]
    us = unary_union(US["geometry"])
    us_ext_pts = []
    for poly in us.geoms:
        ext_pt = list(poly.exterior.coords)
        us_ext_pts.extend(ext_pt)
    dt = pd.read_csv(DIR + "FL_counties.csv", sep=",")
    REGIONS = dt["Region"].unique()
    df = pd.DataFrame()
    # Get SPs location
    for region in REGIONS:
        DF_TEMP = dt.loc[dt["Region"] == region]
        DEMAND_TEMP = DF_TEMP["Demand"].sum()
        CAPACITY_TEMP = DF_TEMP["Shelter capacity"].sum()
        kmeans = KMeans(n_clusters=J, random_state=0,
                        ).fit(DF_TEMP[["X", "Y"]].to_numpy())
        X, Y = kmeans.cluster_centers_[0]
        df[region] = [DEMAND_TEMP, CAPACITY_TEMP, X, Y]
    df = df.set_index(pd.Index(["Demand", "CAPACITY", "X", "Y"]))
    df = df.transpose().rename_axis("Region")
    df[["X", "Y", "CAPACITY"]].to_csv(DIR + "SP.csv", index=True, header=True)
    POINTS = [Point([x, y]) for x, y in zip(df["X"], df["Y"])]
    ax = FL.plot()
    gpd.GeoDataFrame({"geometry": POINTS}).plot(
        ax=ax, markersize=2, color="black", marker="s",
        )
    # Get DPs location
    dist_mat = np.array([[helper.distMiles(point.coords[0], ext)
                          for ext in us_ext_pts]
                         for point in POINTS])
    DP = []
    P, E = dist_mat.shape
    for p in range(P):
        ext_lst_p = [(e, dist_mat[p, e]) for e in range(E)
                     if dist_mat[p, e] >= 50]
        sort = sorted(ext_lst_p, key=lambda x: x[1])
        ext_lst_p = [ext[0] for ext in sort]
        for e in ext_lst_p:
            if min(dist_mat[:, e]) >= 50:
                DP.append(us_ext_pts[e])
                print(p, us_ext_pts[e])
                break
    df[["X", "Y"]] = DP
    df[["X", "Y", "Demand"]].to_csv(
        DIR + "DP.csv", index=True, header=True
        )


def getIanForecastSlopes():
    slope_lst = [0.0]  # Slope at t=0 is not needed
    df = pd.read_csv("Data/Ian/ian_forecast.csv")
    for i in range(1, df.shape[0]):
        # 1 degree long = 54.6 miles; 1 degree lat = 69.0 miles
        dx = (df.loc[i, "Longitude"] - df.loc[i-1, "Longitude"])*54.6
        dy = (df.loc[i, "Latitude"] - df.loc[i-1, "Latitude"])*69.0
        slope = math.atan(dy/dx)
        slope_lst.append(slope)
    pd.DataFrame({"angle": slope_lst},
                 index=range(len(slope_lst))).rename_axis("t").to_csv(
                     "Data/Ian/forecast_slope_angles.csv",
                     index=True, header=True
                     )
    df.plot("Longitude", "Latitude", figsize=(2, 5))


def demand_zone_florence(args):
    GDF = gpd.GeoDataFrame
    all_geoms = GDF()
    coast_edge_ZIP = [29915, 29582]
    forecast_at_t4 = pd.read_csv(
        r"Data/Florence/Florence_forecast.csv", index_col=0
        ).to_numpy()[4, :2]
    # ZIP codes at the Southest and Northest region of SC coast
    SC = US[(US.NAME == "South Carolina")]
    DP_EDGE = list(map(lambda x: query_GIS(x)[
        ['longitude', 'latitude']
        ].to_list(), coast_edge_ZIP))
    # total Euclidean distance of the coastal span (demand region)
    COAST_DIST = math.sqrt(
        (args['Xmiles'] * (DP_EDGE[0][0] - DP_EDGE[1][0]))**2 +
        (args['Ymiles'] * (DP_EDGE[0][1] - DP_EDGE[1][1]))**2
        )
    # extend demand region until study region to South/North side
    # SR = Study Region
    study_region_south = np.array(DP_EDGE[0]) - args['y_max']/COAST_DIST*(
            np.array(DP_EDGE[1]) - np.array(DP_EDGE[0])
            )
    study_region_north = np.array(DP_EDGE[1]) + args['y_max']/COAST_DIST*(
            np.array(DP_EDGE[1]) - np.array(DP_EDGE[0])
            )
    study_region_corners_GIS = [list(study_region_south),
                                list(study_region_north)]
    # slope of the land in radians
    study_region_dist = math.sqrt(
        (args['Xmiles']*abs(study_region_north[0]-study_region_south[0]))**2 +
        (args['Ymiles']*abs(study_region_north[1]-study_region_south[1]))**2
        )
    land_slope = math.atan(
        args['Ymiles'] * (study_region_south[1] - study_region_north[1]) /
        (args['Xmiles'] * (study_region_south[0] - study_region_north[0]))
        )
    # shift coordinates in the tilted land
    # (projection factors of error value on the original coordinate system)
    x_shift_fact = math.cos(land_slope)
    y_shift_fact = math.sin(land_slope)
    study_line = LineString(np.array(study_region_corners_GIS))
    temp_project = study_line.interpolate(
        study_line.project(Point(forecast_at_t4))
        )
    x_max = math.sqrt(
        (args['Xmiles']*(forecast_at_t4[0] - temp_project.x))**2 +
        (args['Ymiles']*(forecast_at_t4[1] - temp_project.y))**2
        )
    demand_zone_v1 = [study_region_south[0]
                      + (x_max * y_shift_fact)/args['Xmiles'],
                      study_region_south[1]
                      - (x_max * x_shift_fact)/args['Ymiles']]
    demand_zone_v2 = [study_region_north[0] +
                      x_max * y_shift_fact/args['Xmiles'],
                      study_region_north[1] -
                      x_max * x_shift_fact/args['Ymiles']]
    demand_zone = Polygon([demand_zone_v1, demand_zone_v2,
                           study_region_north, study_region_south])
    all_geoms = GDF(
        dict(object='SC',
             geometry=SC['geometry'].to_list()[0].difference(demand_zone)
             ), index=[0], crs=CRS,
        )
    all_geoms = pd.concat(
        [all_geoms, GDF(dict(object='SC_original',
                             geometry=SC['geometry'].to_list()[0]),
                        index=[0], crs=CRS,)]
        ).reset_index(drop=True)
    all_geoms = pd.concat(
        [all_geoms, GDF(dict(object='study_region', geometry=study_line),
                        index=[0], crs=CRS,
                        )]).reset_index(drop=True)
    all_geoms = pd.concat(
        [all_geoms, GDF(dict(object='demand_zone', geometry=demand_zone),
                        index=[0], crs=CRS,
                        )]).reset_index(drop=True)
    path = r"Data/Florence/"
    # Export
    float_name = ['x_diff_fact', 'y_diff_fact',
                  'x_max', 'study_region_dist']
    float_vals = [x_shift_fact, y_shift_fact, x_max, study_region_dist]
    with open(path+'geom_params.csv', 'w', encoding='UTF8') as f:
        writer = csv.writer(f)
        writer.writerow(float_name)  # Write the header.
        writer.writerow(float_vals)  # Write the data.
    all_geoms.to_file(path+'geoms.geojson', driver='GeoJSON')
    return all_geoms, x_shift_fact, y_shift_fact, study_region_dist


def demand_zone_ian():
    path = r"Data/Ian/"
    us = US.to_crs(CRS)
    FL = unary_union(us[(us.NAME == "Florida")]["geometry"])
    us = unary_union(us["geometry"])
    gdf = gpd.GeoDataFrame(
        dict(object="US", geometry=[us]), index=[0], crs=CRS
        )
    gdf = pd.concat(
        [gdf, gpd.GeoDataFrame(
            {"object": "FL", "geometry": [FL]}, index=[0], crs=CRS
            )]
        ).reset_index(drop=True)
    gdf.to_file(path + 'geoms.geojson', driver='GeoJSON')
    return gdf