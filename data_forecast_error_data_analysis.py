# %% Libraries
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, norm
from scipy import stats
import json


# %% Initialize
path = r"Data/Forecast error/"
fsize = 30
err_grp = ["AC", "TI"]  # along-and-cross and track-and-intensity errors
err_types = ['along', 'cross', 'track', 'intensity']
hours = ["000", "012", "024", "036", "048", "072", "096", "120"]
plt.rcParams["font.family"] = "Times New Roman"


# %% Processing historical forecast error data
"""Source of data: https://www.nhc.noaa.gov/verification/verify7.shtml
Columns of interest:
- Date/Time : Date and time that forecast was made
- STMID : ID of the storm (ATCF cyclone identifier)
- F12 – F168: Equivalent sample size for use in statistical tests of
    significance, for the 12, 24, 36, 48, 72, 96, 120, 144, and 168 h
    forecasts
- Lat: Best track latitude of the cyclone at t=0 (degrees N).
- Lon: Best track longitude of the cyclone at t=0 (degrees W).
- WS: Best track maximum sustained wind of the cyclone at t=0 (kt).
- 000hT01: Track forecast error (n mi) for t=0 h
    (similarly 012hT1, 024hT1, ...)
- 000hI01: Intensity forecast error (kt) for t=0 h
    (similarly 012hI1, 024hI1, ...)
- 000hA01: Along (longitude, x-axis) track error
- 000C01: Across (latitude, y-axis) track error
"""

dfs = {}
for err in err_grp:
    dfs[err] = pd.read_csv(
        path+"1989-present_OFCL_v_BCD5_ind_ATL_{}_errors.txt".format(err),
        skiprows=7,
        header=0,
        delim_whitespace=True,
        )
# Column names
ofcl_err_cols = dict(
    AC={'along': list(map(lambda h: h + "hA01", hours)),
        'cross': list(map(lambda h: h + "hC01", hours))},
    TI={'track': list(map(lambda h: h + "hT01", hours)),
        'intensity': list(map(lambda h: h + "hI01", hours))}
    )


# %% Filter data for YEAR and drop NA


dfs_new = {}
for i, df in dfs.items():
    df[['Date', 'Time']] = df["Date/Time"].str.split("/", expand=True)
    df[['Day', 'Month', 'Year']] = df['Date'].str.split(
        '-',
        expand=True,
        ).astype('int')
    df.set_index(['STMID', 'Date/Time', 'Lat', 'Lon'], inplace=True)
    df = df.loc[df["Year"] >= 2018]  # last five years of data (2018-2022)
    for key, cols in ofcl_err_cols[i].items():
        df_temp = df[cols]
        df_temp = df_temp.where(df_temp != -9999.0, None)
        # unavailable data is marked by NHC as '-9999.0'
        df_temp.dropna(inplace=True)
        df_temp.drop(ofcl_err_cols[i][key][0], axis=1, inplace=True)
        if key != "intensity":
            df_temp = df_temp.where(df_temp != 0, None)
            df_temp.dropna(inplace=True)
        df_temp.insert(0, ofcl_err_cols[i][key][0], 0)
        dfs_new[key] = df_temp


# %% Coorelation


""" Coorelation between Track and intensity errors """
df = dfs["TI"][
    ofcl_err_cols["TI"]["track"] + ofcl_err_cols["TI"]["intensity"]
    ]
df = df.where(df != -9999.0, None)
df.dropna(how="any", inplace=True)
err_A = df[ofcl_err_cols["TI"]["track"]].values
err_C = df[ofcl_err_cols["TI"]["intensity"]].values
M, T = err_A.shape
fig, axs = plt.subplots(2, 3, figsize=(12, 8))
fig.suptitle("Historical track vs intensity forecast errors", fontsize=fsize)
fig.supxlabel('Track forecast errors', fontsize=fsize)
fig.supylabel('Inensity forecast errors', fontsize=fsize)
corr = []
for t in range(2, T):
    ax = axs[(t-2) // 3, (t-2) % 3]
    ax.scatter(err_A[:, t], err_C[:, t], marker='^', s=1, c="black")
    corr.append(pearsonr(err_A[:, t], err_C[:, t])[0])
    ax.set_title("t={} hours".format(hours[t]), fontsize=fsize-2)
    ax.tick_params(
        left=False,
        right=False,
        labelleft=False,
        labelbottom=False,
        bottom=False,
        )
plt.savefig(path+"Figures/correlation_TI.PNG", dpi=50)
plt.show()

"""Coorelation between Along and cross errors"""
df = dfs["AC"][ofcl_err_cols["AC"]["along"] + ofcl_err_cols["AC"]["cross"]]
df = df.where(df != -9999.0, None)
df.dropna(how="any", inplace=True)
err_A = df[ofcl_err_cols["AC"]["along"]].values
err_C = df[ofcl_err_cols["AC"]["cross"]].values

M, T = err_A.shape
fig, axs = plt.subplots(2, 3, figsize=(12, 8))
fig.suptitle(
    "Historical forecast errors in along- vs cross- directions",
    fontsize=fsize,
    )
fig.supxlabel('Along forecast errors', fontsize=fsize)
fig.supylabel('Cross forecast errors', fontsize=fsize)
corr = []
for t in range(2, T):
    ax = axs[(t-2) // 3, (t-2) % 3]
    ax.scatter(err_A[:, t], err_C[:, t], marker='^', s=1, c="black")
    corr.append(pearsonr(err_A[:, t], err_C[:, t])[0])
    ax.set_title("t={} hours".format(hours[t]), fontsize=fsize-2)
    ax.tick_params(
        left=False,
        right=False,
        labelleft=False,
        labelbottom=False,
        bottom=False,
        )
plt.savefig(path+"Figures/correlation_AC.PNG", dpi=50)


# %% Export errors at 12h
t2_stat = {
    err_type: {} for err_type in err_types
    }
for err_type in err_types:
    dt = dfs_new[err_type]['012h' + list(err_type)[0].upper()+'01']
    t2_stat[err_type]['mu'] = np.mean(dt)
    t2_stat[err_type]['sd'] = np.std(dt)
    x = np.linspace(min(dt), max(dt), 1000)
    param = norm.fit(dt)
    pdf = norm.pdf(x, *param)
    plt.hist(dt, bins=30, density=True, alpha=0.5, color='b', label='Histogram')
    plt.plot(x, pdf, 'r', label='Fitted Normal Distribution')
    plt.legend()
    plt.xlabel('Value')
    plt.ylabel('Frequency/Probability')
    plt.title(f'Fitting {err_type} error to Normal Distribution')
    plt.show()

file = path+'12hr_avg.json'
with open(file, "w") as json_file:
    json.dump(t2_stat, json_file, indent=4)


# %% Plot distribution of errors
for err_type in err_types:
    ax = dfs_new[err_type].drop(
        columns='000h'+list(err_type)[0].upper()+'01'
        ).plot.box(
            colormap="gray",
            patch_artist=True,
            figsize=(7, 7),
            )
    ax.set_xticklabels(hours[1:])
#   plt.title("Distribution of "+err_type+ " forecast error", fontsize=fsize)
    plt.xlabel("Hours", fontsize=fsize),
    plt.ylabel("Forecast error", fontsize=fsize)
    plt.xticks(fontsize=fsize)
    plt.yticks(fontsize=fsize)
    plt.savefig(
        path+"Figures/forecast_error_dist_{}.PNG".format(err_type),
        dpi=50,
        bbox_inches="tight",
        )


# %% Check correlation
corr = {
    err: np.zeros([len(hours) - 1, len(hours) - 1])
    for err in err_types
    }
for err_type in err_types:
    for i in range(1, len(dfs_new[err_type].columns)):
        for j in range(1, len(dfs_new[err_type].columns)):
            corr_val, _ = pearsonr(
                dfs_new[err_type][dfs_new[err_type].columns[i]].values,
                dfs_new[err_type][dfs_new[err_type].columns[j]].values,
                )
            corr[err_type][i-1, j-1] = corr_val
dfs_cor = {}
for err_type in err_types:
    cor = pd.DataFrame(corr[err_type],
                       list(map(lambda s: s+"h", hours[1:])),
                       columns=list(map(lambda s: s+"h", hours[1:])))
    cor = cor.round(decimals=2)
    dfs_cor[err_type] = cor
    cor.to_csv(
        path+"correlation_{}.csv".format(err_type), index=True, header=True
        )
    fig, ax = plt.subplots(figsize=(9, 9))
    norm = mcolors.Normalize(
        vmin=np.min(cor.values), vmax=np.max(cor.values)+0.3
        )
    heatmap = ax.imshow(cor, cmap="Greys", norm=norm)
    ax.set_xticks(range(cor.shape[1]))
    ax.set_yticks(range(cor.shape[0]))
    ax.set_xticklabels(cor.columns, fontsize=fsize)
    ax.xaxis.tick_top()  # Place the x-ticks at the top
    ax.set_yticklabels(cor.index, fontsize=fsize)
    plt.setp(ax.get_xticklabels(), ha="center")
    # plt.title(
    #     "Coorelation of " + err_type.title() + " forecast error",
    #     fontsize=fsize,
    #     )
    for i in range(cor.shape[0]):
        for j in range(cor.shape[1]):
            ax.text(
                j, i, format(cor.iloc[i, j], ".2f"),
                ha="center", va="center", color="black", fontsize=fsize,
                )
    plt.savefig(
        path+"Figures/correlation_heatmap_{}.PNG".format(err_type),
        dpi=50,
        bbox_inches='tight',
        )


# %% Add errors at 60, 84, and 108 hrs
hours_updated = list(
    ("000", "012", "024", "036", "048", "060",
     "072", "084", "096", "108", "120")
    )
for err_type in err_types:
    err = list(err_type)[0].upper()
    col_to_add = "060h"+err+"01"
    dfs_new[err_type][col_to_add] = dfs_new[err_type][
        ["048h"+err+"01", "072h"+err+"01"]
        ].mean(axis=1)
    col_to_add = "084h"+err+"01"
    dfs_new[err_type][col_to_add] = dfs_new[err_type][
        ["072h"+err+"01", "096h"+err+"01"]
        ].mean(axis=1)
    col_to_add = "108h"+err+"01"
    dfs_new[err_type][col_to_add] = dfs_new[err_type][
        ["096h"+err+"01", "120h"+err+"01"]
        ].mean(axis=1)
    updated_cols = list(map(lambda col: col+"h"+err+"01", hours_updated))
    dfs_new[err_type] = dfs_new[err_type][updated_cols]


# %% Q-Q plots
def qqPlot(err='along', hr='120'):
    e = list(err)[0].upper()
    data = dfs_new[err][f"{hr}h"+e+"01"].values
    mean, std = stats.norm.fit(data)
    theoretical_quantiles = np.quantile(data, np.linspace(0.01, 0.99, 100))
    observed_quantiles = stats.norm.ppf(
        np.linspace(0.01, 0.99, 100),
        loc=mean,
        scale=std,
        )
    plt.figure(figsize=(8, 8))
    plt.scatter(
        theoretical_quantiles, observed_quantiles,
        color="black",
        label=f"FE at {hr} hours",
        )
    plt.plot(
        [min(theoretical_quantiles), max(theoretical_quantiles)],
        [min(theoretical_quantiles), max(theoretical_quantiles)],
        color='black',
        linestyle='--',
        label='45-Degree Line',
        )
    plt.xlabel('Theoretical Quantiles', fontsize=fsize)
    plt.ylabel('Observed Quantiles', fontsize=fsize)
    plt.xticks(fontsize=fsize)
    plt.yticks(fontsize=fsize)
    plt.legend(loc=4, fontsize=fsize)
    plt.title(f'Q-Q Plot of original {err} error at {hr}h', fontsize=fsize)
    plt.grid(False)
    plt.savefig(
        path+f"Figures/qq-plt-original_{err}_err_{hr}_hr.PNG",
        dpi=50,
        bbox_inches="tight",
        )
    plt.show()


qqPlot(err='track', hr='120')


# %%  log-transform track error data
data_track_transformed = pd.DataFrame()
for t in range(1, len(dfs_new['track'].columns)):
    error_data = dfs_new['track'][dfs_new['track'].columns[t]].to_numpy()
    tansformed_data = list(map(lambda x: np.log(x+1), error_data))
    data_track_transformed[dfs_new['track'].columns[t]] = tansformed_data
original_cols = list(map(lambda h: h+"hT01", hours[1:]))
dfs_new['track'] = data_track_transformed
# Plot
ax = data_track_transformed[original_cols].plot.box(
    colormap="gray",
    patch_artist=True,
    figsize=(10, 10),
    )
ax.set_xticklabels(hours[1:])
plt.xlabel("Hours", fontsize=fsize)
plt.ylabel("Log-transformed forecast error", fontsize=fsize)
plt.xticks(fontsize=fsize)
plt.yticks(fontsize=fsize)
plt.title(
    "Distribution of log-transformed track forecast error",
    fontsize=fsize,
    )
plt.savefig(
    path+"Figures/forecast_error_dist_track_transformed.PNG",
    dpi=50,
    bbox_inches="tight",
    )


# %% MLE of AR-1
def getAR1Param(data):
    """
    data: 2D array of forecast error data
    """
    M, T = data.shape
    rho = sum(
        sum(data[i, t]*data[i, t-1] for t in range(1, T))
        for i in range(M)
        ) / sum(sum(data[i, t - 1] ** 2 for t in range(1, T))
                for i in range(M))
    sigma = np.sqrt(
        1.0/(M*(T-1)) * sum(
            sum((data[i, t] - rho*data[i, t-1])**2 for t in range(1, T))
            for i in range(M)
            )
        )
    ar1_param = np.array([0, rho, sigma])  # c = 0
    return ar1_param


ar1 = {}
for err_type in err_types:
    ar1[err_type] = getAR1Param(dfs_new[err_type].to_numpy())

export_ar1_param = pd.DataFrame(ar1, index=['c', 'rho', 'sigma'])
export_ar1_param.rename_axis('parameters', inplace=True)
export_ar1_param.to_csv(path+'ar1.csv', index=True)
