import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
import json
import copy

import helper


def createErrorSamples(data, err, S, t, oos_err_at_t=0.0):
    """
    Create sample paths of forecast errors.

    err : "intensity", "track", "along", or "cross"
        Type of error.
    S : int
        Number of samples to generate.
    t : int
        Starting period.
    oos_err_at_t : float
        Deterministic out-of-sample error at 't'.
    """
    T = data.T if err == 'track' else data.T_max
    samples = np.zeros([S, T])
    for s in range(S):
        # t1 is the first time period for which the error is deterministic
        if err == 'track':
            if t == 0:
                samples[s, 1] = np.log(data.avg_err_12h["track"]['mu'] + 1)
                t1 = 2
            else:
                samples[s, t] = np.log(abs(oos_err_at_t) + 1)
                t1 = t + 1
        else:
            samples[s, t] = oos_err_at_t
            t1 = t + 1

        for period in range(t1, T):
            samples[s, period] = sum(
                (data.ar1[err][1] * samples[s, period - 1],
                 np.random.choice(data.eps_grid[err][period]))
                )
        # restore transformation of track error data
        if err == "track":
            for period in range(t, T):
                samples[s, period] = np.exp(samples[s, period]) - 1
            samples[s, :] = samples[s, :] * np.random.choice([1, -1])
            # Bernoulli trial to add or subtract track error from forecast
            if t > 0:
                samples[s, t] = oos_err_at_t  # No Bernoulii at t
    return samples.round(2)


class ForecastError:
    def __init__(self, args):
        self.fe_path = r"Data/Forecast error/"
        self.fig_path = self.fe_path + "Figures/"
        if os.path.isdir(self.fig_path) is False:
            os.makedirs(self.fig_path)
        # df = pd.read_csv("Data/numeric_inputs.csv", index_col=0)
        for attr, val in args.items():
            setattr(self, attr, val)
        self.err_type = ["intensity", "track", "along", "cross"]
        with open(self.fe_path + "12hr_avg.json", 'r') as jsonfile:
            self.avg_err_12h = json.load(jsonfile)
        # MLE estimated AR-1 parameters
        df_ar1 = pd.read_csv(self.fe_path + "ar1.csv", index_col=0)
        self.ar1 = {err: df_ar1[err].values for err in self.err_type}
        with open('Data/num_of_mc_states_at_Tmax.json', 'r') as file:
            self.ST_max = json.load(file)

    def epsGrid(self, mode='r'):
        """Recombined tree of epsilon error of Ar-1 models of each forecast
        error"""
        path = self.fe_path + 'eps_grid.json'
        if not os.path.exists(path):
            mode = 'w'
        if mode == 'r':
            with open(path, 'r') as file:
                self.eps_grid = json.load(file)
            return None
        self.eps_grid = {err: np.zeros([self.n_realization, 1])
                         for err in self.err_type}
        # Since \xi_0 and xi_1 are deterministic,
        # we only have uncertainty in errors from t = 2
        for t in range(1, self.T_max):
            for err in self.err_type:
                self.eps_grid[err] = np.column_stack(
                    (self.eps_grid[err],
                     np.random.normal(
                         0, self.ar1[err][2], self.n_realization
                         ))
                )
        self.eps_grid = {err: arr.tolist()
                         for err, arr in self.eps_grid.items()}
        with open(path, 'w') as file:
            json.dump(self.eps_grid, file, indent=4)

    def errorOOS(self, n_oos, mode="r"):
        """
        n_oos : int
            Number of out-of-sample forecast errors to create
            for each error type.
        mode: 'r' or 'w'
            'r' to read existing data; 'w' to write (create) new data.
        """
        path = self.fe_path + 'oos_err.json'
        if not os.path.exists(path):
            mode = 'w'
        if mode == 'r':
            with open(path, 'r') as file:
                self.oos_err = json.load(file)
            return None
        self.oos_err = {}
        for err in self.err_type:
            oos = createErrorSamples(
                data=self, err=err, S=n_oos, t=0, oos_err_at_t=0
                )
            self.oos_err[err] = oos.tolist()
            # plot
            df = pd.DataFrame(oos)
            fig, ax = plt.subplots()
            df.boxplot(ax=ax)
            plt.grid(False)
            plt.xlabel("t")
            plt.ylabel("Forecast error")
            plt.title("Forecast error samples")
            plt.savefig(
                self.fig_path + f"samples_{err}.PNG",
                dpi=300,
                bbox_inches="tight",
                )
        with open(path, 'w') as file:
            json.dump(self.oos_err, file, indent=4)

    def discretize(self, xi, N):
        """For a given vector of samples, return discretized means.
        xi: list or 1D array
            Error vector
        N: int
            Number of discrete means."""

        # Initialize mu with quantiles
        quant = [i / (N + 1) * 100 for i in range(1, N + 1)]
        mu = np.percentile(xi, quant).tolist()
        S = len(xi)
        for s in range(1, S):
            beta = 1 / s
            diff = [(xi[s] - mu[i]) ** 2 for i in range(N)]
            k = np.argmin(diff)  # index of ``the closest \mu
            for i in range(N):  # update mean
                mu[k] = (1 - beta) * mu[k] + beta * xi[s]
        mu = list(map(lambda x: round(x, 2), mu))
        # Assign samples to the respective partition means
        gamma = {m: [] for m in mu}  # collection of clusters
        assign = [0]*S
        for s in range(S):
            assign[s] = np.argmin([abs(xi[s]-mu[i]) for i in range(N)])
            closest_mu = mu[assign[s]]
            gamma[closest_mu].append(s)
        # Calculate the loss of partition
        loss = sum([abs(xi[s] - mu[assign[s]]) for s in range(S)])
        avg_loss = 1.0 / S * loss
        return mu, avg_loss, assign, gamma

    def numStates(self, err):
        """For a given error name, get the number of
        discrete states to use under an 'error threshold'
        (read from user defined .csv file)."""

        T = 11 if err == "track" else self.T_max
        # ST_max = self.ST.loc["Value", f"S_T_{err}"]
        ST_max = getattr(self, f"S_T_{err}")
        loss = np.zeros([ST_max, T])
        self.ST_lst = [i + 1 for i in range(ST_max)]
        for t in range(T):
            for i, st in enumerate(self.ST_lst):
                discretize_result = self.discretize(
                    xi=[err_matrix[t] for err_matrix in self.oos_err[err]],
                    N=st,
                    )
                avg_loss = discretize_result[1]
                loss[i, t] = avg_loss / st
        self.loss_tol = loss[-1, -1]
        self.loss_df = pd.DataFrame(
            loss,
            columns=["t={}".format(t) for t in range(T)]
            )
        self.loss_df.set_index(pd.Series(self.ST_lst), inplace=True)
        # Get the number of MC states to use at every t using
        # an average loss threshold at T
        ST = []
        for t in range(T):
            for n in range(ST_max):
                if loss[n, t] <= self.loss_tol:
                    ST.append(n + 1)
                    break
            continue
        # self.plotDiscretizeLoss(ST_max, err)
        return loss, ST

    def plotDiscretizeLoss(self, ST_max, err):
        fsize = 20
        min_val, max_val = 0.0, 0.9
        orig_cmap = plt.cm.gray
        colors = orig_cmap(np.linspace(min_val, max_val, 11))
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
            "mycmap", colors
            )
        ax = self.loss_df.plot(
            marker='o', markersize=2, figsize=(8, 8), cmap=cmap
            )
        ax.plot(self.ST_lst, [self.loss_tol] * ST_max,
                color="black", linewidth=2, label="Error threshold",
                )
        plt.xticks(self.ST_lst)
        plt.legend(fontsize=fsize)
        # loc='center left', bbox_to_anchor=(1, 0.5))
        plt.title(f"Avg MC discretization loss for {err} error",
                  fontsize=fsize + 2
                  )
        plt.xlabel("Number of Markovian states", fontsize=fsize)
        plt.ylabel("Average loss per MC state", fontsize=fsize)
        plt.tight_layout()
        plt.savefig(self.fig_path + "avg_loss_{}.PNG".format(err),
                    dpi=50, bbox_inches="tight",
                    )
        plt.show()

    def discretizeAll(self, mode='r'):
        """Discretization of forecast error for all error types.

        mode : 'r' or 'w'
            'r' to read existing data; 'w' to write(create) new data"""
        # Mode: Read
        file_names = ["pi_tree.json", "err_tree.json", "n_states_tree.json"]
        paths = [self.fe_path + file for file in file_names]
        for path in paths:
            if not os.path.exists(path):
                mode = 'w'
                break
        if mode == 'r':
            attr_names = [name.split(".json")[0] for name in file_names]
            for i, file in enumerate(paths):
                with open(file, 'r') as jsonfile:
                    setattr(self, attr_names[i], json.load(jsonfile))
            return None
        # Mode: Write
        MU = {err: {} for err in self.err_type if err != "intensity"}
        pi = copy.deepcopy(MU)
        St = copy.deepcopy(MU)

        for err in MU.keys():
            for ST_last in self.ST_max[err]:
                MU[err][ST_last] = {}
                pi[err][ST_last] = {}
                St[err][ST_last] = {}
                setattr(self, f"S_T_{err}", ST_last)
                _, ST = self.numStates(err=err)
                GAMMA = list()
                ASSIGN = list()
                T = self.T if err == "track" else self.T_max
                for t in range(T):
                    xi_t = [e[t] for e in self.oos_err[err]]
                    try:
                        mu, avg_loss, assign, gamma = self.discretize(
                            xi=xi_t, N=ST[t]
                            )
                    except IndexError:
                        print('error', ST, T)
                        exit(0)
                    ASSIGN.append(assign)
                    GAMMA.append(gamma)
                    MU[err][ST_last][t] = mu
                St[err][ST_last] = {t: n_state for t, n_state in enumerate(ST)}
                # Compute transition probability
                for t in range(T-1):
                    pi[err][ST_last][t] = {}
                    for mu1 in MU[err][ST_last][t]:
                        pi_temp = []
                        for mu2 in MU[err][ST_last][t+1]:
                            set1 = set(GAMMA[t][mu1])
                            set2 = set(GAMMA[t+1][mu2])
                            num_common = len(set1.intersection(set2))
                            num_all = len(set1)
                            prob = num_common/num_all if num_all > 0.0 else 0.0
                            pi_temp.append(prob if prob > 1e-3 else 0.0)
                        F = sum(pi_temp)
                        pi[err][ST_last][t][mu1] = {
                            mu: round(p/F, 3) for mu, p in zip(
                                MU[err][ST_last][t+1], pi_temp
                            )}
        # Export
        files = {file: val for file, val in zip(paths, [pi, MU, St])}
        for file, val in files.items():
            with open(file, 'w') as jsonfile:
                json.dump(val, jsonfile, indent=4)
        self.discretizeAll(mode='r')

    def createFEData(self, args, oos=False):
        if oos is False:
            self.readFEData(args, oos_only=True)
        else:
            self.epsGrid(mode='w')
            self.errorOOS(n_oos=args['n_oos'], mode='w')
        self.discretizeAll(mode='w')

    def readFEData(self, oos_only, args):
        self.epsGrid(mode='r')
        self.errorOOS(n_oos=args["n_oos"], mode='r')
        self.eps_grid = helper.json_import_conversion(self.eps_grid)
        self.oos_err = helper.json_import_conversion(self.oos_err)
        if not oos_only:
            self.discretizeAll(mode='r')
            self.err_tree = helper.json_import_conversion(self.err_tree)
            self.pi_tree = helper.json_import_conversion(self.pi_tree)
            self.n_states_tree = helper.json_import_conversion(self.n_states_tree)
