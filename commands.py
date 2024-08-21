"""In this file, the description of commands for main.py are presented.
Dictionary 'options' has arguments that take one value from choices.
Dictionary 'continuous_args' has arguments that take any real number.
"""
import pandas as pd


options = {
    "task": {
        "type": str,
        "choices": ["solve", "create_data", "read_data", "analyze_result", "plot"],
        "default": "solve",
        "help": "solve: to solve models; create_data: to create instances read_data: to display all variables of data classanalyze_result: to plot results of OOS test"
    },
    "hurricane": {
        "type": str,
        "choices": ["Florence", "Ian"],
        "default": "Ian",
        "help": "Name of storm (Florence or Ian)"
    },
    "landfall": {
        "type": str,
        "choices": ['r', 'd'],
        "default": 'r',
        "help": "hurricane landfall type considered. r=random; d=deterministic (only applicable to Florence)."
    },
    "data_opt": {
        "type": int,
        "choices": [1, 2, 3, 4],
        "default": 2,
        "help": "1 if only creating forecast error data; 2 if only creating demand data; 3 if only creating logistic parameters data; 4 if creating all data at once"
    },
    "model": {
        "type": str,
        "choices": ["mv", "2ssp", "rh", "mssp"],
        "default": "mssp",
        "help": "Model to solve (mean_value, 2ssp, rh, or mssp)"
    },
    "delay": {
        "type": int,
        "choices": [1, 2],
        "default": 1,
        "help": "1 if delayed facilities opening allowed in the first-stage MILP; 2 if only allowed to open at t=0"
    },
    "method": {
        "type": str,
        "choices": ["bb", "bc", "ext"],
        "default": "bc",
        "help": "Algorithm: bb (branch & bound); bc (branch & cut); ext (extended model (2ssp only))"
    },
    "instance": {
        "type": int,
        "choices": [1, 2, 3, 4, 5],
        "default": 3,
        "help": "Instance index"
    },
    "eval": {
        "type": str,
        "choices": ["oos", "mc_tree", "both", "none"],
        "default": "both",
        "help": "Type of sample paths (OOS or from MC tree) to evaluate MSSP models on. Two-stage models do not take (both) input since training of two appraoches in 2SSP are different."
    },
    "first_stg_opt": {
        "type": int,
        "choices": [1, 2],
        "default": 1,
        "help": "1 if the first-stage problem is a mixed-integer or,2 if all SPs are open at the beginning and the first-stage problem is a continuous LP"
    },
    "uboption": {
        "type": str,
        "choices": ["D", "S"],
        "default": "S",
        "help": "Upper bound type for SDDP. D for detrministic (not fully developed yet) ; S for statistical"
    },
    "oos_heur": {
        "type": int,
        "choices": [1, 2],
        "default": 1,
        "help": "Heuristic option to do OOS test. 1 if solving closest transient/absorbing states from the tree for transient/absorbing OOS state. 2 if solving the closest cost function regardless of transient/absorbing characteristic."
    },
    "plot_opt": {
        "type": int,
        "choices": [1, 2, 3, 4],
        "default": 1,
        "help": "Options of plotting GIS data. 1 for facilities only, 2 for facilities and scenarios, 3 for a scenario, and mc_ian, 4 for activated SPs."
    },
    "demand_opt": {
        "type": int,
        "choices": [1, 2],
        "default": 1,
        "help": "1 if demand is represented as a fraction of the remaining population; 2 if demand is represented as the total evacuation demand"
    },
    "fix_along_err": {
        "type": int,
        "choices": [0, 1],
        "default": 0,
        "help": "1 if along error is fixed to zero; 0 o/w. Only works for " +
        "hurricane Florence with random landfall case."
    },
    "ST_track": {
        "type": int,
        "choices": [3, 5, 10, 15],
        "default": 15,
        "help": "The number of MC States to use for track-error in Florence case with "+
        "deterministic landfall."
    },

}


continuous_args = {
    "gfact": {
        "type": int,
        "default": 100,
        "help": "Emergency cost factor"
    },
    "pfact": {
        "type": int,
        "default": 200,
        "help": "Penalty cost factor"
    },
    "ffact": {
        "type": int,
        "default": 3,
        "help": "Fixed cost factor"
    },
    "purchase_cost": {
        "type": float,
        "default": 1,
        "help": "Per unit relief items purchase cost"
    },
    "time_limit_train": {
        "type": int,
        "default": 3600,
        "help": "Time limit (in sec) for training (optimization algorithm)"
    },
    "time_limit_test": {
        "type": int,
        "default": 3600,
        "help": "Time limit (in sec) for evaluation (testing on out-sample or samples from the tree)"
    },
    "n_UB_samples": {
        "type": int,
        "default": 1000,
        "help": "Number of sample to use for statistical UB"
    },
    "n_oos": {
        "type": int,
        "default": 1000,
        "help": "Number of out-of-samples to evaluate the model on"
    },
}


def get_commands():
    # Read defaults from .csv for ease
    df = pd.read_csv('command_defaults.csv', sep=' ', index_col=0)
    defaults = {k: v for k, v in zip(df.index, df.values.ravel())}
    for k, v in defaults.items():
        try:
            defaults[k] = int(v)
        except ValueError:
            try:
                defaults[k] = float(v)
            except ValueError:
                defaults[k] = v
    return options, continuous_args, defaults
