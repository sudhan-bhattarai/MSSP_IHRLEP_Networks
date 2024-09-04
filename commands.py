"""In this file, the description of commands for main.py are presented.
Dictionary 'options' has arguments that take one value from choices.
Dictionary 'continuous_args' has arguments that take any real number.
"""
import pandas as pd


options = {
    "task": {
        "type": str,
        "choices": ["solve", "create_data"],
        "default": "solve",
        "help": "solve: to solve models (mssp or 2ssp); create_data: to create instances"
    },
    "hurricane": {
        "type": str,
        "choices": ["Florence", "Ian"],
        "default": "Ian",
        "help": "Name of hurricanr (Florence or Ian)"
    },
    "landfall": {
        "type": str,
        "choices": ['r', 'd'],
        "default": 'r',
        "help": "hurricane landfall type considered. r=random; d=deterministic. " +
         "d is only applicable to hurricane Florence and r only to Ian. " +
         "This command is only used to clarify the nature of two case studies. " +
         "It is useless in operation."
    },
    "data_opt": {
        "type": int,
        "choices": [1, 2, 3, 4],
        "default": 2,
        "help": "This arg is only applicable to task = create_data. " +
                "option = 1 if creating data related to forecast errors only, " +
                "2 if creating demand data based on pre-existing forecast error data, " +
                "3 if only creating logistic parameters data, " +
                "4 if creating all data at once."
    },
    "model": {
        "type": str,
        "choices": ["2ssp", "mssp"],
        "default": "mssp",
        "help": "This arg is needed when task = solve. " +
                "The options are models to solve (2ssp, or mssp)."
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
        "help": "Algorithm: bb (naive branch & bound); "+
        "bc (branch & bound with lazy cuts through callback); "+
        "ext (extended model (works on 2ssp model only))"
    },
    "instance": {
        "type": int,
        "choices": [1, 2, 3],
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
    "oos_heur": {
        "type": int,
        "choices": [1, 2],
        "default": 1,
        "help": "Heuristic option to do OOS test for MSSP model. 1 if solving closest transient/absorbing states from the tree for transient/absorbing OOS state. 2 if solving the closest cost function regardless of transient/absorbing characteristic."
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
        "help": "Number of sample to use for statistical UB in MSSP model."
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
