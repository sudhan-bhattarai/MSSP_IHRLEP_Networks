from data_main import Data
import two_stage
import sddp
import results_analysis as ra
import commands

import pandas as pd
import os
from argparse import ArgumentParser
from argparse import ArgumentDefaultsHelpFormatter as arg_help_format
import warnings
warnings.simplefilter("ignore")


parser = ArgumentParser(formatter_class=arg_help_format)
options, continuous_args, defaults = commands.get_commands()


for arg_name, val_dict in options.items():
    parser.add_argument("--" + arg_name,
                        type=val_dict['type'],
                        choices=val_dict['choices'],
                        default=defaults[arg_name],
                        help=val_dict['help'])

for arg_name, val_dict in continuous_args.items():
    parser.add_argument("--" + arg_name,
                        type=val_dict['type'],
                        default=defaults[arg_name],
                        help=val_dict['help'])

args = vars(parser.parse_args())

# correct wrong combinations
if args['model'] == '2ssp':
    if args['eval'] not in ['mc_tree', 'oos']:
        args['eval'] = 'oos'

if __name__ == "__main__":

    print("Solving for: ")
    print("{:<20} {:<20}".format("arg", "value"))
    print('.' * 30)
    for key, value in args.items():
        print("{:<20} {:<20}".format(key, value))
    print('.' * 30, "\n")

    data = Data(args)

    if args["task"] == "create_data":
        data.create_data()
    else:
        data.read_data()

    if args["task"] == "read_data":
        data_attrs = sorted(list(vars(data).keys()))
        print("\n Data attributes: {} \n".format(data_attrs))
        # print(data.demand_in_sample, '\n', data.demand_oos)
        print(data.demand_oos.keys(), data.demand_in_sample.keys())

    if args["task"] == "solve":
        data.n_itr_lb_rate = 100
        if args["model"] == "mssp":
            sddp.solve_mssp(data, **args)

        elif args["model"] == "rh":
            RH = two_stage.RollingHorizon(data, args)
            RH.solve_rolling_horizon()

        elif args["model"] == "2ssp":
            # if args['demand_opt'] == 2:
            #     data.DIR[4] = data.DIR[4] + '2SSP-alternative/'
            #     os.makedirs(data.DIR[4]) \
            #         if os.path.isdir(data.DIR[4]) is False else \
            #         None
            Model2SSP = two_stage.SolveStaticTwoStage(data, args)
            # # Model2SSP.solve_static_2ssp()
            Model2SSP.oos_test_anticipative()
            # Model2SSP.oos_test_myopic()

        elif args["model"] == "mv":
            print('incomplete work on reactive_plan.py')
            exit(0)
            # reactive_plan.solve_mvp(data, args)

    elif args["task"] == "plot":
        from plot_gis import Plot
        PLOT = Plot(data, args)
        PLOT.plot()

    elif args["task"] == "analyze_result":
        ra.summary(data.DIR[4])
