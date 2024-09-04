from data_main import Data
import two_stage
import sddp
import results_analysis_archive as ra
import commands
from plot_gis_archive import Plot

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
        print('argument \'--eval\' changed to: \'oos\'')
        args['eval'] = 'oos'
else: # mssp
    args['eval'] = 'both'
    if args['method'] == 'ext':
        print('argument \'--method\' changed to: \'bb\'')
        args['method'] = 'bb'

if args['hurricane'] == 'Ian':
    if args['landfall'] == 'd':
        print('argument \'--landfall\' changed to: \'r\'')
        args['landfall'] = 'r'
    if args['instance'] != 3:
        print('argument \'--instance\' changed to: \'3\'')
        args['instance'] = 3
else: # Florence
    if args['landfall'] == 'r':
        print('argument \'--landfall\' changed to: \'d\'')
        args['landfall'] = 'd'
    if args['oos_heur'] == 2:
        print('argument \'--oos_heur\' changed to: \'1\'')
        args['oos_heur'] == 1


if __name__ == "__main__":

    print("Running main.py with:")
    print('.' * 30)
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

    if args["task"] == "solve":
        if args["model"] == "mssp":
            sddp.solve_mssp(data, **args)

        elif args["model"] == "2ssp":
            Model2SSP = two_stage.SolveStaticTwoStage(data, args)
            Model2SSP.solve_static_2ssp()
            Model2SSP.oos_test_anticipative()
            Model2SSP.oos_test_myopic()
    # archive >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    elif args["task"] == "plot":
        PLOT = Plot(data, args)
        PLOT.plot()

    elif args["task"] == "analyze_result":
        ra.summary(data.DIR[4])
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<