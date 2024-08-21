import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
import os


def plotResults(data, **arg):
    result = {"OOS": {}, "MC Tree": {}}
    for m in ["2ssp", "mssp"]:
        result["OOS"][m] = pd.read_csv(
            data.DIR["result"] + f"{m}_eval_oos.csv",
            index_col=0,
            )
        result["MC Tree"][m] = pd.read_csv(
            data.DIR["result"] + f"{m}_eval_mc_tree.csv",
            index_col=0,
            )
    # Average component costs
    for test in ["OOS", "MC Tree"]:
        n_col = len(result[test]["mssp"].columns)
        fig, ax = plt.subplots()
        ind = np.arange(n_col)
        width = 0.35
        ax.bar(ind - width/2, result[test]["mssp"].mean(), 
               width, label='mssp')
        ax.bar(ind + width/2, result[test]["2ssp"].mean(), 
               width, label='2ssp')
        ax.set_xlabel('Cost components')
        ax.set_ylabel('Average cost')
        ax.set_title('MSSP vs 2SSP: {}'.format(test))
        ax.set_xticks(ind)
        ax.set_xticklabels(result[test]["mssp"].columns, 
                           rotation = 45, 
                           ha="right")
        ax.legend()
        if test == "OOS":
            ymin, ymax = ax.get_ylim()
        else:
            ax.set_ylim(ymin, ymax)
        plt.savefig(data.DIR["result"] + "{}_comp.PNG".format(test), 
                    dpi=200, bbox_inches="tight")
    # Average total cost: box plot
    for test in ["OOS", "MC Tree"]:
        total_mssp = result[test]["mssp"].mean(axis=1)
        total_2ssp = result[test]["2ssp"].mean(axis=1)
        fig, ax = plt.subplots()
        ax.boxplot([total_mssp, total_2ssp], labels=['mssp', '2ssp'])
        ax.set_title('MSSP vs 2SSP: {}'.format(test))
        ax.set_ylabel('Average cost')
        plt.savefig(data.DIR["result"] + "{}_total_box.PNG".format(test), 
                    dpi=200, bbox_inches="tight")
    # Average total cost: bar plot
    for test in ["OOS", "MC Tree"]:
        avg_mssp = np.mean(result[test]["mssp"].mean(axis=1))
        avg_2ssp = np.mean(result[test]["2ssp"].mean(axis=1))
        fig, ax = plt.subplots()
        ax.bar("mssp", avg_mssp, label="mssp")
        ax.bar("2ssp", avg_2ssp, label="2ssp")
        ax.set_title('MSSP vs 2SSP: {}'.format(test))
        ax.set_ylabel('Average total cost')
        if test == "OOS":
            ymin, ymax = ax.get_ylim()
        else:
            ax.set_ylim(ymin, ymax)
        plt.savefig(data.DIR["result"] + "{}_total_bar.PNG".format(test), 
                    dpi=200, bbox_inches="tight") 


def make_dirs(confg_lst, hurr, opt):
    path = f'Results/{hurr}/instance3/'
    if opt == 1:
        return path
    else:
        path_list = list(path + 'ff{}_gf100_pf{}_pcost{}/'.format(
            c[0], c[2], c[1]
            ) for c in confg_lst
            )
        return path_list


def aggregate_comp_costs(df_old):
    df = df_old[[*df_old.columns]]
    df['Inventory'] = df['Relief Inventory'] + df['Evacuee Inventory']
    df['Transportation'] = df['Relief Transportation'] +\
        df['Evacuee Transportation']
    df.drop(columns=['Relief Inventory',
                     'Evacuee Inventory',
                     'Relief Transportation',
                     'Evacuee Transportation',
                     'Relief Dumping',
                     'Total'],
            inplace=True,
            )
    try:
        df.drop(columns=['s.1'], inplace=True)
    except KeyError:
        pass
    return df


def summary(path):
    """
    Given path to an model result folder, produce analysis result.
    """
    # order: oos, tree, 2ssp
    label = [
        'mssp_eval_oos_heur1_bb',
        'mssp_eval_mc_tree_bb',
        '2ssp_eval_oos_bc',
        '2ssp_eval_oos_heur1',
        ]
    names = list(map(lambda name: name + '.csv', label))
    dfs = []
    for name in names:
        try:
            dfs.append(pd.read_csv(path + name, index_col=0))
        except FileNotFoundError:
            print('file not found', path + name)
            continue
    # Box plots
    dfs_agg = list(map(lambda df: aggregate_comp_costs(df), dfs))
    for name, df in zip(label, dfs_agg):
        boxplot = df.plot.box(cmap='viridis', legend=True)
        boxplot.set_xticklabels(boxplot.get_xticklabels(), rotation=45)
        plt.savefig(path + f'{name}.PNG', dpi=200, bbox_inches='tight')
    # total box-plot
    df = pd.DataFrame()
    df['MSSP'] = dfs[0]['Total']
    df['2SSP-1'] = dfs[-2]['Total']
    df['2SSP-2'] = dfs[-1]['Total']
    ax = df.plot.box(cmap='viridis')
    plt.savefig(path + 'box_plot.PNG', dpi=200, bbox_inches='tight')
    # barplots comparison
    df = pd.DataFrame()
    df['MSSP'] = dfs_agg[0].mean(axis=0)
    df['2SSP-1'] = dfs_agg[2].mean(axis=0)
    df['2SSP-2'] = dfs_agg[3].mean(axis=0)

    ax = df.transpose().plot.bar(stacked=True, cmap='viridis', edgecolor='k')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig(path + 'bar_plot.PNG', dpi=200, bbox_inches='tight')
    # summary
    summary = [df.describe().transpose() for df in dfs]
    for name, df in zip(names, summary):
        total = df.loc['Total']['mean']
        percent_lst = [c / total for c in df['mean']]
        percent_lst = list(map(lambda x: round(x, 2), percent_lst))
        df['percent'] = percent_lst
        df = df.round(2)
        df = df.rename_axis('cost')
        df = df.drop(['count'], axis=1)
        df.to_csv(path + 'summary_' + name)


def aggregate_results(confg_lst, hurr):
    common_dir = make_dirs(confg_lst, hurr, 1)
    dirs = make_dirs(confg_lst, hurr, 2)
    ffact = [c[0] for c in confg_lst]
    pfact = [c[2] for c in confg_lst]
    pcost = [c[1] for c in confg_lst]

    # # Aggregate algorithm results.
    # results_lst = {'2ssp': [], 'sddp': []}
    # for path in dirs:
    #     for json_file in ['2ssp_bc_oos.json', 'sddp_bb.json']:
    #         try:
    #             with open(path + json_file, 'r') as open_file:
    #                 file = json.load(open_file)
    #         except FileNotFoundError:
    #             print('error! file not found')
    #             print(path + json_file)
    #             exit(0)
    #         index = json_file.split('_')[0]
    #         results_lst[index].append(file)
    # alg_results = {
    #     'ffact': ffact,
    #     'pfact': pfact,
    #     'pcost': pcost,
    #     }
    # temp = dict(
    #     benders_ub=list(file['ub'] for file in results_lst['2ssp']),
    #     benders_comp_time=list(file['comp_time']
    #                            for file in results_lst['2ssp']),
    #     benders_num_nodes=list(file['num_nodes']
    #                            for file in results_lst['2ssp']),
    #     benders_mip_gap=list(file['MIP_gap'] for file in results_lst['2ssp']),
    #     # sddp_lb=list(file['lb'] for file in results_lst['sddp']),
    #     # sddp_ub=list(file['ub_avg'] for file in results_lst['sddp']),
    #     # sddp_sd=list(file['ub_sd'] for file in results_lst['sddp']),
    #     # sddp_comp_time=list(file['comp_time'] for file in results_lst['sddp']),
    #     # sddp_lb_change_rate=list(file['lb_change_rate']
    #     #                          for file in results_lst['sddp']),
    #     # sddp_num_paths=list(file['num_paths']
    #     #                     for file in results_lst['sddp']),
    #     )
    # alg_results.update(temp)
    # alg_df = pd.DataFrame(alg_results).round(3)
    # alg_df.to_csv(common_dir + 'algorithm_results.csv',
    #               index=False)

    # objectives and sensitivity
    names = [
        # 'mssp_eval_oos_heur1_bb.csv',
        # 'mssp_eval_mc_tree_bb.csv',
        '2ssp_eval_oos_bc_heur1.csv',
        # '2ssp_eval_mc_tree_bc.csv',
        # '2ssp_eval_mc_tree_bc_heur1.csv'
        ]
    for name in names:
        dfs = list(pd.read_csv(path + 'summary_' + name, index_col=0) for path in dirs)
        # upper bound results
        ub_mean = [df.loc['Total', 'mean'] for df in dfs]
        ub_std = [df.loc['Total', 'std'] for df in dfs]
        ub_df = pd.DataFrame({'ffact': ffact,
                            'pfact': pfact,
                            'pcost': pcost,
                            'total_mean': ub_mean,
                            'total_std': ub_std,
                            }).round(2)
        ub_df.to_csv(common_dir + 'aggregated_' + name,
                     index=False
                    )
        cost_comp_name = ['Fixed', 'Relief Inventory', 'Evacuee Inventory', 'Penalty',
                          'Emergency', 'Relief Purchase', 'Relief Transportation',
                          'Evacuee Transportation', 'Relief Dumping']

        # percent comp cost results
        percent_lst =  [[df.loc[cost_name, 'percent']
                            for cost_name in cost_comp_name]
                            for df in dfs]
        percent = {cost_name: [p[i] for p in percent_lst]
                    for i, cost_name in enumerate(cost_comp_name)
                    }
        data = {'ffact': ffact, 'pfact': pfact, 'pcost': pcost}
        data.update(percent)
        percent_df = pd.DataFrame(data)
        percent_df.to_csv(common_dir + 'percent_' + name,
                     index=False
                    )


def sensitivity_plot(paths):
    df_2ssp = [pd.read_csv(path + 'summary_2ssp_eval_oos_bc_heur1.csv',
                       index_col=0)
           for path in paths]
    df_mssp = [pd.read_csv(path + 'summary_2ssp_eval_oos_bc_heur1.csv',
                       index_col=0)
           for path in paths]


def summary_plot(path):
    names = ['summary_mssp_eval_oos_heur1_bb',
             'summary_2ssp_eval_oos_bc',
             'summary_2ssp_eval_oos_heur1',
             ]
    models = ['MSSP', '2SSP-anticipative', '2SSP-myopic']
    files = list(map(lambda x: path + x + '.csv', names))
    df = pd.DataFrame()
    for file, model in zip(files, models):
        df_temp = pd.read_csv(file, index_col=0)
        df[model] = df_temp['mean']
    df = df.transpose()
    # df = df.drop(index=['s.1', 'Total', 'Relief Dumping'])
    # df.append(df.loc['Relief Inventory'] + df.loc['Evacuee Inventory'])
    df['Inventory'] = df['Relief Inventory'] + df['Evacuee Inventory']
    df['Transportation'] = df['Relief Transportation'] +\
        df['Evacuee Transportation']
    df = df.drop(columns=['s.1', 'Total', 'Relief Dumping',
                          'Relief Inventory', 'Evacuee Inventory',
                          'Relief Transportation', 'Evacuee Transportation'
                          ])
    df.plot.bar(stacked=True, legend=True, cmap='copper')
    plt.xticks(rotation=0)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig('temp.PNG', dpi=200, bbox_inches='tight')


base_case_confg_list = [
    [5, 5.0, 200]
]
fixed_cost_confg_list = [
    [1, 5.0, 200],
    [10, 5.0, 200],
    [20, 5.0, 200],
    [50, 5.0, 200],
    ]
penalty_cost_confg_list = [
    [5, 5.0, 50],
    [5, 5.0, 100],
    [5, 5.0, 300],
    [5, 5.0, 500],
    ]
purchase_cost_confg_list = [
    [5, 1.0, 200],
    [5, 10.0, 200],
    [5, 50.0, 200],
    ]


confg_list = [
         # base case
         [5, 5.0, 200],
         # fixed cost sensitivity
         [1, 5.0, 200],
         [10, 5.0, 200],
         [20, 5.0, 200],
         [50, 5.0, 200],
         # penalty cost sensitivity
         # [5, 5.0, 10],
         [5, 5.0, 50],
         [5, 5.0, 100],
         [5, 5.0, 300],
         [5, 5.0, 500],
         # purchase cost sensitivity
         [5, 1.0, 200],
         [5, 10.0, 200],
         [5, 50.0, 200],
         [5, 100.0, 200],
         ]


if __name__ == '__main__':
    # for path in make_dirs(confg_list, "Ian", 2):
    #     summary(path)
    # aggregate_results(confg_list, 'Ian')
    print(summary_plot(
        r'Results/Florence/instance3/ff5_gf100_pf200_pcost5.0/'
        ))
