import numpy as np
import gurobipy as gb
import pandas as pd
import time
import json

import helper

ct = gb.GRB.CONTINUOUS
bi = gb.GRB.BINARY
inf = gb.GRB.INFINITY


# Sampling method
def random_sample(data, hurr):
    sample = [data.state_space[0][0]]
    for t in range(1, data.T):
        p_lst = data.pi_mssp[t-1][sample[-1]]
        s = np.random.choice(
            range(len(data.state_space[t])),
            p=list(p_lst[s1] for s1 in data.state_space[t])
            )
        sample.append(data.state_space[t][s])
        # Absorbing state
        if hurr == "Ian":
            if data.absorb_mssp[t][sample[-1]] is True:
                break
    return sample


# Build Model
class Model:
    def __init__(self, data, t, args):
        self.args = args
        first_stage = True if t == 0 else False
        delayed = True if args["delay"] == 1 else False
        no_integers = True if args["first_stg_opt"] == 2 else False
        m = gb.Model()
        m.setParam("OutputFlag", 0)
        # VARIABLES
        zJ_lst = [data.J, data.T] if delayed else [data.J]
        # Number of zJ variables according to delayed opening option
        Var = {
            "u": m.addVars(data.I, vtype=ct, lb=0, ub=inf, name='u'),
            "eI": m.addVars(data.I, vtype=ct, lb=0, ub=inf, name='eI'),
            "eJ": m.addVars(data.J, vtype=ct, lb=0, ub=inf, name='eJ'),
            "g": m.addVars(data.J, vtype=ct, lb=0, ub=inf, name='g'),
            "h": m.addVars(data.J, vtype=ct, lb=0, ub=inf, name='h'),
            "y": m.addVars(data.I, data.J, vtype=ct, lb=0, ub=inf, name='y'),
            "lJ": m.addVars(data.J, vtype=ct, lb=0, ub=inf, name='lJ'),
            "xKJ": m.addVars(
                data.K, data.J, vtype=ct, lb=0, ub=inf, name='xKJ'
                ),
            "xJJ": m.addVars(
                data.J, data.J, vtype=ct, lb=0, ub=inf, name='xJJ'
                ),
            "zJ": m.addVars(
                *zJ_lst, vtype=bi if first_stage else ct, name="zJ"
                ),
            }
        m.update()
        Var['theta'] = m.addVar(
            vtype=ct,
            lb=sum(q*h for q, h in zip(data.q_J, data.c_H_J)),
            ub=inf,
            name='theta',
            )
        m.update()
        # CONSTRAINTS
        Constr = {}
        Constr['b'] = m.addConstrs(
            (gb.quicksum(Var['y'][i, j] for j in range(data.J))
             + Var['u'][i] ==
             data.demand_mssp[0][data.state_space[0][0]][i] *
             data.INIT['eI'][i] for i in range(data.I))
            )  # Replace RHS by data.DF_MSSP[t][s][i] *
        # Var['eI'][i, t-1].x for t>0
        Constr['c'] = m.addConstrs(
            (Var['eI'][i]
             + gb.quicksum(Var['y'][i, j] for j in range(data.J))
             == data.INIT['eI'][i] for i in range(data.I))
            )  # replace RHS by Var['eI'][i, t-1].x for t>0
        Constr['d'] = m.addConstrs(
            (Var['eJ'][j]
             - gb.quicksum(Var['y'][i, j] for i in range(data.I)) ==
             data.INIT['eJ'][j] for j in range(data.J))
            )   # replace RHS by Var['eJ'][j, t-1].x for t>0
        Constr['e'] = m.addConstrs(
            (-Var['eJ'][j]
             + data.q_J[j] * (Var['zJ'][j, t] if delayed else Var['zJ'][j])
             >= 0 for j in range(data.J))
            )  # RHS is not updated
        Constr['f'] = m.addConstrs(
            (-Var['lJ'][j]
             + data.phi * data.q_J[j] * (Var['zJ'][j, t] if delayed else
                                         Var['zJ'][j])
             >= 0 for j in range(data.J))
            )
        Constr['g'] = m.addConstrs(
            (gb.quicksum(Var['xKJ'][k, j] for k in range(data.K))
             + gb.quicksum(Var['xJJ'][j_, j] for j_ in range(data.J))
             - gb.quicksum(Var['xJJ'][j, j_] for j_ in range(data.J))
             - data.phi*Var['eJ'][j]
             + Var['g'][j]
             - Var['h'][j]
             - Var['lJ'][j] ==
             - data.INIT['lJ'][j] for j in range(data.J))
            )    # replace RHS by -Var['lJ'][j, t-1].x for t > 0
        Constr['h'] = m.addConstrs(
            (- gb.quicksum(Var['xJJ'][j, j_] for j_ in range(data.J))
             + Var['g'][j]
             - data.phi * Var['eJ'][j] >=
             - data.INIT['lJ'][j] for j in range(data.J))
            )  # replace RHS by - Var['lJ'][j, t-1].x for t>0
        Constr['i'] = m.addConstrs(
            (-Var['g'][j]
             + data.phi * Var['eJ'][j]
             + data.phi * data.q_J[j] * (Var['zJ'][j, t] if delayed else
                                         Var['zJ'][j])
             >=
             data.INIT['lJ'][j] for j in range(data.J))
            )  # replace RHS by Var['lJ'][j, t-1].x for t>0
        m.addConstrs(
            (Var['xJJ'][j, j] == 0 for j in range(data.J))
            )
        # Constraints added only to the model at t=0
        if no_integers is True:
            # Open all SPs at all time (first-stage problem=LP)
            m.addConstrs((
                (Var["zJ"][j, 0] if delayed else Var["zJ"][j])
                == 1 for j in range(data.J)
                ))
        Constr['m'] = {}
        if delayed:
            if first_stage is True:
                m.addConstrs(
                    (Var['zJ'][j, t_+1] >= Var['zJ'][j, t_]
                     for j in range(data.J)
                     for t_ in range(data.T - 1))
                    )
            else:
                for j in range(data.J):
                    for t_ in range(data.T):
                        Constr['m'][j, t_] = m.addConstr(
                            Var["zJ"][j, t_] == 0
                        )  # replace RHS by Var['zJ'][j, t].x from model at t=0
        else:
            if first_stage is False:
                Constr["m"] = m.addConstrs(
                    (Var["zJ"][j] == 0 for j in range(data.J))
                    )  # replace RHS by Var['zJ'][j].x from model at t-1
        m.update()
        if args['absorbing'] is True:
            m.addConstr(Var['theta'] == 0)
            m.update()

        # OBJECTIVE
        # 1. Define fixed-cost at first stage for different options
        if first_stage is True:
            # One time fixed setup cost of Shelters
            fixed_cost = gb.quicksum(
                data.c_F_J[j] *
                (Var["zJ"][j, data.T-1] if delayed is True else Var["zJ"][j])
                for j in range(data.J)
                )
            # Per period maintenance cost of shelters
            fixed_cost += gb.quicksum(
                gb.quicksum(
                    data.c_F_J_var[j] *
                    (Var["zJ"][j, t_] if delayed is True else Var["zJ"][j])
                    for t_ in range(data.T)
                    ) for j in range(data.J)
                )
        else:
            fixed_cost = 0.0
        # 2. Define the overall objective
        m.setObjective(
            fixed_cost
            + gb.quicksum(gb.quicksum(
                (data.c_R_KJ[k][j] + data.c_P_K[k]) * Var['xKJ'][k, j]
                for j in range(data.J)) for k in range(data.K))
            + gb.quicksum(gb.quicksum(
                data.c_R_JJ[j][j_] * Var['xJJ'][j, j_]
                for j in range(data.J)) for j_ in range(data.J))
            + gb.quicksum(gb.quicksum(
                data.c_E_IJ[i][j] * Var['y'][i, j]
                for i in range(data.I)) for j in range(data.J))
            + gb.quicksum(data.c_G_J[j] * Var['g'][j] for j in range(data.J))
            + gb.quicksum(data.c_H_J[j] * Var['h'][j] for j in range(data.J))
            + gb.quicksum(data.c_PE[i] * Var['u'][i] for i in range(data.I))
            + gb.quicksum(data.c_invE_J[j] * Var['eJ'][j]
                          for j in range(data.J))
            + gb.quicksum(data.c_invR_J[j] * Var['lJ'][j]
                          for j in range(data.J))
            + Var['theta'],
            gb.GRB.MINIMIZE
            )

        m.update()

        self.m = m
        self.m._vars = Var
        self.m._constrs = Constr
        self.cuts_rhs = []  # store cut RHS after adding cuts at "t"
        self.m._constrs['cuts'] = []   # to store the cuts added later
        self.first_stage = first_stage
        self.data = data
        self.t = t
        self.delayed = delayed
        self.sol = {}

    def get_sol(self, state_vars_only=True, cb=False):
        if state_vars_only:
            var_names = ["eJ", "eI", "lJ", "zJ", "theta"]
        else:
            var_names = list(self.m._vars.keys())
        x_hat = {name: {} for name in var_names}
        for var, var_dict in self.m._vars.items():
            if var in var_names:
                if var == "theta":
                    x_hat[var] = self.m.cbGetSolution(var_dict) if cb \
                        else var_dict.x
                else:
                    for i, val in var_dict.items():
                        x_hat[var][i] = self.m.cbGetSolution(val) if cb \
                            else val.x
        self.sol = x_hat
        return x_hat

    def cost_component(self):
        x_hat = self.get_sol(state_vars_only=False, cb=False)
        # Fixed cost
        if self.first_stage is True:
            if self.delayed is True:
                fixed1 = sum(self.data.c_F_J[j] * x_hat["zJ"][j, self.data.T-1]
                             for j in range(self.data.J))
                fixed2 = sum(sum(self.data.c_F_J_var[j] * x_hat["zJ"][j, t]
                                 for j in range(self.data.J))
                             for t in range(self.data.T))
            else:
                fixed1 = sum(self.data.c_F_J[j] * x_hat["zJ"][j]
                             for j in range(self.data.J))
                fixed2 = sum(sum(self.data.c_F_J_var[j] * x_hat["zJ"][j]
                                 for j in range(self.data.J))
                             for t in range(self.data.T))
            fixed = fixed1 + fixed2
        else:
            fixed = 0.0
        return {
            "Fixed":  fixed,
            "Relief Inventory": sum(
                self.data.c_invR_J[j] * x_hat['lJ'][j]
                for j in range(self.data.J)
                ),
            "Evacuee Inventory": sum(
                self.data.c_invE_J[j] * x_hat['eJ'][j]
                for j in range(self.data.J)
                ),
            "Penalty": sum(
                self.data.c_PE[i] * x_hat['u'][i] for i in range(self.data.I)
                ),
            "Emergency": sum(
                self.data.c_G_J[j] * x_hat['g'][j] for j in range(self.data.J)
                ),
            "Relief Purchase": sum(sum(
                self.data.c_P_K[k] * x_hat['xKJ'][k, j]
                for j in range(self.data.J)) for k in range(self.data.K)),
            "Relief Transportation": sum(
                [sum(sum(self.data.c_R_JJ[j][j_]*x_hat['xJJ'][j, j_]
                         for j in range(self.data.J)
                         ) for j_ in range(self.data.J)),
                 sum(sum(self.data.c_R_KJ[k][j] * x_hat['xKJ'][k, j]
                         for j in range(self.data.J)
                         ) for k in range(self.data.K))
                 ]
                ),
            "Evacuee Transportation": sum(
                sum(self.data.c_E_IJ[i][j] * x_hat['y'][i, j]
                    for i in range(self.data.I)
                    ) for j in range(self.data.J)
                ),
            "Relief Dumping": sum(
                self.data.c_H_J[j] * x_hat['h'][j]
                for j in range(self.data.J)
                )
            }

    def update_rhs(self, x_sol, XI):
        """
        Update RHS of constraints $$(Ax_t=b+Bx_{t-1})$$ of
        the model at 't' with
        (1) state variables' solution from 't-1' and,
        (2) the uncertain data (XI).
        At t = 0, we do not need to update RHS as the model is
        constructed with the deterministic initial conditions
        """
        rhs = gb.GRB.Attr.RHS
        # updating RHS is not required for the model at the root node
        if self.t > 0:
            for i in range(self.data.I):
                self.m._constrs['b'][i].setAttr(
                    rhs, XI[i] * max(x_sol['eI'][i], 0)
                    )
                self.m._constrs['c'][i].setAttr(
                    rhs, max(x_sol['eI'][i], 0)
                    )
            for j in range(self.data.J):
                self.m._constrs['d'][j].setAttr(
                    rhs, max(x_sol['eJ'][j], 0)
                    )
                self.m._constrs['g'][j].setAttr(
                    rhs, - max(x_sol['lJ'][j], 0)
                    )
                self.m._constrs['h'][j].setAttr(
                    rhs, - max(x_sol['lJ'][j], 0)
                    )
                self.m._constrs['i'][j].setAttr(
                    rhs, max(x_sol['lJ'][j], 0)
                    )
                if self.delayed is True:
                    for t in range(self.data.T):
                        self.m._constrs['m'][j, t].setAttr(
                            rhs, round(x_sol['zJ'][j, t])
                            )
                else:
                    self.m._constrs['m'][j].setAttr(
                        rhs, round(x_sol['zJ'][j])
                        )
            self.m.update()

    def get_duals(self):
        """ Store the dual coefficients of the constraints of
        cost-to-go functions """
        duals = {}
        # dual of the constraints before cuts
        for constr in ['b', 'c']:
            duals[constr] = list(
                self.m._constrs[constr][i].pi for i in range(self.data.I)
                )
        for constr in ['d', 'g', 'h', 'i']:
            duals[constr] = list(
                self.m._constrs[constr][j].pi for j in range(self.data.J)
                )
        if self.delayed is True:
            duals['m'] = list(
                list(self.m._constrs['m'][j, t].pi
                     for t in range(self.data.T))
                for j in range(self.data.J)
                )
        else:
            duals['m'] = list(
                self.m._constrs['m'][j].pi
                for j in range(self.data.J)
                )
        # dual of the cuts
        duals['cuts'] = list(map(
            lambda constr: constr.pi, self.m._constrs['cuts']
            ))
        self.duals = duals


def make_models(data, args):
    """Make one model per MC state per period.
    Hurricane choice is embedded in data.
    """
    models = {}
    for t in range(data.T):
        for s in data.state_space[t]:
            try:
                args['absorbing'] = data.absorb_mssp[t][s]
            except KeyError:
                print(data.absorb_mssp)
                print(data.state_space[t])
                exit(0)
            models[t, s] = Model(data=data, t=t, args=args)
            models[t, s].t = t
            models[t, s].s = s
    return models


class SDDP:
    def __init__(self, data, initial_models, **kwargs):
        self.data = data
        self.models = initial_models
        self.args = kwargs

    def forward_pass(self, sample):
        """
        sample: sample path $$s_t, t=0, ..., T$$ where $s_t$ = 0, 1, ...
        is the sampled state at period t
        XI: list of arrays of the uncertain demand realization (demand factor)
        for all states at all periods. XI for training and out-of-sample
        testing are different
        """
        T = len(sample)
        Obj = 0.0
        if self.root is True:  # solve model at t=0 (root=True)
            s1 = sample[0]
            self.models[0, s1].m.optimize()
            Obj += self.models[0, s1].m.objVal - \
                self.models[0, s1].m._vars['theta'].x
        for t in range(1, T):  # solve from t=2 to T
            s = sample[t]  # MC state of sample at "t"
            s_ = sample[t-1]  # MC state of sample at "t-1"
            self.models[t-1, s_].get_sol(
                state_vars_only=True,
                cb=self.args["cb"] if t-1 == 0 else False,
                )
            self.models[t, s].update_rhs(
                self.models[t-1, s_].sol,
                XI=self.data.demand_mssp[t][s]
                )
            self.models[t, s].m.optimize()
            if self.models[t, s].m.status != gb.GRB.OPTIMAL:
                print('Model infeasible in forward pass')
                self.models[t, s].m.write('temp.lp')
                print(t, s, s_, self.models[t-1, s_].sol)
                exit(0)
            if self.root:
                Obj += self.models[t, s].m.objVal - \
                    self.models[t, s].m._vars['theta'].x
        return Obj if self.root else None

    def generate_cut(self, t, st, X):
        state_space_t_plus_1 = list(
            state for state in self.data.state_space[t+1]
            if self.data.pi_mssp[t][st][state] > 0.0
            )
        pi_lst = list(
            self.data.pi_mssp[t][st][st1]
            for st1 in state_space_t_plus_1
            )
        CUTS = []
        for s_ in state_space_t_plus_1:
            DUALS = self.models[t+1, s_].duals
            XI = self.data.demand_mssp[t+1][s_]
            CUT_RHS = self.models[t+1, s_].cuts_rhs
            # dot product 'a.x' of cuting plane 'ax+b'
            lhs1 = sum(
                DUALS['b'][i] * X['eI'][i] * XI[i] for i in range(self.data.I)
                )
            lhs2 = sum(DUALS['c'][i] * X['eI'][i] for i in range(self.data.I))
            lhs3 = sum(DUALS['d'][j] * X['eJ'][j] for j in range(self.data.J))
            lhs4 = sum(DUALS['g'][j] * -X['lJ'][j] for j in range(self.data.J))
            lhs5 = sum(DUALS['h'][j] * -X['lJ'][j] for j in range(self.data.J))
            lhs6 = sum(DUALS['i'][j] * X['lJ'][j] for j in range(self.data.J))
            if self.args["delay"] == 1:
                lhs7 = sum(sum(
                    DUALS['m'][j][t_] * X['zJ'][j, t_]
                    for t_ in range(self.data.T)
                    ) for j in range(self.data.J))
            else:
                lhs7 = sum(
                    DUALS['m'][j] * X['zJ'][j] for j in range(self.data.J)
                    )
            ax = sum([lhs1, lhs2, lhs3, lhs4, lhs5, lhs6, lhs7])
            b = sum(d*rhs for d, rhs in zip(DUALS['cuts'], CUT_RHS))
            cut = sum([ax, b])
            CUTS.append(cut)
        return sum(p*x for p, x in zip(pi_lst, CUTS))

    def backward_pass(self, sample, master=None):
        T = len(sample)
        num_cuts = 0
        for t in reversed(range(T-1)):
            s = sample[t]
            Cal_Q = []
            for s_ in self.data.state_space[t+1]:
                self.models[t+1, s_].update_rhs(
                    self.models[t, s].sol,
                    self.data.demand_mssp[t+1][s_],
                    )
                self.models[t+1, s_].m.optimize()
                if self.models[t+1, s_].m.status != gb.GRB.OPTIMAL:
                    print("Infeasibility  in the backward pass")
                    exit(0)
                else:
                    Cal_Q.append(self.models[t+1, s_].m.objval)
                    self.models[t+1, s_].get_duals()
            # STEP 2: CHECK CUT VIOLATION AND ADD CUTS
            for st in self.data.state_space[t]:
                pi_lst = list(
                    self.data.pi_mssp[t][st][st_1]
                    for st_1 in self.data.state_space[t+1]
                    )
                Cal_Q_approx = sum(p*Q for p, Q in zip(pi_lst, Cal_Q))
                if abs(Cal_Q_approx) < 1e-9:
                    cut_violated = False
                elif st == s:
                    DIFF = Cal_Q_approx - self.models[t, st].sol["theta"]
                    if DIFF/Cal_Q_approx > self.data.cut_tol:
                        cut_violated = True
                    else:
                        cut_violated = False
                else:
                    cut_violated = True
                if cut_violated:
                    num_cuts += 1
                    # Generate cuts (\sum_{s \in S_{t+1}} a_{t+1}.x+b_{t+1})
                    if self.args['method'] == 'bc' and t == 0:
                        cut_aggregated = self.generate_cut(
                            t, st, master._vars
                            )
                        master._constrs['cuts'].append(cut_aggregated)
                        master.cbLazy(
                            master._vars['theta'] >= cut_aggregated
                            )
                        master.update()
                    else:
                        cut_aggregated = self.generate_cut(
                            t, st, self.models[t, st].m._vars
                            )
                        self.models[t, st].m._constrs['cuts'].append(
                            self.models[t, st].m.addConstr(
                                self.models[t, st].m._vars['theta']
                                >= cut_aggregated
                                )
                            )
                        self.models[t, st].m.update()
                        self.models[t, st].cuts_rhs.append(
                            self.models[t, st].m._constrs['cuts'][-1].RHS
                            )
        return cut_violated, num_cuts

    def statistical_ub(self):
        self.args["cb"] = False
        self.root = True
        OBJ = []
        for i in range(self.data.n_UB_samples):
            sample = random_sample(self.data, self.args['hurricane'])
            obj = self.forward_pass(sample)
            OBJ.append(obj)
        return list([np.average(OBJ), np.std(OBJ)] +
                    [np.quantile(OBJ, i/4) for i in range(5)])

    def export_and_print(self, method):
        # Algorithm results.
        export = {
            'comp_time': self.comp_time,
            'num_paths': self.num_paths,
            'num_cuts': self.num_cuts,
            'ub_avg': self.ub[0],
            'ub_sd': self.ub[1],
            'ub_quantiles': self.ub[2:],
            }
        if method == 'bb':
            export.update({
                'lb': self.lb,
                'lb_change_rate': self.lb_change_rate,
                'lb_list': self.lb_list,
                })
        else:
            export.update({
                'MIP_gap': self.MIP_gap,
                'num_nodes': self.num_nodes,
                'master_obj': self.master_obj,
                })
        # zJ solution
        st_0 = self.data.state_space[0][0]
        Master = self.models[0, st_0]
        zJ_sol_dict = Master.get_sol()['zJ']
        if self.args['delay'] == 1:
            zJ_sol = [[0 for t in range(self.data.T)]
                    for j in range(self.data.J)]
            for key, val in zJ_sol_dict.items():
                j, t = key[0], key[1]
                zJ_sol[j][t] = int(val)
        else:
            zJ_sol = [0 for j in range(self.data.J)]
            for j, val in zJ_sol_dict.items():
                zJ_sol[j] = int(val)
        export['zJ'] = zJ_sol
        # Export
        with open(self.data.DIR[4] +
                  'sddp_{}.json'.format(
                      self.args['method'],
                      ), 'w'
                  ) as file:
            json.dump(export, file, indent=4)
        # Print
        print(f'\nSolving for {method} method. Results: \n')
        for key, val in export.items():
            print(f'{key}: {val}')

    def branch_and_bound(self):
        """
        Solve first-stage problem to optimality using Branch and Bound
        algorithm and add cuts using the respective optimal
        first-stage solutions
        """
        self.num_cuts = 0
        self.num_paths = 0
        self.lb_list = []
        self.comp_time = 0.0
        self.lb_change_rate = 1.0
        self.root = True
        itr = 0
        MAX_TIME = self.args["time_limit_train"]
        MAX_ITER = self.data.max_itr
        while itr < MAX_ITER and self.comp_time < MAX_TIME:
            itr += 1
            itr_start_time = time.time()
            sample = random_sample(self.data, self.args['hurricane'])
            self.num_paths += 1
            self.forward_pass(sample)
            cut_violated, n_cuts = self.backward_pass(sample)
            self.num_cuts += n_cuts
            if cut_violated is False:  # Redo SDDP if cut not violated at t=0.
                self.root = False
                print(f"cut not violated at itr {itr}")
            else:
                self.root = True
                self.models[0, sample[0]].m.optimize()
            self.comp_time += time.time() - itr_start_time
            self.lb = self.models[0, sample[0]].m.objVal
            self.lb_list.append(self.lb)
            # Check LB convergence
            # if itr >= self.data.n_itr_lb_rate and self.root:
            if itr >= self.data.n_itr_lb_rate:
                lb_old = self.lb_list[itr - self.data.n_itr_lb_rate]
                lb_new = self.lb
                self.lb_change_rate = (lb_new - lb_old) * 1.0/abs(lb_old)
                if self.lb_change_rate <= self.data.lb_tol:
                    break
            print('Itr: {} LB: {:.2f} Rate: {:.5f}: Time: {:.2f}'.format(
                itr, self.lb, self.lb_change_rate, self.comp_time,
                ))
        self.ub = self.statistical_ub()
        self.export_and_print(method='bb')

    def callback(self, model, where):
        if where == gb.GRB.Callback.MIPSOL:
            run_sddp = True
            count = 0
            MAX_ITER = self.data.max_itr_sddp_rerun
            while run_sddp is True and count < MAX_ITER:
                sample = random_sample(self.data, self.args['hurricane'])
                self.forward_pass(sample=sample)
                cut_violated, n_cuts = self.backward_pass(
                    sample=sample,
                    master=model,
                    )
                self.num_paths += 1
                self.num_cuts += n_cuts
                self.comp_time = time.time() - self.start_time
                if cut_violated is True:
                    run_sddp = False
                else:
                    count += 1
                    print(f"rerunning SDDP itr [{count}]")

    def branch_and_cut(self):
        """Add cutting planes from SDDP as lazy constraints to incumbent
        integer solutions of Branch and Bound tree
        data: data structure as a python class.
        """
        self.root = False
        self.comp_time = 0.0
        self.num_cuts = 0
        self.num_paths = 0

        st_0 = self.data.state_space[0][0]
        Master = self.models[0, st_0].m
        Master.setParam('OutputFlag', 1)
        Master.params.LazyConstraints = 1
        Master.setParam('TimeLimit', self.args["time_limit_train"])
        cb_func = lambda model, where : self.callback(model, where)
        self.start_time = time.time()
        Master.optimize(cb_func)
        Master.setParam('OutputFlag', 0)
        self.master_obj = Master.objVal
        self.comp_time = time.time() - self.start_time
        self.MIP_gap = Master.MIPGap
        self.num_nodes = Master.NodeCount
        self.ub = self.statistical_ub()
        self.export_and_print(method='lazy cuts')

    def in_sample_eval(self, S):
        cost = []
        st_0 = self.data.state_space[0][0]
        self.root = False
        start_time = time.time()
        test_time = 0.0
        for s in range(S):
            test_time = time.time() - start_time
            if test_time > self.args['time_limit_test']:
                break
            comp_cost = self.models[0, st_0].cost_component()
            test_sample = self.data.test_samples_from_tree[s]
            # TODO : fix json import for 'test_samples_from_tree' so
            # conversion to tuple is not necessary.
            test_sample = [tuple(v) for v in test_sample]
            self.forward_pass(test_sample)
            for t, s_ in enumerate(test_sample):
                if t > 0:
                    comp_cost_temp = self.models[t, s_].cost_component()
                    for key in comp_cost.keys():
                        comp_cost[key] += comp_cost_temp[key]
            df = pd.DataFrame(comp_cost, index=[s])
            df.rename_axis("s")
            df['Total'] = df.sum(axis=1)
            df = df.round(2)
            cost.append(df)
        result = pd.concat(cost).rename_axis('s').round(2)
        result.to_csv(
            self.data.DIR[4] + "mssp_eval_mc_tree_{}.csv".format(
                self.args['method'],
                )
            )
        summary = helper.summerize_result(result)
        summary.to_csv(
            self.data.DIR[4] + "summary_mssp_eval_mc_tree_{}.csv".format(
                self.args['method'],
                )
            )
        return cost

    def closest_cost_func_seq(self, s, heur):
        """Given an out-sample, return the closest sample path from
        the MC Tree.

        Parameters
        ----------
        s : int
            Out-of-sample index
        heur : options (1 or 2) for Ian case
            1 if closest transient/absorbing state from Tree is picked for
             transient/absorbing OOS state, respectively.
             2 if closest state from Tree is picked regardless of
             transient/absorbing characteristic of it.
        """
        # Root node is common to all scenarios.
        closest_sample_to_oos = [self.data.state_space[0][0]]
        if self.args['hurricane'] == "Ian":
            ts = self.data.ts_oos[s]  # Terminal stage
        else:
            ts = self.data.T - 1
        for t in range(1, ts + 1):
            cat = self.data.cat_scen_oos[s][t]
            # Hurricane category under OOS 's'
            if self.args['hurricane'] == 'Florence':
                oos_track_err = self.data.oos_err['track'][s][t]
                min_track_err = 1e10
                for state in self.data.state_space[t]:
                    tree_track_err = state[0]
                    abs_track_diff = abs(tree_track_err - oos_track_err)
                    if abs_track_diff < min_track_err:
                        min_track_err = abs_track_diff
                        closest_track_err = tree_track_err
                closest_sample_to_oos.append((closest_track_err, cat))
            else:
                oos_along_err = self.data.oos_err['along'][s][t]
                oos_cross_err = self.data.oos_err['cross'][s][t]
                oos_point = helper.transform_gis_random_landfall(
                    data=self.data, t=t, xi=[oos_along_err, oos_cross_err]
                    )
                dist_max = 1e10
                if heur == 1:
                    # Only consider transient states at t < ts
                    # Only consider absorbing states at t = ts
                    prob = self.data.pi_mssp[t-1][closest_sample_to_oos[-1]]
                    state_space = list(
                        state for state in self.data.state_space[t]
                        if self.data.absorb_mssp[t][state] is (
                            False if t < ts else True
                            ) and prob[state] > 0.0
                        )
                    if len(state_space) < 1:
                        state_space = list(
                            state for state in self.data.state_space[t]
                            if self.data.absorb_mssp[t][state] is (
                                False if t < ts else True
                                )
                            )
                    if len(state_space) < 1:
                        state_space = self.data.state_space[t]
                else:
                    state_space = self.data.state_space[t]
                for state in state_space:
                    tree_point = helper.transform_gis_random_landfall(
                        data=self.data, t=t, xi=state[:2]
                        )
                    dist = helper.distMiles(oos_point, tree_point)
                    if dist < dist_max:
                        dist_max = dist
                        xi_closest = state[:2]
                closest_sample_to_oos.append((*xi_closest, cat))
        absorbed = False
        cost_func_seq = []
        for t, state in enumerate(closest_sample_to_oos):
            if heur == 1 or absorbed is False:
                cost_func_seq.append(self.models[t, state])
                if self.data.absorb_mssp[t][state] is True:
                    absorbed = True
                    continue
            if absorbed is True:
                cost_func_seq.append(cost_func_seq[-1])
        return closest_sample_to_oos, cost_func_seq

    def out_sample_eval(self, n_oos, heur):
        """Get out-of-sample cost after solving SDDP.

        n_oos : int
            Number of out-of-samples to test on.
        heur : option (1 or 2)
            1 if OOS is evaluated on the closest cost functions from the
            MC tree until absorbing state. 2 if OOS is evaluated
            on the closest transient nodes of MC tree until ts-1
            and on the closest absorbing cost function at ts.
        """
        oos_cost_components = []
        st_0 = self.data.state_space[0][0]
        start_time = time.time()
        test_time = 0.0
        for s in range(n_oos):
            test_time = time.time() - start_time
            if test_time > self.args['time_limit_test']:
                break
            _, cost_func = self.closest_cost_func_seq(s=s, heur=heur)
            for t, model in enumerate(cost_func):
                if t == 0:
                    cost_components = self.models[0, st_0].cost_component()
                    sol = self.models[0, st_0].get_sol()
                else:
                    try:
                        xi = self.data.demand_oos[s][t]
                    except KeyError:
                        print(s, t, len(cost_func), self.data.demand_oos[s])
                        exit(0)
                    model.update_rhs(x_sol=sol, XI=xi)
                    model.m.optimize()
                    sol = model.get_sol(state_vars_only=False, cb=False)
                    cost_t = model.cost_component()
                    for key, val in cost_t.items():
                        cost_components[key] += val
            # Create DataFrame of OOS cost components.
            df_comp = pd.DataFrame(cost_components, index=[s])
            df_comp['Total'] = df_comp.sum(axis=1)
            df_comp.rename_axis("s")
            oos_cost_components.append(df_comp)
        # Export results of all out-of-samples
        result = pd.concat(oos_cost_components).rename_axis('s').round(2)
        result.to_csv(
            self.data.DIR[4] + "mssp_eval_oos_heur{}_{}.csv".format(
                heur, self.args['method'],
                ))
        summary = helper.summerize_result(result)
        summary.to_csv(
            self.data.DIR[4] + "summary_mssp_eval_oos_heur{}_{}.csv".format(
                heur, self.args['method'],
                ))
        return oos_cost_components


def solve_mssp(data, **kwargs):
    args = kwargs
    args["cb"] = True if args["method"] == "bc" else False
    models = make_models(data=data, args=args)
    sddp = SDDP(data, models, **args)
    if args["method"] == "bb":
        sddp.branch_and_bound()
    elif args['method'] == 'bc':
        sddp.branch_and_cut()
    else:
        print('error! method should be \'bb\' or \'bc\' for MSSP model')
        exit(0)
    if args['eval'] != 'none':
        cost_insample = sddp.in_sample_eval(args["n_oos"])
        cost_oos = sddp.out_sample_eval(args["n_oos"], args["oos_heur"])
