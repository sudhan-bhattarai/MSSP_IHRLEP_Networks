import string
import time
import numpy as np
import pandas as pd
import gurobipy as gb
import json
import helper
import copy

import data_forecast_error_scenarios as fe_scen

BI = gb.GRB.BINARY
CT = gb.GRB.CONTINUOUS
INF = gb.GRB.INFINITY
rhs = gb.GRB.Attr.RHS
optimal = gb.GRB.OPTIMAL


class Model2SSP:
    def __init__(self, data, init, **args):
        self.args = args
        self.data = data
        self.init = init
        # Variable names
        self.master_vars = ["zJ", "lJ", "theta"]
        self.recourse_vars = ["eI", "eJ", "xKJ", "xJJ", "y", "u", "h", "g"]
        alphabets = list(string.ascii_lowercase)
        self.constr_name = alphabets[:12]  # From 'a' to 'l'.
        self.master = gb.Model("Master")
        self.master.setParam("OutputFlag", 0)
        self.master._vars = {
            var: {} for var in self.master_vars + self.recourse_vars
            }  # Local copy of recourse vars are the first-stage vars.
        self.master._constrs = {c: {} for c in self.constr_name}
        self.master._constrs["1st_stg"] = list()
        self.master._constrs["cuts"] = list()
        self.sub_prob = list(
            gb.Model(f"scen{s}") for s in range(data.S)
            )
        # deterministic "T" else make "S" models
        for s, m in enumerate(self.sub_prob):
            m.setParam("OutputFlag", 0)
            m._s = s
            m._vars = {var: {} for var in self.recourse_vars}
            m._constrs = {c: {} for c in self.constr_name}
            m._duals = {c: {} for c in self.constr_name}
        # Benders initialization
        self.lb = 0.0
        self.ub = 1e15
        self.comp_time = 0.0

    def master_prob(self):
        M = self.master
        t_ = self.init.t
        var = self.master._vars
        constrs = self.master._constrs
        for j in range(self.data.J):
            for t in self.init.T_PRIME:
                var['zJ'][j, t] = M.addVar(
                    vtype=BI, lb=0, ub=1.0, name=f'zJ{j}{t}'
                    )
                var['lJ'][j, t] = M.addVar(
                    vtype=CT, lb=0, ub=INF, name=f'lJ{j}{t}'
                    )
        M.update()
        for i in range(self.data.I):
            var['u'][i, t_] = M.addVar(
                vtype=CT, lb=0, ub=INF, name=f'u{i}{t_}'
                )
            if self.args['demand_opt'] == 1:
                var['eI'][i, t_] = M.addVar(
                    vtype=CT, lb=0, ub=INF, name=f'eI{i}{t_}'
                    )
            for j in range(self.data.J):
                var['y'][i, j, t_] = M.addVar(
                    vtype=CT, lb=0, ub=INF, name=f'y{i}{j}{t_}'
                    )
        for j in range(self.data.J):
            var['eJ'][j, t_] = M.addVar(
                vtype=CT, lb=0, ub=INF, name=f'eJ{j}{t_}'
                )
            var['g'][j, t_] = M.addVar(
                vtype=CT, lb=0, ub=INF, name=f'g{j}{t_}'
                )
            var['h'][j, t_] = M.addVar(
                vtype=CT, lb=0, ub=INF, name=f'h{j}{t_}'
                )
            for j_ in range(self.data.J):
                var['xJJ'][j, j_, t_] = M.addVar(
                    vtype=CT, lb=0, ub=INF, name=f'xJJ{j}{j_}{t_}'
                    )
        for k in range(self.data.K):
            for j in range(self.data.J):
                var['xKJ'][k, j, t_] = M.addVar(
                    vtype=CT, lb=0, ub=INF, name=f'xKJ{k}{j}{t_}'
                    )
        M.update()
        # Theta variable
        for s in range(self.data.S):
            var['theta'][s] = M.addVar(
                vtype=CT, lb=0, ub=INF, name=f'theta{s}'
                )
        # ------------------ first-stage constraints -------------------------
        for t in self.init.T_PRIME:
            for j in range(self.data.J):
                constrs["1st_stg"].append(M.addConstr(
                    var['lJ'][j, t] <=
                    self.data.phi * self.data.q_J[j] * var['zJ'][j, t]
                    ))
                # ---------- Opening of SPs decisions ------------------------
                if self.args["first_stg_opt"] == 1:
                    # At t' > 0; use integer solutions from SLAM at t'=0
                    if t_ == 0:
                        if self.args['model'] == 'rh':
                            constrs["1st_stg"].append(M.addConstr(
                                    var['zJ'][j, t]
                                    == self.init.zJ_hat[j, t]
                                    ))
                        else:
                            if t > t_:
                                if self.args["delay"] == 1:
                                    constrs["1st_stg"].append(M.addConstr(
                                        var['zJ'][j, t]
                                        >= var['zJ'][j, t-1]
                                        ))
                                else:
                                    constrs["1st_stg"].append(M.addConstr(
                                        var['zJ'][j, t]
                                        == var['zJ'][j, t-1]
                                        ))
                    else:
                        constrs["1st_stg"].append(M.addConstr(
                            var['zJ'][j, t] == self.init.zJ_hat[j, t]
                            ))
                elif self.args["first_stg_opt"] == 2:
                    constrs["1st_stg"].append(
                        M.addConstr(var['zJ'][j, t] == 1)
                        )
                # TODO: fill up adaptive 'z' for RH case
                # If adaptive SPs opening is allowed in RH
                # else:
                #     if t == t_:
                #         constrs["1st_stg"].append(M.addConstr(
                #             var['zJ'][j, t] >= self.init.zJ_hat[j, t_]
                #             ))
                #     else:
                #         constrs["1st_stg"].append(M.addConstr(
                #             var['zJ'][j, t] >= var['zJ'][j, t-1]
                #             ))
        # TODO Avoid two of the following constraints for 
        # demand_opt = 2.
        for i in range(self.data.I):
            constrs["1st_stg"].append(M.addConstr(
                gb.quicksum(var['y'][i, j, t_] for j in range(self.data.J))
                + var['u'][i, t_] ==
                self.init.demand_tprime[i] *
                (self.init.eI_hat[i, t_] if self.args['demand_opt'] == 1 else 1)
                ))
            if self.args['demand_opt'] == 1:
                constrs["1st_stg"].append(M.addConstr(
                    var['eI'][i, t_] ==
                    self.init.eI_hat[i, t_]
                    - gb.quicksum(var['y'][i, j, t_]
                                for j in range(self.data.J))
                    ))
        for j in range(self.data.J):
            constrs["1st_stg"].append(M.addConstr(
                var['xJJ'][j, j, t_] == 0
                ))  # can not move items from a SP to itself
            constrs["1st_stg"].append(M.addConstr(
                gb.quicksum(var['y'][i, j, t_]
                            for i in range(self.data.I))
                + self.init.eJ_hat[j, t_] ==
                var['eJ'][j, t_]
                ))
            constrs["1st_stg"].append(M.addConstr(
                var['eJ'][j, t_] <=
                self.data.q_J[j] * var['zJ'][j, t_]
                ))
            constrs["1st_stg"].append(M.addConstr(
                self.init.lJ_hat[j, t_]
                + gb.quicksum(var['xKJ'][k, j, t_]
                              for k in range(self.data.K))
                + gb.quicksum(var['xJJ'][j_, j, t_]
                              for j_ in range(self.data.J))
                - gb.quicksum(var['xJJ'][j, j_, t_]
                              for j_ in range(self.data.J))
                - self.data.phi * var['eJ'][j, t_]
                + var['g'][j, t_]
                - var['h'][j, t_] ==
                var['lJ'][j, t_]
                ))
            constrs["1st_stg"].append(M.addConstr(
                gb.quicksum(
                    var['xJJ'][j, j_, t_]
                    for j_ in range(self.data.J)
                    )
                <= self.init.lJ_hat[j, t_]
                + var['g'][j, t_]
                - self.data.phi * var['eJ'][j, t_]
                ))
            constrs["1st_stg"].append(M.addConstr(
                self.init.lJ_hat[j, t_]
                + var['g'][j, t_]
                - self.data.phi * var['eJ'][j, t_] <=
                self.data.phi * self.data.q_J[j] * var['zJ'][j, t_]
                ))
        M.update()
        # ----------------------- first-stage objective -------------------
        objs_dict = self.obj(M, option=1)
        obj = sum([sum(cost_vector) for cost_vector in objs_dict.values()])
        M.setObjective(
            obj  # first-stage obj
            + gb.quicksum(
                self.data.p[s] * var['theta'][s]
                for s in range(self.data.S)
                )  # Multi-cut objective related to \theta_s
            - sum(1.0/self.data.S *
                  sum(sum(self.data.c_invR_J[j] * var['lJ'][j, t]
                          for j in range(self.data.J))
                      for t in self.init.T_PRIME[self.init.ts[s] + 1:]
                      ) for s in range(self.data.S)
                  ),  # Return on inventory as reimbursement cost.
            gb.GRB.MINIMIZE
            )
        M.update()

    def obj(self, m, option):
        """Build objective or return the cost after optimization.

        option: int (1 or 2)
            1 if constructing obj function; = 2 if getting result
            after optimization.
        extended: bool
            True if constructing objective for an extended formulation.
        """
        var = m._vars if option == 1 else self.solution(m)
        if m == self.master:
            periods = [self.init.t]
        else:
            periods = self.init.T_2ND[m._s]
        names = ["Fixed", "Relief Inventory", "Evacuee Inventory", "Penalty",
                 "Emergency", "Relief Purchase", "Relief Transportation",
                 "Evacuee Transportation", "Relief Dumping"]
        comp_cost = {name: [] for name in names}
        # One copy of local cost at t=1 for the Master problem
        # T copies of local cost at t=t for Sub-problems
        for t in periods:
            comps = [
                # Evacuee Inventory
                sum(self.data.c_invE_J[j] * var['eJ'][j, t]
                    for j in range(self.data.J)),
                # Penalty
                sum(self.data.c_PE[i] * var['u'][i, t]
                    for i in range(self.data.I)),
                # Emergency transportation
                sum(self.data.c_G_J[j] * var['g'][j, t]
                    for j in range(self.data.J)),
                # Relief purchase
                sum(sum(self.data.c_P_K[k] * var['xKJ'][k, j, t]
                        for j in range(self.data.J))
                    for k in range(self.data.K)),
                # Relief transportation
                sum(sum(self.data.c_R_JJ[j][j_] * var['xJJ'][j, j_, t]
                        for j in range(self.data.J))
                    for j_ in range(self.data.J))
                + sum(sum(self.data.c_R_KJ[k][j] * var['xKJ'][k, j, t]
                          for j in range(self.data.J))
                      for k in range(self.data.K)),
                # Evacuee transportation
                sum(sum(self.data.c_E_IJ[i][j] * var['y'][i, j, t]
                    for i in range(self.data.I))
                    for j in range(self.data.J)),
                # Relief dumping
                sum(self.data.c_H_J[j] * var['h'][j, t]
                    for j in range(self.data.J))
                ]
            for name, comp in zip(names[2:], comps):
                comp_cost[name].append(comp)

        if m == self.master:
            for t in self.init.T_PRIME:
                # First-stage inventory costs
                comp_cost["Relief Inventory"].append(
                    sum(self.data.c_invR_J[j] * var['lJ'][j, t]
                        for j in range(self.data.J))
                    )
                # Update first-stage fixed costs
                if self.init.t == 0:
                    if t == 0:
                        # One time activation cost
                        comp_cost["Fixed"].append(
                            sum(self.data.c_F_J[j] * var['zJ'][j, t]
                                for j in range(self.data.J))
                            )
                    else:
                        # One time activation cost
                        comp_cost["Fixed"].append(
                            sum(self.data.c_F_J[j] *
                                (var['zJ'][j, t] - var['zJ'][j, t-1])
                                for j in range(self.data.J))
                            )
                    # Maintenance cost per period
                    comp_cost["Fixed"][t] += sum(
                        self.data.c_F_J_var[j] * var['zJ'][j, t]
                        for j in range(self.data.J)
                        )
        return comp_cost

    def make_sub_prob(self, s, train):
        """train = bool; True if extended model is used in training. False if
           sub_prob is used to create OOS test model
        """
        extended = self.args["method"] == "ext" and train
        M = self.master if extended else self.sub_prob[s]
        if extended is True:
            M._vars[s] = {v: {} for v in self.recourse_vars}
            M._constrs[s] = {c: {} for c in self.constr_name}
            var = M._vars[s]
            constr = M._constrs[s]
        else:
            var = M._vars
            constr = M._constrs
        # ---------------- second-stage variables ----------------------------
        for i in range(self.data.I):
            if self.args['demand_opt'] == 1:
                for t in self.init.T_PRIME:
                    var['eI'][i, t] = M.addVar(
                        vtype=CT, lb=0, ub=INF, name=f'eI{i}{t}'
                        )
            for t in self.init.T_2ND[s]:
                var['u'][i, t] = M.addVar(
                    vtype=CT, lb=0, ub=INF, name=f'u{i}{t}'
                    )
                for j in range(self.data.J):
                    var['y'][i, j, t] = M.addVar(
                        vtype=CT, lb=0, ub=INF, name=f'y{i}{j}{t}'
                        )
        for j in range(self.data.J):
            for t in self.init.T_PRIME:
                var['eJ'][j, t] = M.addVar(
                    vtype=CT, lb=0, ub=INF, name=f'eJ{j}{t}'
                    )
            for t in self.init.T_2ND[s]:
                var['g'][j, t] = M.addVar(
                    vtype=CT, lb=0, ub=INF, name=f'g{j}{t}'
                    )
                var['h'][j, t] = M.addVar(
                    vtype=CT, lb=0, ub=INF, name=f'h{j}{t}'
                    )
                for j_ in range(self.data.J):
                    var['xJJ'][j, j_, t] = M.addVar(
                        vtype=CT, lb=0, ub=INF, name=f'xJJ{j}{j_}{t}'
                        )
        for k in range(self.data.K):
            for t in self.init.T_2ND[s]:
                for j in range(self.data.J):
                    var['xKJ'][k, j, t] = M.addVar(
                        vtype=CT, lb=0, ub=INF, name=f'xKJ{k}{j}{t}'
                        )
            M.update()
        # ------------------ second-stage constraints -------------------
        for i in range(self.data.I):
            for t in self.init.T_2ND[s]:
                constr['a'][i, t] = M.addConstr(
                    gb.quicksum(var['y'][i, j, t] for j in range(self.data.J))
                    + var['u'][i, t] ==
                    self.init.demand_insample[s][t][i] *
                    (var['eI'][i, t-1] if self.args['demand_opt'] == 1 else 1)
                    )
        if self.args['demand_opt'] == 1:
            constr['b'] = M.addConstrs((
                var['eI'][i, t] == var['eI'][i, t-1]
                - gb.quicksum(var['y'][i, j, t] for j in range(self.data.J))
                for i in range(self.data.I) for t in self.init.T_2ND[s]
            ))
            constr['c'] = M.addConstrs((
                var['eI'][i, self.init.t] == (M._vars['eI'][i, self.init.t]
                                              if extended else 0)
                for i in range(self.data.I) for t in [self.init.t]
            ))
        for j in range(self.data.J):
            constr['e'][j, self.init.t] = M.addConstr(
                var['eJ'][j, self.init.t]
                == (M._vars['eJ'][j, self.init.t]
                    if extended else 0)
                )

            for t in self.init.T_2ND[s]:
                M.addConstr(var['xJJ'][j, j, t] == 0)
                # can not move items from a SP to itself
                constr['d'][j, t] = M.addConstr(
                    gb.quicksum(
                        var['y'][i, j, t]
                        for i in range(self.data.I)
                        )
                    + var['eJ'][j, t - 1]
                    == var['eJ'][j, t]
                    )
                constr['f'][j, t] = M.addConstr(
                    -var['eJ'][j, t]
                    >= (-self.data.q_J[j] * M._vars['zJ'][j, t]
                        if extended else 0.0)
                    )
                constr['g'][j, t] = M.addConstr(
                    gb.quicksum(var['xKJ'][k, j, t]
                                for k in range(self.data.K))
                    + gb.quicksum(var['xJJ'][j_, j, t]
                                  for j_ in range(self.data.J))
                    - gb.quicksum(var['xJJ'][j, j_, t]
                                  for j_ in range(self.data.J))
                    - self.data.phi * var['eJ'][j, t]
                    + var['g'][j, t] - var['h'][j, t]
                    == (M._vars['lJ'][j, t] - M._vars['lJ'][j, t - 1]
                        if extended else 0.0)
                    )
                constr['h'][j, t] = M.addConstr(
                    - gb.quicksum(var['xJJ'][j, j_, t]
                                  for j_ in range(self.data.J))
                    + var['g'][j, t]
                    - self.data.phi * var['eJ'][j, t]
                    >= (-M._vars['lJ'][j, t - 1] if extended else 0)
                    )
                constr['i'][j, t] = M.addConstr(
                    -var['g'][j, t]
                    + self.data.phi * var['eJ'][j, t]
                    >= ((M._vars['lJ'][j, t - 1]
                         - self.data.phi * self.data.q_J[j] *
                         M._vars['zJ'][j, t]
                         ) if extended else 0.0)
                    )
        M.update()
        objs_dict = self.obj(self.sub_prob[s], option=1)
        obj = sum([sum(cost_vector) for cost_vector in objs_dict.values()])
        if extended is True:
            self.master.setObjective(
                self.master.getObjective() + 1.0/self.data.S * obj,
                gb.GRB.MINIMIZE
                )
            self.master.update()
        else:
            self.sub_prob[s].setObjective(obj, gb.GRB.MINIMIZE)
            self.sub_prob[s].update()

    def solution(self, m):
        """m : Gurobi Model (variables should be in m._vars)"""
        result = {key: {} for key in m._vars.keys()}
        for v_name in m._vars.keys():
            for key, var in m._vars[v_name].items():
                try:
                    result[v_name][key] = var.x
                except AttributeError:
                    if v_name == 'zJ':
                        result[v_name][key] = round(m.cbGetSolution(var))
                    else:
                        result[v_name][key] = m.cbGetSolution(var)
        return result

    def get_dual(self, s):
        t_ = self.init.t  # starting period (t_prime) of the SLAM
        dual = self.sub_prob[s]._duals
        constr = self.sub_prob[s]._constrs
        if self.args['demand_opt'] == 1:
            dual["c"] = list(constr["c"][i, t_].pi for i in range(self.data.I))
        dual["e"] = list(constr["e"][j, t_].pi for j in range(self.data.J))
        for t in self.init.T_2ND[s]:
            for j in range(self.data.J):
                for c in ["a", "f", "g", "h", "i"]:
                    dual[c][j, t] = constr[c][j, t].pi

    def update_rhs(self, master_sol, m, xi):
        """
        master_sol: first-stage solution
        m: sub_prob Model for a scenario "s"
        xi: (Array of T by I), Demand data for sub_prob "s"
        """
        t_ = self.init.t
        s = m._s
        if self.args['demand_opt'] == 1:
            for i in range(self.data.I):
                for t in self.init.T_2ND[s]:
                    m.remove(m._constrs['a'][i, t])
            m.update()
            for i in range(self.data.I):
                for t in self.init.T_2ND[s]:
                    m._constrs['a'][i, t] = m.addConstr(
                        gb.quicksum(m._vars['y'][i, j, t]
                                    for j in range(self.data.J))
                        + m._vars['u'][i, t]
                        == xi[t][i] * m._vars['eI'][i, t-1])
            m.update()

            for i in range(self.data.I):
                m._constrs['c'][i, t_].setAttr(rhs, master_sol['eI'][i, t_])

        for j in range(self.data.J):
            m._constrs['e'][j, t_].setAttr(rhs, master_sol['eJ'][j, t_])

        for t in self.init.T_2ND[s]:
            for j in range(self.data.J):
                lj_t_minus_1 = master_sol['lJ'][j, t-1]
                lj_t = master_sol['lJ'][j, t]
                zj_t = master_sol['zJ'][j, t]
                qj = self.data.q_J[j]

                m._constrs['f'][j, t].setAttr(rhs, - qj * zj_t)
                m._constrs['g'][j, t].setAttr(rhs, lj_t - lj_t_minus_1)
                m._constrs['h'][j, t].setAttr(rhs, - lj_t_minus_1)

                rhs_ci = min(0.0, lj_t_minus_1 - self.data.phi * qj * zj_t)
                m._constrs['i'][j, t].setAttr(rhs, rhs_ci)
        m.update()

    def sub_prob_test(self, xi_test, rnn_test):
        # keep only one sub-problem for the test of an 'oos'
        if len(self.sub_prob) > self.data.S:
            self.sub_prob.remove(self.sub_prob[-1])
        self.sub_prob.append(gb.Model("test"))
        self.sub_prob[-1].setParam("OutputFlag", 0)
        self.sub_prob[-1]._vars = {var: {} for var in self.recourse_vars}
        self.sub_prob[-1]._constrs = {c: {} for c in self.constr_name}
        self.sub_prob[-1]._duals = {c: {} for c in self.constr_name}
        self.sub_prob[-1]._s = self.data.S
        self.sub_prob[-1].reset()
        self.make_sub_prob(s=self.data.S, train=False)
        sol, _, _, obj_master = read_first_stage_result(self.data, self.args)
        self.update_rhs(sol, self.sub_prob[-1], xi_test)
        self.sub_prob[-1].optimize()
        if self.sub_prob[-1].status != gb.GRB.OPTIMAL:
            print("error on test sample")
            exit(0)
        obj_oos = {}
        ts = len(xi_test) - 1
        obj_sub = self.obj(m=self.sub_prob[-1], option=2)
        for key, val in obj_master.items():
            obj_oos[key] = val + sum(obj_sub[key])
        # reimbursement cost
        obj_oos["Relief Inventory"] -= sum(sum(
            self.data.c_invR_J[j] * sol['lJ'][j, t]
            for j in range(self.data.J)
            ) for t in self.init.T_PRIME[ts + 1:])
        # total cost
        obj_oos['Total'] = sum(obj_oos.values())
        # solution (added on July 24, 2024) begins -----------------------
        sol_test_model = self.solution(self.sub_prob[-1])
        xKJ = {j[1:]: val for j, val in sol_test_model['xKJ'].items()}
        sol_temp = {
            'xKJ': xKJ,
            'eJ': sol_test_model['eJ'],
            'eI': sol_test_model['eI'],
        }
        # solution (added on July 24, 2024) ends -------------------------
        return obj_oos, sol_temp


class SolveSLAM:
    def __init__(self, data, init, **args):
        self.data = data
        self.init = init
        self.args = args
        self.slam = Model2SSP(self.data, self.init, **args)
        self.slam.master_prob()
        for s in range(self.data.S):
            self.slam.make_sub_prob(s, train=True)
        self.itr = 0
        self.num_cuts = 0
        self.comp_time = 0.0
        self.gap = 1.0
        self.lb = 0.0
        self.ub = 1e15

    def extended(self):
        self.slam.master_prob()
        for s in range(self.data.S):
            self.slam.make_sub_prob(s, train=True)
        start_time = time.time()
        self.master.setParam('OutputFlag', 1)
        self.master.setParam('TimeLimit', self.args['time_limit_train'])
        self.slam.master.optimize()
        self.comp_time = time.time() - start_time
        print("Obj: ", self.slam.master.objVal)

    def cut(self, X, s):
        """
        s = 1, ..., S = scenario
        eval_ = True if evaluating the cut for first-stage sol; = False if
        generating cut for first-stage vars
        """
        dual = self.slam.sub_prob[s]._duals
        sec_stg_periods = self.init.T_2ND
        t = self.init.t
        ax = list((
            sum(dual['e'][j] * X['eJ'][j, t] for j in range(self.data.J)),
            sum(sum(dual['f'][j, t] * (-self.data.q_J[j] * X['zJ'][j, t])
                    for j in range(self.data.J)) for t in sec_stg_periods[s]),
            sum(sum(dual['g'][j, t] * (X['lJ'][j, t] - X['lJ'][j, t-1])
                    for j in range(self.data.J))for t in sec_stg_periods[s]),
            sum(sum(dual['h'][j, t] * (-X['lJ'][j, t - 1])
                    for j in range(self.data.J)) for t in sec_stg_periods[s]),
            sum(sum(dual['i'][j, t] * (
                X['lJ'][j, t-1] -
                self.data.phi * self.data.q_J[j] * X['zJ'][j, t]
                ) for j in range(self.data.J)) for t in sec_stg_periods[s]),
            ))
        if self.args['demand_opt'] == 1:
            ax = ax + [sum(dual['c'][i] * X['eI'][i, t]
                           for i in range(self.data.I))]
        else:
            ax = ax + [sum(sum(
                dual['a'][i, t] * self.init.demand_insample[s][t][i]
                for i in range(self.data.I)
                ) for t in sec_stg_periods[s])]
        return [sum(ax), ax]

    def export_and_print(self, method, write):
        export = {
            'ub': self.ub,
            'num_cuts': self.num_cuts,
            'comp_time': self.comp_time,
            }
        if method == 'bb':
            export.update({
                'gap': self.gap,
                'lb': self.lb,
                'num_itr': self.itr,
            })
        else:
            export.update({
                'obj': self.slam.master.objVal,
                'MIP_gap': self.slam.master.MIPGap,
                'num_nodes': self.slam.master.NodeCount,
            })
        # Export relevant first-stage solutions.
        sol_dict = self.slam.solution(self.slam.master)
        sol = {}
        for name, vals in sol_dict.items():
            if name in ['zJ', 'lJ']:
                sol[name] = [[0 for t in range(self.data.T)]
                             for j in range(self.data.J)]
                for (j, t), val in vals.items():
                    val = int(val) if name == 'zJ' else val
                    sol[name][j][t] = max(val, 0)
            elif name in ['eI', 'eJ']:
                index = name + '_t0'
                sol[index] = [0 for j in range(self.data.J)]
                for (j, t), val in vals.items():
                    sol[index][j] = max(val, 0)
        export.update(sol)
        cost_t0 = {}  # Export cost at t=0. It will be used in RH method.
        obj_all = self.slam.obj(self.slam.master, option=2)
        for name, cost in obj_all.items():
            if name == 'Fixed':  # Total fixed cost is paid at t=0.
                cost_t0['Fixed'] = sum(obj_all['Fixed'])
            else:
                cost_t0[name] = obj_all[name][0]
        export.update({'cost_t0': cost_t0})
        print(f'\n Static 2ssp solved with {method} method. Results: \n')
        for key, val in export.items():
            print(f'{key}: {val}')
        if write is True:
            with open(self.data.DIR[4] + '{}_{}_{}_demand_opt{}.json'.format(
                self.args['model'],
                method,
                self.args['eval'],
                self.args['demand_opt'],
            ), 'w') as file:
                json.dump(export, file, indent=4)

    def benders(self):
        """Naive Benders decomposition where Master problem (MIP)
        is solved to optimality at every iteration (no lazy constraints)
        """
        while (self.gap > self.data.lb_tol and
               self.comp_time < self.args["time_limit_train"]):
            start_time = time.time()
            self.itr = self.itr + 1
            self.slam.master.update()
            self.slam.master.optimize()
            if self.slam.master.status != gb.GRB.OPTIMAL:
                print('error! infeasible master problem')
                self.slam.master.write('temp.lp')
                exit(0)
            master_sol = self.slam.solution(self.slam.master)
            self.lb = self.slam.master.objVal
            if self.init.last_stage:
                break
            else:
                temp_ub = self.solve_sub_prob(master_sol)
                if temp_ub < self.ub:
                    self.ub = temp_ub
                self.gap = (self.ub - self.lb)/max(abs(self.ub), 1e-8)
                self.comp_time += time.time() - start_time
                if self.gap <= self.data.lb_tol:
                    print('lb = {} ub = {} gap = {}'.format(
                        self.lb, self.ub, self.gap)
                        )
                    break
                else:
                    for scen in range(self.data.S):
                        cut_lhs = master_sol['theta'][scen]
                        cut_rhs, _ = self.cut(master_sol, s=scen)
                        if ((cut_rhs-cut_lhs > self.data.cut_tol) and
                            ((cut_rhs-cut_lhs) * 1.0/(max(abs(cut_rhs), 1e-8))
                             > self.data.cut_tol)):
                            cut, _ = self.cut(self.slam.master._vars, s=scen)
                            self.slam.master._constrs["cuts"].append(
                                self.slam.master.addConstr(
                                    self.slam.master._vars['theta'][scen]
                                    >= cut
                                    )
                                )
                            self.num_cuts += 1
                    print("LB = {:.2f} UB = {:.2f} gap = {:.3f}%".format(
                        self.lb, self.ub, self.gap * 100
                        ) + " time={:.2f}".format(self.comp_time))

    def solve_sub_prob(self, master_sol):
        try:
            get_ub = True
            ub = self.slam.master.objVal - sum(
                self.data.p[s] * master_sol['theta'][s]
                for s in range(self.data.S)
                )
        except AttributeError:
            get_ub = False
        for s in range(self.data.S):
            self.slam.update_rhs(
                master_sol,
                m=self.slam.sub_prob[s],
                xi=self.init.demand_insample[s],
                )
            self.slam.sub_prob[s].optimize()
            if self.slam.sub_prob[s].status != optimal:
                print('error! infeasible sub problem')
                exit(0)
            else:
                self.slam.get_dual(s)
                if get_ub is True:
                    ub += self.data.p[s] * self.slam.sub_prob[s].objVal
        return ub if get_ub is True else None

    def callback(self, model, where):
        if where == gb.GRB.Callback.MIPSOL:
            master_sol = self.slam.solution(model)
            self.solve_sub_prob(master_sol=master_sol)
            for scen in range(self.data.S):
                cut_lhs = master_sol["theta"][scen]
                cut_rhs, _ = self.cut(X=master_sol, s=scen)
                cut_gap = cut_rhs - cut_lhs
                stop_cond = cut_gap / max(abs(cut_rhs), 1e-8)
                if stop_cond > self.data.cut_tol:
                    self.num_cuts += 1
                    cut, _ = self.cut(X=model._vars, s=scen)
                    model.cbLazy(model._vars['theta'][scen] >= cut)

    def benders_lazy(self):
        self.slam.master_prob()
        for s in range(self.data.S):
            self.slam.make_sub_prob(s, train=True)
        self.slam.master.params.LazyConstraints = 1
        self.slam.master.setParam('OutputFlag', 1)
        self.slam.master.setParam('TimeLimit', self.args['time_limit_train'])
        cb_func = lambda model, where: self.callback(model, where)
        start_time = time.time()
        self.slam.master.optimize(cb_func)
        self.comp_time = time.time() - start_time
        master_sol = self.slam.solution(self.slam.master)
        self.ub = self.solve_sub_prob(master_sol)

    def solve(self):
        if self.args["method"] == "ext":
            self.extended()
            # solve the model using extended formulation
        elif self.args["method"] == "bb":
            self.benders()
            # solve the model using benders decomposition
        elif self.args["method"] == "bc":
            self.benders_lazy()
            # benders with lazy cuts


def read_first_stage_result(data, args):
    """After solving the static 2SSP model, read the first stage result."""
    with open(
        data.DIR[4] + '{}_{}_{}_demand_opt{}.json'.format(
            args['model'], args['method'], args['eval'], args['demand_opt'],
            ),
        'r') as file:
        static_2ssp_result = json.load(file)
    # TODO  remove sol_t0 use static_2ssp_result directly instead.
    sol_t0 = {
        'zJ': np.array(static_2ssp_result['zJ']),
        'lJ': np.array(static_2ssp_result['lJ']),
        'eI': {(i, 0): static_2ssp_result['eI_t0'][i]
               for i in range(data.I)},
        'eJ': {(j, 0): static_2ssp_result['eJ_t0'][j]
               for j in range(data.J)},
        }
    cost_t0 = static_2ssp_result['cost_t0']
    comp_time = static_2ssp_result['comp_time']
    # compute mater model's objective
    master_obj = copy.deepcopy(cost_t0)
    inv_cost = np.matmul(sol_t0['lJ'].transpose(), data.c_invR_J).sum()
    inv_cost = inv_cost.round(2)
    master_obj['Relief Inventory'] = inv_cost
    return sol_t0, cost_t0, comp_time, master_obj


class InitSLAM:
    def __init__(self, data):
        """ Initial conditions for SLAM at t=0 """
        self.data = data
        self.lJ_hat = np.zeros([data.J, data.T])
        self.zJ_hat = np.zeros([data.J, data.T]).astype('int')
        self.lK_hat = np.zeros([data.K, data.T])
        self.eJ_hat = np.zeros([data.J, data.T])
        self.eI_hat = np.zeros([data.I, data.T])
        self.eI_hat[:, 0] = data.DP_POP[:data.I]

    def update_init(self, t, sol, first_stage_MILP=True):
        for j in range(self.data.J):
            self.lJ_hat[j, t+1] = sol['lJ'][j, t]
            self.eJ_hat[j, t+1] = sol['eJ'][j, t]
            if t == 0:
                for t_ in range(self.data.T):
                    self.zJ_hat[j, t_] = sol['zJ'][j, t_]
        for i in range(self.data.I):
            self.eI_hat[i, t+1] = sol['eI'][i, t]


class SolveStaticTwoStage:
    """Do the out-of-sample test of static two-stage model at t=0.
    Read the first-stage solution from the results of solving two-stage model.
    Solve a deterministic problem at every period to achieve the optimal cost.
    In other words, only use OOS data at a period not the entire path.
    """

    def __init__(self, data, args):
        self.data = data
        self.args = args

    def update_static_2ssp_init(self, init, eval_type):
        init.t = 0
        init.demand_tprime = [0.0] * self.data.I
        init.last_stage = False
        init.T_PRIME = list(range(self.data.T))
        if eval_type == 'oos':
            init.ts = self.data.ts_in_sample
            if self.args['demand_opt'] == 1:
                init.demand_insample = self.data.demand_in_sample
            else:
                init.demand_insample = helper.compute_total_demand(
                    self.data, self.data.demand_in_sample,
                    )
        # TODO : change mc_tree demand according to self.args['demand_opt']
        elif eval_type == 'mc_tree':
            init.ts = [len(self.data.in_sample_from_tree_2ssp[s]) - 1
                       for s in range(self.data.S)]
            demand_insample = {s: {} for s in range(self.data.S)}
            for s in range(self.data.S):
                samples = self.data.in_sample_from_tree_2ssp[s]
                for t, state in enumerate(samples):
                    state = tuple(state)
                    demand = self.data.demand_mssp[t][state]
                    demand_insample[s][t] = demand
            init.demand_insample = helper.compute_total_demand(
                self.data, demand_insample
            )
        init.T_2ND = {
            s: init.T_PRIME[1: init.ts[s] + 1]
            for s in range(self.data.S)
            }

    def solve_static_2ssp(self):
        """Solve static 2SSP at t=0 and get the cost on out-of-samples (OOS)
        generated from MC Tree or the true OOS generated from AR-1 model
        """
        init = InitSLAM(self.data)
        self.update_static_2ssp_init(init, eval_type=self.args['eval'])
        solve_slam = SolveSLAM(self.data, init, **self.args)
        solve_slam.solve()
        solve_slam.export_and_print(method=self.args['method'], write=True)

    def oos_test_anticipative(self, rnn_test=False):
        init = InitSLAM(self.data)
        self.update_static_2ssp_init(init, eval_type=self.args['eval'])
        solve_slam = SolveSLAM(self.data, init, **self.args)
        # out-sample testing: (solve sub-problems for 'n_oos' OOS scenarios)
        test_result = []
        if self.args['eval'] == 'oos':
            oos_demand_frac = {s: self.data.demand_oos[s]
                               for s in range(self.args['n_oos'])}
        else:
            oos_demand_frac = {
                s: {
                    t: self.data.demand_mssp[t][tuple(state)]
                    for t, state in enumerate(
                        self.data.test_samples_from_tree[s]
                        )
                    }
                    for s in range(self.args['n_oos'])
                    }
        if self.args['demand_opt'] == 1:
            oos_demand = oos_demand_frac
        else:
            oos_demand = helper.compute_total_demand(
                self.data, oos_demand_frac
                )
        start_time = time.time()
        # solution export begins ----------------------------------------
        sols_export = {}
        # solution export ends   ----------------------------------------
        print('-' * 40, '\n2SSP-anticipative OOS cost \n',)
        for so in range(self.args["n_oos"]):
            if time.time() - start_time > self.args["time_limit_test"]:
                break
            if self.args["eval"] == "oos":
                ts = self.data.ts_oos[so]
            elif self.args["eval"] == "mc_tree":
                ts = len(self.data.test_samples_from_tree[so]) - 1
            demand = oos_demand[so]
            demand = helper.compute_total_demand(self.data, {0: demand})
            # Add test sample cost
            solve_slam.slam.init.demand_insample[self.data.S] = demand
            solve_slam.slam.init.T_2ND[self.data.S] = init.T_PRIME[1: ts + 1]
            obj, sol_temp = solve_slam.slam.sub_prob_test(
                demand, rnn_test=rnn_test,
                )
            print('s={} \t cost={:.2f}'.format(so, obj['Total']))
            sols_export[so] = sol_temp
            if rnn_test is True:
                sol = solve_slam.slam.solution(m=solve_slam.slam.sub_prob[-1])
                test_result.append(sol)

            else:
                test_result.append(pd.DataFrame(obj, index=[so]))
                result = pd.concat(test_result).rename_axis("s").round(2)
                result.to_csv(
                    self.data.DIR[4] +
                    "2ssp_eval_{}_{}.csv".format(
                        self.args["eval"],
                        self.args['method'],
                        )
                    )
                summary = helper.summerize_result(result)
                summary.to_csv(
                    self.data.DIR[4] +
                    "summary_2ssp_eval_{}_{}.csv".format(
                        self.args["eval"],
                        self.args['method'],
                        )
                    )
                # new (July 24, 2024) begins -------------------------------
                with open(
                    self.data.DIR[4] + 'model_sol_anticipative.json', 'w',
                    ) as file:
                    export = transform_keys(sols_export)
                    json.dump(export, file, indent=4)
                # new (July 24, 2024) ends --------------------------------
        print('-' * 40)
        return test_result

    def deterministic_model_cost(self, var):
        cost = {
            "Fixed": 0.0,
            "Relief Inventory": 0.0,
            "Evacuee Inventory": sum(
                self.data.c_invE_J[j] * var['eJ'][j]
                for j in range(self.data.J)
                ),
            "Penalty": sum(
                self.data.c_PE[i] * var['u'][i] for i in range(self.data.I)
                ),
            "Emergency": sum(
                self.data.c_G_J[j] * var['g'][j] for j in range(self.data.J)
                ),
            "Relief Purchase": sum(sum(
                self.data.c_P_K[k] * var['xKJ'][k, j]
                for j in range(self.data.J)
                ) for k in range(self.data.K)),
            "Relief Transportation": sum(sum(
                self.data.c_R_JJ[j][j_] * var['xJJ'][j, j_]
                for j in range(self.data.J)
                ) for j_ in range(self.data.J)) + sum(sum(
                    self.data.c_R_KJ[k][j] * var['xKJ'][k, j]
                    for j in range(self.data.J)
                ) for k in range(self.data.K)),
            "Evacuee Transportation": sum(sum(
                self.data.c_E_IJ[i][j] * var['y'][i, j]
                for i in range(self.data.I)
                ) for j in range(self.data.J)),
            'Relief Dumping': sum(map(lambda x, y: x * y,
                                      self.data.c_H_J, list(var['h'].values())))
            }
        return cost

    def solve_deterministic_model(self, init):
        t = init['t']
        cost = {}
        m = gb.Model(f'm_detrministic_t{t}')
        m.setParam("OutputFlag", 0)
        var = dict(
            u=m.addVars(self.data.I, vtype=CT, lb=0, ub=INF, name='u'),
            y=m.addVars(self.data.I, self.data.J,
                        vtype=CT, lb=0, ub=INF, name='y'),
            g=m.addVars(self.data.J, vtype=CT, lb=0, ub=INF, name='g'),
            h=m.addVars(self.data.J, vtype=CT, lb=0, ub=INF, name='h'),
            eJ=m.addVars(self.data.J, vtype=CT, lb=0, ub=INF, name='eJ'),
            xJJ=m.addVars(self.data.J, self.data.J,
                          vtype=CT, lb=0, ub=INF, name='xJJ'),
            xKJ=m.addVars(self.data.K, self.data.J,
                          vtype=CT, lb=0, ub=INF, name='xKJ'),
            )
        m.update()
        # ----- Constraints -----
        m.addConstrs((
            gb.quicksum(var['y'][i, j] for j in range(self.data.J))
            + var['u'][i]
            ==
            init['oos_demand_at_t'][i] * (
                init['eI'][i] if self.args['demand_opt'] == 1 else 1
                )
            for i in range(self.data.I)
            ))
        if self.args['oos_heur'] == 2:
            m.addConstrs((
                var['y'][i, j] <= init['y_sol'][i, j, t]
                for i in range(self.data.I)
                for j in range(self.data.J)
            ))
        m.addConstrs((var['xJJ'][j, j] == 0 for j in range(self.data.J)))
        m.addConstrs((
            gb.quicksum(var['y'][i, j] for i in range(self.data.I))
            + init['eJ'][j]
            ==
            var['eJ'][j]
            for j in range(self.data.J)
            ))
        m.addConstrs((
            var['eJ'][j] <= self.data.q_J[j] * self.sol_t0['zJ'][j, t]
            for j in range(self.data.J))
            )
        m.addConstrs((
            self.sol_t0['lJ'][j, t-1]
            + gb.quicksum(var['xKJ'][k, j] for k in range(self.data.K))
            + gb.quicksum(var['xJJ'][j_, j] for j_ in range(self.data.J))
            - gb.quicksum(var['xJJ'][j, j_] for j_ in range(self.data.J))
            - self.data.phi * var['eJ'][j]
            + var['g'][j]
            - var['h'][j]
            ==
            self.sol_t0['lJ'][j, t]
            for j in range(self.data.J)
        ))
        m.addConstrs((
            gb.quicksum(var['xJJ'][j, j_]
                        for j_ in range(self.data.J) if j != j_)
            <=
            self.sol_t0['lJ'][j, t-1]
            + var['g'][j]
            - self.data.phi * var['eJ'][j]
            for j in range(self.data.J)
            ))
        m.addConstrs((
            var['g'][j]
            - self.data.phi * var['eJ'][j]
            <=
            max(self.data.phi * self.data.q_J[j] * self.sol_t0['zJ'][j, t] -
                self.sol_t0['lJ'][j, t-1], 0)
            for j in range(self.data.J)
            ))
        # lJ[j, t-1] sol is moved to rhs to avoid numerical issues.
        # Inventory should always be under max capacity so RHS is valid.
        m.update()
        m.setObjective(sum(self.deterministic_model_cost(var=var).values()),
                       gb.GRB.MINIMIZE)
        m.update()
        start_time = time.time()
        m.optimize()
        comp_time = time.time() - start_time
        if m.status != gb.GRB.OPTIMAL:
            print('error! infeasible deterministic stage t problem')
            m.write('temp.lp')
            print(init)
            exit(0)
        sol = {}
        for v_name, v_dict in var.items():
            sol[v_name] = {index: val.x for index, val in v_dict.items()}
        if self.args['demand_opt'] == 1:
            sol['eI'] = list(
                init['eI'][i] - sum(sol['y'][i, j] for j in range(self.data.J))
                for i in range(self.data.I)
            )
        cost = self.deterministic_model_cost(var=sol)
        return cost, sol, comp_time

    def get_insample_sol(self):
        init = InitSLAM(self.data)
        self.update_static_2ssp_init(init, eval_type=self.args['eval'])
        solve_slam = SolveSLAM(self.data, init, **self.args)
        # fix the first-stage solutions
        for j in range(self.data.J):
            for t in init.T_PRIME:
                solve_slam.slam.master.addConstr(
                    solve_slam.slam.master._vars['zJ'][j, t] ==
                    int(self.sol_t0['zJ'][j][t])
                    )
                solve_slam.slam.master.addConstr(
                    solve_slam.slam.master._vars['lJ'][j, t] ==
                    max(0, round(self.sol_t0['lJ'][j][t], 3))
                    )
                solve_slam.slam.master.update()
        solve_slam.solve()
        solve_slam.slam.master.write('temp.lp')
        sol = []
        for s in range(self.data.S):
            sol.append(
                solve_slam.slam.solution(
                    solve_slam.slam.sub_prob[s]
                    )
                )
        return sol

    def export_oos_result(self, oos_cost):
        # export: all oos result
        result_df = pd.concat(oos_cost).round(2).rename_axis('s')
        result_df.to_csv(
            self.data.DIR[4] +
            '2ssp_eval_{}_heur_{}.csv'.format(
                self.args['eval'],
                self.args['oos_heur'],
                )
            )
        # export: oos result summary
        summary = helper.summerize_result(result_df)
        summary.to_csv(
            self.data.DIR[4] +
            "summary_2ssp_eval_{}_{}_heur{}.csv".format(
                self.args['eval'],
                self.args['method'],
                self.args['oos_heur'],
                )
            )
        return summary

    def oos_test_myopic(self):
        # read first-stage solutions from model already solved
        result_1st_stg = read_first_stage_result(self.data, self.args)
        self.sol_t0, self.cost_t0, self.comp_time, _ = result_1st_stg
        # solve in_samples if heuristic = 2:
        if self.args['oos_heur'] == 2:
            in_sample_sol = self.get_insample_sol()
        oos_cost = []
        start_time = time.time()
        oos_demand_frac = {
            s: self.data.demand_oos[s]
            for s in range(self.args['n_oos'])
            }
        # export model solution begins ------------------------------------
        solution = {s: {} for s in range(self.args['n_oos'])}
        # export model solution ends --------------------------------------
        print('-' * 40, '\n2SSP-myopic OOS cost \n',)
        for s in range(self.args['n_oos']):
            if self.args['oos_heur'] == 2:
                closest_s_list = helper.get_closest_s(
                    self.data, s, self.args['eval']
                    )
            eval_time = time.time() - start_time
            if eval_time >= self.args['time_limit_test']:
                break
            cost_s = copy.deepcopy(self.cost_t0)
            sol = {
                'eI': list(self.sol_t0['eI'].values()),
                'eJ': list(self.sol_t0['eJ'].values()),
                }
            if self.args['eval'] == 'oos':
                ts = self.data.ts_oos[s]
                demand = self.data.demand_oos[s]
            else:
                # TODO do the same (demand recomputation) for mc_tree
                ts = len(self.data.test_samples_from_tree[s]) - 1
                demand = []
                for t, state in enumerate(self.data.test_samples_from_tree[s]):
                    state = tuple(state)
                    demand.append(self.data.demand_mssp[t][state])
            for t in range(1, ts + 1):
                y_sol = None
                if self.args['oos_heur'] == 2:
                    closest_in_sample = closest_s_list[t]
                    y_sol = in_sample_sol[closest_in_sample]['y']
                init = {
                    't': t,
                    'oos_demand_at_t': demand[t],
                    'y_sol': y_sol,
                    }
                init.update(sol)
                result_period_t = self.solve_deterministic_model(init)
                cost_t, sol_t, comp_time_t = result_period_t
                self.comp_time += comp_time_t
                sol = sol_t
                for name, val in cost_s.items():
                    cost_s[name] = val + cost_t[name]
                # export model solution begins ------------------------------
                solution[s][t] = {
                    'xKJ': sol['xKJ'],
                    'eJ': sol['eJ'],
                    'eI': sol['eI'],
                }
                # export model solution ends --------------------------------
            cost_s['Relief Inventory'] = sum(sum(
                self.data.c_invR_J[j] * self.sol_t0['lJ'][j, t]
                for j in range(self.data.J))
                for t in range(1, ts + 1))
            cost_s['Total'] = sum(cost_s.values())
            oos_cost.append(pd.DataFrame(cost_s, index=[s]))
            print('s = {} \t cost = {:.2f}'.format(s, cost_s['Total']))
            # export model solution begins ------------------------------
            with open(self.data.DIR[4] + 'model_sol_myopic.json', 'w') as file:
                json.dump(transform_keys(solution), file, indent=4)
            # export model solution ends --------------------------------
        print('-' * 40)
        summary = self.export_oos_result(oos_cost)
        print('\nSummary: \n', summary)

def transform_keys(d):
    if isinstance(d, dict):
        return {str(k): transform_keys(v) for k, v in d.items()}
    return d

class RollingHorizon:
    def __init__(self, data, args):
        self.data = data
        self.args = args
        result_1st_stg = read_first_stage_result(data, args)
        self.sol_t0, self.cost_t0, self.comp_time, _ = result_1st_stg
        print('\nStatic 2ssp cost at t=0 is {} \n'.format(
            sum(self.cost_t0.values())))

    def in_sample_scen_for_slam(self, s, t):
        # generate in-sample demand and t_s.
        if self.args['eval'] == 'oos':
            # generate forecast error (fe) samples first.
            fe_in_samples = {}
            for err in ['intensity', 'track']:
                fe_in_samples[err] = fe_scen.createErrorSamples(
                    data=self.data,
                    err=err,
                    S=self.data.S,
                    t=t,
                    oos_err_at_t=self.data.oos_err[err][s][t]
                    )
            demand, _, _, ts = data_demand_mapping.createDemand(
                data=self.data,
                err=fe_in_samples,
                S=self.data.S,
                tprime=t,
                )
        else:
            ts = [self.data.T - 1 for s in range(self.data.S)]
            demand = {scen: {} for scen in range(self.data.S)}
            sample_dict = {scen: {} for scen in range(self.data.S)}
            for scen in range(self.data.S):
                state = self.data.test_samples_from_tree[s][t]
                sample_dict[scen][t] = state
                for t2 in range(t + 1, self.data.T):
                    s_ = sample_dict[scen][t2 - 1]
                    indices = range(len(self.data.state_space[t2]))
                    i = np.random.choice(
                        indices,
                        p=list(self.data.pi_mssp[t2-1][s_].values())
                        )
                    state_sampled = self.data.state_space[t2][i]
                    sample_dict[scen][t2] = state_sampled
                    demand[scen][t2] = self.data.demand_mssp[t2][state_sampled]
        return demand, ts

    def update_init(self, init, sol, t, s):
        # update state variables and first-stage solution.
        for i in range(self.data.I):
            init.eI_hat[i, t] = sol['eI'][i, t - 1]
        for j in range(self.data.J):
            init.lJ_hat[j, t] = sol['lJ'][j, t - 1]
            init.eJ_hat[j, t] = sol['eJ'][j, t - 1]
            # 'z' variables are only optimized at t = 0
            if t - 1 == 0:
                for t_ in range(self.data.T):
                    init.zJ_hat[j, t_] = sol['zJ'][j, t_]
        init.t = t
        if self.args['eval'] == 'oos':
            init.demand_tprime = self.data.demand_oos[s][t]
        else:
            state = self.data.test_samples_from_tree[s][t]
            init.demand_tprime = self.data.demand_mssp[t][state]
        init.last_stage = True if t == self.data.ts_oos[s] else False
        init.T_PRIME = list(range(t, self.data.T))
        demand, ts = self.in_sample_scen_for_slam(s, t)
        init.demand_insample = demand
        init.ts = ts
        init.T_2ND = {s: init.T_PRIME[1:] for s in range(self.data.S)}

    def solve_rolling_horizon(self):
        self.result_alg = {}
        self.result_cost = []
        start_time = time.time()
        for s in range(self.args['n_oos']):
            test_time = time.time() - start_time
            if test_time > self.args['time_limit_test']:
                break
            cost = copy.deepcopy(self.cost_t0)
            sol = copy.deepcopy(self.sol_t0)
            init = InitSLAM(self.data)
            for t in range(1, self.data.ts_oos[s] + 1):
                self.update_init(init, sol, t, s)
                # solve SLAM with initial conditions.
                solve_slam = SolveSLAM(self.data, init, **self.args)
                print(f'Solving SLAM for OOS = {s} at t = {t} \n')
                solve_slam.solve()
                self.comp_time += solve_slam.comp_time
                master_prob_cost = solve_slam.slam.obj(
                    solve_slam.slam.master, option=2
                    )
                cost_tprime = {
                    name: cost[0] for name, cost in
                    master_prob_cost.items() if name != 'Fixed'
                    }
                for name, val in cost_tprime.items():
                    if name not in list(self.cost_t0.keys()):
                        print('error! bug found on cost dictionary')
                        exit(0)
                    if name != 'Fixed':
                        cost[name] += val
                sol = solve_slam.slam.solution(solve_slam.slam.master)
                print('\nSLAM obj = {:.2f}, Cost at t{} = {:.2f}, '.format(
                    solve_slam.slam.master.objVal,
                    t,
                    sum(cost_tprime.values()),
                    ) +
                    'proportion = {:.2f} % \n'.format(
                        100 * sum(cost_tprime.values()) /
                        solve_slam.slam.master.objVal,
                    ))
            total_rh_cost = sum(cost.values())
            print('\ns = {} \t rh cost = {:.2f} \n'.format(s, total_rh_cost))
            self.result_alg[s] = {
                'rh_cost': total_rh_cost,
                'comp_time': self.comp_time,
                'lj': vars(init)['lJ_hat'][:, 1:].tolist(),
                }
            df_comp = pd.DataFrame(cost, index=[s])
            df_comp['Total'] = df_comp.sum(axis=1)
            self.result_cost.append(df_comp)
            self.export()

    def export(self):
        with open(
                self.data.DIR[4] + 'rh_{}.json'.format(self.args['method']),
                'w'
                ) as file:
            json.dump(self.result_alg, file, indent=4)
        # Create DataFrame of OOS cost components.
        result_df = pd.concat(self.result_cost).rename_axis('s').round(2)
        result_df.to_csv(
            self.data.DIR[4] + "rh_eval_{}_{}.csv".format(
                self.args['eval'],
                self.args['method'],
                )
            )
        summary = helper.summerize_result(result_df)
        summary.to_csv(
            self.data.DIR[4] + "summary_rh_eval_{}_{}.csv".format(
                self.args['eval'],
                self.args['method'],
                )
            )
