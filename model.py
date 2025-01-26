import pulp
import pandas as pd
import os
import orloge as ol

from util import *
from typing import Dict, List
from instance import *

def solve_selected_model(inst: Instance):
    match inst.MODEL:
        case "det":
            res, sol = solve_det_model(inst)
        case "ro":
            res, sol = solve_ro_model(inst)
        case "so":
            res, sol = solve_so_model(inst)    
    return res, sol

def solve_ro_model(inst: Instance):
    # 모델 생성
    PERIOD = inst.PERIOD
    efficiency = inst.eta
    LOAD = inst.load
    PV = inst.pv
    tou_price = inst.CHARGING_PRICE

    # ==============================================
    
    model = pulp.LpProblem('robustic', pulp.LpMinimize)
    model, u, d_load, d_bess, v_load, v_bess, c = common_vars_and_cons(model,inst)
    
    #define constraints
    if inst.CTR_STYLE==0:    # standard balance equation type

        # Load Constraint
        for t in PERIOD:
            model += d_load[t] + v_load[t] + efficiency * c[t] == LOAD[t]

        # PV Constraint
        for t in PERIOD:
            model += v_bess[t] + v_load[t] == PV[t]

        # Objective Function
        for t in PERIOD:
                model.objective += pulp.lpSum(tou_price[t] * (d_bess[t]+d_load[t]))

    else:   # dense type
        return None
    
                
    result_dict = solve_model(model, 60)    
    sol_list = save_sols(model, u, d_load, d_bess, v_load, v_bess, c)
    
    del model 
    
    return result_dict, sol_list



def solve_so_model(inst: Instance):
# 모델 생성
    PERIOD = inst.PERIOD
    efficiency = inst.eta
    LOAD = inst.load
    PV = inst.pv
    tou_price = inst.CHARGING_PRICE

    # ==============================================
    
    model = pulp.LpProblem('stochastic', pulp.LpMinimize)
    model, u, d_load, d_bess, v_load, v_bess, c = common_vars_and_cons(model,inst)
    
    #define constraints
    if inst.CTR_STYLE==0:    # standard balance equation type

        # Load Constraint
        for t in PERIOD:
            model += d_load[t] + v_load[t] + efficiency * c[t] == LOAD[t]

        # PV Constraint
        for t in PERIOD:
            model += v_bess[t] + v_load[t] == PV[t]

        # Objective Function
        for t in PERIOD:
                model.objective += pulp.lpSum(tou_price[t] * (d_bess[t]+d_load[t]))

    else:   # dense type
        return None
    
                
    result_dict = solve_model(model, 60)    
    sol_list = save_sols(model, u, d_load, d_bess, v_load, v_bess, c)
    
    del model 
    
    return result_dict, sol_list


    
def solve_det_model(inst: Instance):
    # 모델 생성
    PERIOD = inst.PERIOD
    BUILDINGS = inst.BUILDINGS
    efficiency = inst.eta
    LOAD = inst.load
    PV = inst.pv
    tou_price = inst.CHARGING_PRICE

    # ==============================================
    
    model = pulp.LpProblem('deterministic', pulp.LpMinimize)
    model, u, d_load, d_bess, v_load, v_bess, c = common_vars_and_cons(model,inst)
    
    #define constraints
    if inst.CTR_STYLE==0:    # standard balance equation type

        # Load Constraint
        for t in PERIOD:
            model += d_load[t] + pulp.lpSum(v_load[t,b] for b in BUILDINGS) + efficiency * c[t] == LOAD[t]

        # PV Constraint
        for t in PERIOD:
            for b in BUILDINGS:
                model += v_bess[t,b] + v_load[t,b] == PV[b][t]

        # Objective Function
        for t in PERIOD:
            model.objective += pulp.lpSum(tou_price[t] * (d_bess[t]+d_load[t]))

    else:   # dense type
        return None
    
                
    result_dict = solve_model(model, 60)    
    sol_list = save_sols(model, u, d_load, d_bess, v_load, v_bess, c)
    
    del model 
    
    return result_dict, sol_list
      
            
def common_vars_and_cons(model:pulp.LpProblem , inst: Instance):
    PB = {(p,b) for p in inst.PERIOD for b in inst.BUILDINGS}
    
    # Variables
    u = pulp.LpVariable.dicts('soc', indices=inst.PERIOD, lowBound=0.2, upBound=0.8, cat='Continuous')
    d_bess = pulp.LpVariable.dicts('db', indices=inst.PERIOD, lowBound=0, cat='Continuous')
    d_load = pulp.LpVariable.dicts('dl', indices=inst.PERIOD, lowBound=0, cat='Continuous')
    v_load = pulp.LpVariable.dicts('vl', indices=PB, lowBound=0, cat='Continuous')
    v_bess = pulp.LpVariable.dicts('vb', indices=PB, lowBound=0, cat='Continuous')
    c = pulp.LpVariable.dicts('dc', indices=inst.PERIOD, lowBound=0, cat='Continuous')
    

    # Initial and Final SoC
    model += u[0] == 0.5
    model += u[inst.T-1] == 0.5

    # SoC Constraint
    for t in range(0,inst.T-2):
        model += u[t] + (inst.eta * (d_bess[t] + pulp.lpSum(v_bess[t, b] for b in inst.BUILDINGS)) - c[t]) / inst.CAPA == u[t+1]


    # BESS power Constraint
    for t in inst.PERIOD:
        model += c[t] + inst.eta * (d_bess[t] + pulp.lpSum(v_bess[t, b] for b in inst.BUILDINGS)) <= inst.POWER
    
    return model, u, d_load, d_bess, v_load, v_bess, c


def solve_model(model: pulp.LpProblem, MAX_TIME: int, DEBUG=True):
    result={}
    
    if DEBUG:
        model.writeLP('./model_{}.lp'.format(model.name))


    path = "log/log_file.txt"
    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=MAX_TIME, logPath=path) # We set msg=False since logPath disables CBC's logs in the Python console
    status = model.solve(solver)

    logs_dict = ol.get_info_solver(path, 'CBC') # Orloge returns a dict with all logs info

    if status == 1:
        best_solution = pulp.value(model.objective)
        best_bound = 0
        # best_bound = logs_dict["best_bound"]
        # gap = abs(best_solution - best_bound) / (eps + abs(best_bound)) * 100
        gap = 0
        print(f"Gap (relative to the lower bound) is {gap:.2f}%.")
    else :
        print("Unable to retrieve gap.")
        print(logs_dict)


    result['sol_time'] = np.round(model.solutionTime,5)
    try:
        result['sol_gap'] = np.round(gap,5)
        result['sol_obj'] = np.round(best_solution,3)
        result['sol_UB'] = np.round(best_bound,3)
    except UnboundLocalError:
        result['sol_gap'] = -1
        result['sol_obj'] =  -1
        result['sol_UB'] = -1
    
    return result 



def save_sols(model: pulp.LpProblem, u, d_load, d_bess, v_load, v_bess, c ):
    
    dl_values = {key: np.round(var.varValue,4) for key, var in d_load.items()}
    db_values = {key: np.round(var.varValue,4) for key, var in d_bess.items()}
    vl_values = {key: np.round(var.varValue,4) for key, var in v_load.items()}
    vb_values = {key: np.round(var.varValue,4) for key, var in v_bess.items()}
    u_values = {key: np.round(var.varValue,4) for key, var in u.items()}
    c_values = {key: np.round(var.varValue,4) for key, var in c.items()}
    
    sol_list = {"dl": dl_values,
                "db": db_values,
                "vl": vl_values,
                "vb": vb_values,
                "soc": u_values,
                "dc": c_values}

    pd.DataFrame(sol_list).to_csv("log/vars_result.csv")
    return sol_list