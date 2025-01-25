import pulp
import pandas as pd
import os
import orloge as ol

from util import *
from typing import Dict, List
from instance import *

    
def solve_det_model(inst: Instance):
    # 모델 생성
    PERIOD = inst.PERIOD
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
      
            
def common_vars_and_cons(model:pulp.LpProblem , inst: Instance):
    
    # Variables
    u = pulp.LpVariable.dicts('BESS State of Charge', [t for t in inst.PERIOD], lowBound=0.2, upBound=0.8, cat='Continuous')
    d_load = pulp.LpVariable.dicts('Diesel to Load at t', [t for t in inst.PERIOD], lowBound=0, cat='Continuous')
    d_bess = pulp.LpVariable.dicts('Diesel to BESS at t', [t for t in inst.PERIOD], lowBound=0, cat='Continuous')
    v_load = pulp.LpVariable.dicts('PV to Load at t', [t for t in inst.PERIOD], lowBound=0, cat='Continuous')
    v_bess = pulp.LpVariable.dicts('PV to BESS at t', [t for t in inst.PERIOD], lowBound=0, cat='Continuous')
    c = pulp.LpVariable.dicts('BESS to Load at t', [t for t in inst.PERIOD], lowBound=0, cat='Continuous')
    

    # Initial and Final SoC
    model += u[0] == 0.5
    model += u[inst.T-1] == 0.5

    # SoC Constraint
    for t in range(0,inst.T-2):
        model += u[t] + (inst.eta * (d_bess[t] + v_bess[t]) - c[t]) / inst.CAPA == u[t+1]


    # BESS power Constraint
    for t in inst.PERIOD:
        model += c[t] + inst.eta * (d_bess[t] + v_bess[t]) <= inst.POWER
    
    return model, u, d_load, d_bess, v_load, v_bess, c


def solve_model(model: pulp.LpProblem, MAX_TIME: int, DEBUG=True):
    result={}
    
    #model.setParam("TimeLimit", MAX_TIME)
    # model.Params.OutputFlag=1
    if DEBUG:
        model.writeLP('./model_{}.lp'.format(model.name))


    path = "log/log_file.txt"
    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=99, logPath=path) # We set msg=False since logPath disables CBC's logs in the Python console
    status = model.solve(solver)

    logs_dict = ol.get_info_solver(path, 'CBC') # Orloge returns a dict with all logs info
    best_bound, best_solution = logs_dict["best_bound"], logs_dict["best_solution"]
        

    if status == 1:
        gap = abs(best_solution - best_bound) / (eps + abs(best_bound)) * 100
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
    return None
    dl_values = {key: var.X for key, var in x.items()}
    db_values = {key: var.X for key, var in y.items()}
    vl_values = {key: var.X for key, var in xr.items()}
    vb_values = {key: var.X for key, var in yr.items()}

    u_values = {key: var.X for key, var in u.items()}
    c_values = {key: var.X for key, var in e.items()}
    
    
    sol_list = [x_values, y_values]
    return sol_list