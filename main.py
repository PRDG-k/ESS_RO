
from model import solve_selected_model
from util import *
from instance import *
import argparse

def evaluate_sol(params: Params, model_res, model_sol):
    i_ = Instance(params)
    i_.MODEL="real"
    i_.read_predict_data("pv_real.csv")
    opt_res, opt_sol = solve_selected_model(i_)

    sum_v_i = i_.sum_v_i

    penalty = 0
    for idx, (vb, vl) in enumerate(zip(model_sol['vb'].values(), model_sol['vl'].values())):
        penalty += i_.CHARGING_PRICE[idx] * ( vb + vl - i_.sum_v_i[idx]) if ( vb + vl - i_.sum_v_i[idx]) > 0 else 0

    score = opt_res['sol_obj'] - model_res['sol_obj'] - penalty
    
    print(f"Gap between models: {-score}")
    print(f"Penalty: {penalty}")
    
    return score

def run_experiment(params: Params):
    instance = Instance(params)
    res, sol = solve_selected_model(instance)

    score = evaluate_sol(params, res, sol)

    return res, sol

def main():
    parser = setparser(argparse.ArgumentParser(description="Experiments."))
    
    args = parser.parse_args()
    
    params = Params(
        
        SEASON= args.season,
        DATANUM=args.dnum,
        MODEL = args.model,
        INSAMPLE = args.insample, 
        NUM_SCENARIO= args.num_scenario,
        CV_uncertainty_set= args.cv,
        CV_scenario= args.cv_scenario,
        CTR_STYLE= args.ctr_style, 
        GAMMA= args.gamma
    )

    res, sol = run_experiment(params)

if __name__ == "__main__":
    #sys.argv = ['test', '--model', 'box', '--obj_type', '1']
    main()