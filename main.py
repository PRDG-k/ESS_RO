
from model import solve_det_model
from util import *
from instance import *
import argparse


def run_deterministic(instance: Instance):
    res, sol = solve_det_model(instance)
    return res, sol

def run_experiment(params: Params):
    instance = Instance(params)
    if params.MODEL =='det':
        solve_result, sol = run_deterministic(instance)
    else:
        print("invalid MODEL")
        return 0
    


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

    run_experiment(params)

if __name__ == "__main__":
    #sys.argv = ['test', '--model', 'box', '--obj_type', '1']
    main()