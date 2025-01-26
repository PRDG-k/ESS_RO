
from model import solve_selected_model
from util import *
from instance import *
import argparse

def run_experiment(params: Params):
    instance = Instance(params)
    res, sol = solve_selected_model(instance)
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