import os 
import pulp
import numpy as np
from dataclasses import dataclass
import pandas as pd

eps = 10e-6

@dataclass 
class Params:
    
    #====================================================
    DATANUM: int
    SEASON: str
    #====================================================
    MODEL: str                  # 3  |  Deterministic, Robustic, Stochastic
    
    CTR_STYLE: bool     # 0: sparse, 1: dense
    CV_uncertainty_set: float   # 3  | 0.1, 0.3, 0.5, (0.7, 0.9) 
    
    INSAMPLE: bool              # 2  | 0, 1(test)
    
    CV_scenario : float         # 5  | 0.1, 0.3, 0.5 0.7, 0.9 
    NUM_SCENARIO: int           # 1  | 1000 scenarios

    GAMMA: int          # 1, 2, 3



class Instance:
    def __init__(self, param: Params):
        
        self.SEASON = param.SEASON
        self.DATANUM = param.DATANUM

        self.INSAMPLE= param.INSAMPLE
        self.MODEL = param.MODEL
        self.NUM_SCENARIO = param.NUM_SCENARIO
        
        
        self.CV_consumption = param.CV_uncertainty_set   # To generate uncertainty set range
        self.CV_reduction = param.CV_uncertainty_set     # To generate uncertainty set range
        
        self.real_CV_consumption = param.CV_scenario   #To generate scenario
        self.real_CV_reduction = param.CV_scenario     #To generate scenario
        
        self.CTR_STYLE = param.CTR_STYLE
        self.GAMMA = param.GAMMA
        
        self.T = 24
        self.HOUR = 60
        self.PERIOD = list(range(0,self.T))
        
        self.CAPA = 750     # BESS Capacity
        self.eta = 0.93     # Efficiency of BESS
        self.POWER = 250    # Max power of BESS(0 -> 1 soc in 3 hours)


        self.read_price_data()
        self.read_predict_data("pv_pred.csv")

        # self.processing_data()

    
    def read_price_data(self):
        current_dir = os.getcwd()
        data_dir = os.path.join(current_dir, "Data")
        
        #Master
        temp_df = pd.read_csv(os.path.join(data_dir, "electricity_price.csv"))
        self.CHARGING_PRICE = dict(zip(temp_df['tw_s'],temp_df[self.SEASON]))

        # 계절 별 peak에 해당하는 시간 저장
        if self.SEASON == "winter":
            self.peak_time = dict(zip(temp_df['tw_s'],temp_df['peak_w']))  
        else:
            self.peak_time = dict(zip(temp_df['tw_s'],temp_df['peak_s']))
        
        
        self.DISCHARGING_PRICE = {key: value * 1 for key, value in self.CHARGING_PRICE.items()}
        self.PENALTY_PRICE = {key: value *10 for key, value in self.CHARGING_PRICE.items()}
    
    def read_predict_data(self, filename):
        current_dir = os.getcwd()
        data_dir = os.path.join(current_dir, "Data")

        pv = pd.read_csv(os.path.join("ml", filename), index_col=0)

        assert (np.array(pv) >= 0).all() == True
        assert len(pv) == self.T

        self.load = pd.read_csv(os.path.join(data_dir, 'LOAD_RESULT.csv'), usecols=['Total Load'])['Total Load']
        self.pv = {col.split("_")[0]: np.array(pv[col]) for col in pv.columns}
        self.sum_v_i = pv.sum(axis=1)
        self.sum_v_i.index = self.PERIOD

        self.BUILDINGS = list(self.pv.keys())

        if self.MODEL == 'ro':
            # ML의 성능을 불롹실성으로 모델링
            res = pd.read_csv(os.path.join("ml", "pv_pred_error.csv"), index_col=0)
            res.columns = self.BUILDINGS
            self.res_mean = res.groupby(level=0).mean()
            self.res_sd = res.groupby(level=0).std()
            

            res = pd.read_csv(os.path.join("ml", "pv_pred_error_by_time.csv"), index_col=0)
            # res = res.astype(float)
            self.tres_mean = list(res.mean())
            self.tres_sd = list(res.std())
            self.skewness = list(sum(res.values < 0) / len(res))
            # self.volta = 

            
    def processing_data(self):

        self.charging_idx_list = []
        self.discharging_idx_list = []

    
    def export_instance(self, model:pulp.LpProblem):
        charge = []
        discharge = []