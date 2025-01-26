import pandas as pd
import numpy as np
import os
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


cur_dir = os.getcwd()
data_dir = os.path.join(cur_dir, "Data")

data = pd.read_csv(os.path.join(data_dir, "pv_data.csv"))
excol = [col for col in data.columns if "시간당발전량" not in col]
pv = data.drop(columns=excol)

# 테스트
pred_value = abs(pv[-48:-24])
true_value = pv[-24:]

err = abs(np.array(true_value) - np.array(pred_value))

pred_value.to_csv(os.path.join("ml","pv_pred.csv"))
pd.DataFrame(err, columns=pred_value.columns).to_csv(os.path.join("ml", "pv_pred_error.csv"))