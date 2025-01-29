import pandas as pd
import numpy as np
import os
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler


cur_dir = os.getcwd()
data_dir = os.path.join(cur_dir, "Data")

data = pd.read_csv(os.path.join(data_dir, "pv_data.csv"))
excol = [col for col in data.columns if "시간당발전량" not in col]
pv = data.drop(columns=excol)


# 평균, 표준편차 반환
# pv['시간'] = [i//24 for i in range(len(data['시간']))]
# result = pv.groupby('시간')[pv.columns].agg(['mean', 'std'])

# result.to_excel("result.xlsx")

# 테스트
count = int(len(pv) // 24 * 0.3)
res_df = pd.DataFrame()
for i in range(1,count):
    pred_value = abs(pv[i * 24: (i+1) * 24])    # 음수가 간혹 포함되서 그냥 절댓값 씌움
    true_value = abs(pv[(i+1) * 24: (i+2) * 24])
    err = np.array(true_value) - np.array(pred_value)
    temp = pd.DataFrame(err, columns=pred_value.columns)

    # scaler = MinMaxScaler()
    # scaled_data = scaler.fit_transform(temp)
    # scaled_df = pd.DataFrame(scaled_data, columns=temp.columns)

    res_df = pd.concat([res_df, temp])

index = res_df.index

res_by_time = {}
for idx in index.unique():
    l = res_df.index == idx
    res_by_time[idx] = np.array(res_df.iloc[l]).reshape(1, -1).tolist()[0]

pd.DataFrame(res_by_time).to_csv(os.path.join("ml", "pv_pred_error_by_time.csv"))

pred_value.to_csv(os.path.join("ml","pv_pred.csv"))
res_df.to_csv(os.path.join("ml", "pv_pred_error.csv"))