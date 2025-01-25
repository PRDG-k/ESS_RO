\* deterministic *\
Minimize
OBJ: 75 Diesel_to_BESS_at_t_0 + 75 Diesel_to_BESS_at_t_1
 + 89.2 Diesel_to_BESS_at_t_10 + 109 Diesel_to_BESS_at_t_11
 + 89.2 Diesel_to_BESS_at_t_12 + 109 Diesel_to_BESS_at_t_13
 + 109 Diesel_to_BESS_at_t_14 + 109 Diesel_to_BESS_at_t_15
 + 109 Diesel_to_BESS_at_t_16 + 109 Diesel_to_BESS_at_t_17
 + 89.2 Diesel_to_BESS_at_t_18 + 89.2 Diesel_to_BESS_at_t_19
 + 75 Diesel_to_BESS_at_t_2 + 89.2 Diesel_to_BESS_at_t_20
 + 89.2 Diesel_to_BESS_at_t_21 + 75 Diesel_to_BESS_at_t_22
 + 75 Diesel_to_BESS_at_t_23 + 75 Diesel_to_BESS_at_t_3
 + 75 Diesel_to_BESS_at_t_4 + 75 Diesel_to_BESS_at_t_5
 + 75 Diesel_to_BESS_at_t_6 + 75 Diesel_to_BESS_at_t_7
 + 89.2 Diesel_to_BESS_at_t_8 + 89.2 Diesel_to_BESS_at_t_9
 + 75 Diesel_to_Load_at_t_0 + 75 Diesel_to_Load_at_t_1
 + 89.2 Diesel_to_Load_at_t_10 + 109 Diesel_to_Load_at_t_11
 + 89.2 Diesel_to_Load_at_t_12 + 109 Diesel_to_Load_at_t_13
 + 109 Diesel_to_Load_at_t_14 + 109 Diesel_to_Load_at_t_15
 + 109 Diesel_to_Load_at_t_16 + 109 Diesel_to_Load_at_t_17
 + 89.2 Diesel_to_Load_at_t_18 + 89.2 Diesel_to_Load_at_t_19
 + 75 Diesel_to_Load_at_t_2 + 89.2 Diesel_to_Load_at_t_20
 + 89.2 Diesel_to_Load_at_t_21 + 75 Diesel_to_Load_at_t_22
 + 75 Diesel_to_Load_at_t_23 + 75 Diesel_to_Load_at_t_3
 + 75 Diesel_to_Load_at_t_4 + 75 Diesel_to_Load_at_t_5
 + 75 Diesel_to_Load_at_t_6 + 75 Diesel_to_Load_at_t_7
 + 89.2 Diesel_to_Load_at_t_8 + 89.2 Diesel_to_Load_at_t_9
Subject To
_C1: BESS_State_of_Charge_0 = 0.5
_C10: BESS_State_of_Charge_7 - BESS_State_of_Charge_8
 - 0.00133333333333 BESS_to_Load_at_t_7 + 0.00124 Diesel_to_BESS_at_t_7
 + 0.00124 PV_to_BESS_at_t_7 = 0
_C11: BESS_State_of_Charge_8 - BESS_State_of_Charge_9
 - 0.00133333333333 BESS_to_Load_at_t_8 + 0.00124 Diesel_to_BESS_at_t_8
 + 0.00124 PV_to_BESS_at_t_8 = 0
_C12: - BESS_State_of_Charge_10 + BESS_State_of_Charge_9
 - 0.00133333333333 BESS_to_Load_at_t_9 + 0.00124 Diesel_to_BESS_at_t_9
 + 0.00124 PV_to_BESS_at_t_9 = 0
_C13: BESS_State_of_Charge_10 - BESS_State_of_Charge_11
 - 0.00133333333333 BESS_to_Load_at_t_10 + 0.00124 Diesel_to_BESS_at_t_10
 + 0.00124 PV_to_BESS_at_t_10 = 0
_C14: BESS_State_of_Charge_11 - BESS_State_of_Charge_12
 - 0.00133333333333 BESS_to_Load_at_t_11 + 0.00124 Diesel_to_BESS_at_t_11
 + 0.00124 PV_to_BESS_at_t_11 = 0
_C15: BESS_State_of_Charge_12 - BESS_State_of_Charge_13
 - 0.00133333333333 BESS_to_Load_at_t_12 + 0.00124 Diesel_to_BESS_at_t_12
 + 0.00124 PV_to_BESS_at_t_12 = 0
_C16: BESS_State_of_Charge_13 - BESS_State_of_Charge_14
 - 0.00133333333333 BESS_to_Load_at_t_13 + 0.00124 Diesel_to_BESS_at_t_13
 + 0.00124 PV_to_BESS_at_t_13 = 0
_C17: BESS_State_of_Charge_14 - BESS_State_of_Charge_15
 - 0.00133333333333 BESS_to_Load_at_t_14 + 0.00124 Diesel_to_BESS_at_t_14
 + 0.00124 PV_to_BESS_at_t_14 = 0
_C18: BESS_State_of_Charge_15 - BESS_State_of_Charge_16
 - 0.00133333333333 BESS_to_Load_at_t_15 + 0.00124 Diesel_to_BESS_at_t_15
 + 0.00124 PV_to_BESS_at_t_15 = 0
_C19: BESS_State_of_Charge_16 - BESS_State_of_Charge_17
 - 0.00133333333333 BESS_to_Load_at_t_16 + 0.00124 Diesel_to_BESS_at_t_16
 + 0.00124 PV_to_BESS_at_t_16 = 0
_C2: BESS_State_of_Charge_23 = 0.5
_C20: BESS_State_of_Charge_17 - BESS_State_of_Charge_18
 - 0.00133333333333 BESS_to_Load_at_t_17 + 0.00124 Diesel_to_BESS_at_t_17
 + 0.00124 PV_to_BESS_at_t_17 = 0
_C21: BESS_State_of_Charge_18 - BESS_State_of_Charge_19
 - 0.00133333333333 BESS_to_Load_at_t_18 + 0.00124 Diesel_to_BESS_at_t_18
 + 0.00124 PV_to_BESS_at_t_18 = 0
_C22: BESS_State_of_Charge_19 - BESS_State_of_Charge_20
 - 0.00133333333333 BESS_to_Load_at_t_19 + 0.00124 Diesel_to_BESS_at_t_19
 + 0.00124 PV_to_BESS_at_t_19 = 0
_C23: BESS_State_of_Charge_20 - BESS_State_of_Charge_21
 - 0.00133333333333 BESS_to_Load_at_t_20 + 0.00124 Diesel_to_BESS_at_t_20
 + 0.00124 PV_to_BESS_at_t_20 = 0
_C24: BESS_State_of_Charge_21 - BESS_State_of_Charge_22
 - 0.00133333333333 BESS_to_Load_at_t_21 + 0.00124 Diesel_to_BESS_at_t_21
 + 0.00124 PV_to_BESS_at_t_21 = 0
_C25: BESS_to_Load_at_t_0 + 0.93 Diesel_to_BESS_at_t_0
 + 0.93 PV_to_BESS_at_t_0 <= 250
_C26: BESS_to_Load_at_t_1 + 0.93 Diesel_to_BESS_at_t_1
 + 0.93 PV_to_BESS_at_t_1 <= 250
_C27: BESS_to_Load_at_t_2 + 0.93 Diesel_to_BESS_at_t_2
 + 0.93 PV_to_BESS_at_t_2 <= 250
_C28: BESS_to_Load_at_t_3 + 0.93 Diesel_to_BESS_at_t_3
 + 0.93 PV_to_BESS_at_t_3 <= 250
_C29: BESS_to_Load_at_t_4 + 0.93 Diesel_to_BESS_at_t_4
 + 0.93 PV_to_BESS_at_t_4 <= 250
_C3: BESS_State_of_Charge_0 - BESS_State_of_Charge_1
 - 0.00133333333333 BESS_to_Load_at_t_0 + 0.00124 Diesel_to_BESS_at_t_0
 + 0.00124 PV_to_BESS_at_t_0 = 0
_C30: BESS_to_Load_at_t_5 + 0.93 Diesel_to_BESS_at_t_5
 + 0.93 PV_to_BESS_at_t_5 <= 250
_C31: BESS_to_Load_at_t_6 + 0.93 Diesel_to_BESS_at_t_6
 + 0.93 PV_to_BESS_at_t_6 <= 250
_C32: BESS_to_Load_at_t_7 + 0.93 Diesel_to_BESS_at_t_7
 + 0.93 PV_to_BESS_at_t_7 <= 250
_C33: BESS_to_Load_at_t_8 + 0.93 Diesel_to_BESS_at_t_8
 + 0.93 PV_to_BESS_at_t_8 <= 250
_C34: BESS_to_Load_at_t_9 + 0.93 Diesel_to_BESS_at_t_9
 + 0.93 PV_to_BESS_at_t_9 <= 250
_C35: BESS_to_Load_at_t_10 + 0.93 Diesel_to_BESS_at_t_10
 + 0.93 PV_to_BESS_at_t_10 <= 250
_C36: BESS_to_Load_at_t_11 + 0.93 Diesel_to_BESS_at_t_11
 + 0.93 PV_to_BESS_at_t_11 <= 250
_C37: BESS_to_Load_at_t_12 + 0.93 Diesel_to_BESS_at_t_12
 + 0.93 PV_to_BESS_at_t_12 <= 250
_C38: BESS_to_Load_at_t_13 + 0.93 Diesel_to_BESS_at_t_13
 + 0.93 PV_to_BESS_at_t_13 <= 250
_C39: BESS_to_Load_at_t_14 + 0.93 Diesel_to_BESS_at_t_14
 + 0.93 PV_to_BESS_at_t_14 <= 250
_C4: BESS_State_of_Charge_1 - BESS_State_of_Charge_2
 - 0.00133333333333 BESS_to_Load_at_t_1 + 0.00124 Diesel_to_BESS_at_t_1
 + 0.00124 PV_to_BESS_at_t_1 = 0
_C40: BESS_to_Load_at_t_15 + 0.93 Diesel_to_BESS_at_t_15
 + 0.93 PV_to_BESS_at_t_15 <= 250
_C41: BESS_to_Load_at_t_16 + 0.93 Diesel_to_BESS_at_t_16
 + 0.93 PV_to_BESS_at_t_16 <= 250
_C42: BESS_to_Load_at_t_17 + 0.93 Diesel_to_BESS_at_t_17
 + 0.93 PV_to_BESS_at_t_17 <= 250
_C43: BESS_to_Load_at_t_18 + 0.93 Diesel_to_BESS_at_t_18
 + 0.93 PV_to_BESS_at_t_18 <= 250
_C44: BESS_to_Load_at_t_19 + 0.93 Diesel_to_BESS_at_t_19
 + 0.93 PV_to_BESS_at_t_19 <= 250
_C45: BESS_to_Load_at_t_20 + 0.93 Diesel_to_BESS_at_t_20
 + 0.93 PV_to_BESS_at_t_20 <= 250
_C46: BESS_to_Load_at_t_21 + 0.93 Diesel_to_BESS_at_t_21
 + 0.93 PV_to_BESS_at_t_21 <= 250
_C47: BESS_to_Load_at_t_22 + 0.93 Diesel_to_BESS_at_t_22
 + 0.93 PV_to_BESS_at_t_22 <= 250
_C48: BESS_to_Load_at_t_23 + 0.93 Diesel_to_BESS_at_t_23
 + 0.93 PV_to_BESS_at_t_23 <= 250
_C49: 0.93 BESS_to_Load_at_t_0 + Diesel_to_Load_at_t_0 + PV_to_Load_at_t_0
 = 5426.75
_C5: BESS_State_of_Charge_2 - BESS_State_of_Charge_3
 - 0.00133333333333 BESS_to_Load_at_t_2 + 0.00124 Diesel_to_BESS_at_t_2
 + 0.00124 PV_to_BESS_at_t_2 = 0
_C50: 0.93 BESS_to_Load_at_t_1 + Diesel_to_Load_at_t_1 + PV_to_Load_at_t_1
 = 5387.81
_C51: 0.93 BESS_to_Load_at_t_2 + Diesel_to_Load_at_t_2 + PV_to_Load_at_t_2
 = 5376.11
_C52: 0.93 BESS_to_Load_at_t_3 + Diesel_to_Load_at_t_3 + PV_to_Load_at_t_3
 = 5331.97
_C53: 0.93 BESS_to_Load_at_t_4 + Diesel_to_Load_at_t_4 + PV_to_Load_at_t_4
 = 5405.54
_C54: 0.93 BESS_to_Load_at_t_5 + Diesel_to_Load_at_t_5 + PV_to_Load_at_t_5
 = 5528.28
_C55: 0.93 BESS_to_Load_at_t_6 + Diesel_to_Load_at_t_6 + PV_to_Load_at_t_6
 = 5659.07
_C56: 0.93 BESS_to_Load_at_t_7 + Diesel_to_Load_at_t_7 + PV_to_Load_at_t_7
 = 5859.35
_C57: 0.93 BESS_to_Load_at_t_8 + Diesel_to_Load_at_t_8 + PV_to_Load_at_t_8
 = 6055.78
_C58: 0.93 BESS_to_Load_at_t_9 + Diesel_to_Load_at_t_9 + PV_to_Load_at_t_9
 = 6213.29
_C59: 0.93 BESS_to_Load_at_t_10 + Diesel_to_Load_at_t_10 + PV_to_Load_at_t_10
 = 6370.88
_C6: BESS_State_of_Charge_3 - BESS_State_of_Charge_4
 - 0.00133333333333 BESS_to_Load_at_t_3 + 0.00124 Diesel_to_BESS_at_t_3
 + 0.00124 PV_to_BESS_at_t_3 = 0
_C60: 0.93 BESS_to_Load_at_t_11 + Diesel_to_Load_at_t_11 + PV_to_Load_at_t_11
 = 6499.27
_C61: 0.93 BESS_to_Load_at_t_12 + Diesel_to_Load_at_t_12 + PV_to_Load_at_t_12
 = 6531.75
_C62: 0.93 BESS_to_Load_at_t_13 + Diesel_to_Load_at_t_13 + PV_to_Load_at_t_13
 = 6552.75
_C63: 0.93 BESS_to_Load_at_t_14 + Diesel_to_Load_at_t_14 + PV_to_Load_at_t_14
 = 6564.77
_C64: 0.93 BESS_to_Load_at_t_15 + Diesel_to_Load_at_t_15 + PV_to_Load_at_t_15
 = 6546.74
_C65: 0.93 BESS_to_Load_at_t_16 + Diesel_to_Load_at_t_16 + PV_to_Load_at_t_16
 = 6440.41
_C66: 0.93 BESS_to_Load_at_t_17 + Diesel_to_Load_at_t_17 + PV_to_Load_at_t_17
 = 6343.07
_C67: 0.93 BESS_to_Load_at_t_18 + Diesel_to_Load_at_t_18 + PV_to_Load_at_t_18
 = 6245.44
_C68: 0.93 BESS_to_Load_at_t_19 + Diesel_to_Load_at_t_19 + PV_to_Load_at_t_19
 = 6042.35
_C69: 0.93 BESS_to_Load_at_t_20 + Diesel_to_Load_at_t_20 + PV_to_Load_at_t_20
 = 5944.55
_C7: BESS_State_of_Charge_4 - BESS_State_of_Charge_5
 - 0.00133333333333 BESS_to_Load_at_t_4 + 0.00124 Diesel_to_BESS_at_t_4
 + 0.00124 PV_to_BESS_at_t_4 = 0
_C70: 0.93 BESS_to_Load_at_t_21 + Diesel_to_Load_at_t_21 + PV_to_Load_at_t_21
 = 5837.97
_C71: 0.93 BESS_to_Load_at_t_22 + Diesel_to_Load_at_t_22 + PV_to_Load_at_t_22
 = 5699.86
_C72: 0.93 BESS_to_Load_at_t_23 + Diesel_to_Load_at_t_23 + PV_to_Load_at_t_23
 = 5538.59
_C73: PV_to_BESS_at_t_0 + PV_to_Load_at_t_0 = 0
_C74: PV_to_BESS_at_t_1 + PV_to_Load_at_t_1 = 0
_C75: PV_to_BESS_at_t_2 + PV_to_Load_at_t_2 = 0
_C76: PV_to_BESS_at_t_3 + PV_to_Load_at_t_3 = 0
_C77: PV_to_BESS_at_t_4 + PV_to_Load_at_t_4 = 0
_C78: PV_to_BESS_at_t_5 + PV_to_Load_at_t_5 = 0
_C79: PV_to_BESS_at_t_6 + PV_to_Load_at_t_6 = 161.211808821
_C8: BESS_State_of_Charge_5 - BESS_State_of_Charge_6
 - 0.00133333333333 BESS_to_Load_at_t_5 + 0.00124 Diesel_to_BESS_at_t_5
 + 0.00124 PV_to_BESS_at_t_5 = 0
_C80: PV_to_BESS_at_t_7 + PV_to_Load_at_t_7 = 234.547297617
_C81: PV_to_BESS_at_t_8 + PV_to_Load_at_t_8 = 331.591383119
_C82: PV_to_BESS_at_t_9 + PV_to_Load_at_t_9 = 364.524825712
_C83: PV_to_BESS_at_t_10 + PV_to_Load_at_t_10 = 267.708380327
_C84: PV_to_BESS_at_t_11 + PV_to_Load_at_t_11 = 274.49129425
_C85: PV_to_BESS_at_t_12 + PV_to_Load_at_t_12 = 456.608192091
_C86: PV_to_BESS_at_t_13 + PV_to_Load_at_t_13 = 420.459313993
_C87: PV_to_BESS_at_t_14 + PV_to_Load_at_t_14 = 452.64957488
_C88: PV_to_BESS_at_t_15 + PV_to_Load_at_t_15 = 381.682222967
_C89: PV_to_BESS_at_t_16 + PV_to_Load_at_t_16 = 180.205029885
_C9: BESS_State_of_Charge_6 - BESS_State_of_Charge_7
 - 0.00133333333333 BESS_to_Load_at_t_6 + 0.00124 Diesel_to_BESS_at_t_6
 + 0.00124 PV_to_BESS_at_t_6 = 0
_C90: PV_to_BESS_at_t_17 + PV_to_Load_at_t_17 = 84.6027362479
_C91: PV_to_BESS_at_t_18 + PV_to_Load_at_t_18 = 35.6406736774
_C92: PV_to_BESS_at_t_19 + PV_to_Load_at_t_19 = 37.4097737604
_C93: PV_to_BESS_at_t_20 + PV_to_Load_at_t_20 = 0
_C94: PV_to_BESS_at_t_21 + PV_to_Load_at_t_21 = 0
_C95: PV_to_BESS_at_t_22 + PV_to_Load_at_t_22 = 0
_C96: PV_to_BESS_at_t_23 + PV_to_Load_at_t_23 = 0
Bounds
 0.2 <= BESS_State_of_Charge_0 <= 0.8
 0.2 <= BESS_State_of_Charge_1 <= 0.8
 0.2 <= BESS_State_of_Charge_10 <= 0.8
 0.2 <= BESS_State_of_Charge_11 <= 0.8
 0.2 <= BESS_State_of_Charge_12 <= 0.8
 0.2 <= BESS_State_of_Charge_13 <= 0.8
 0.2 <= BESS_State_of_Charge_14 <= 0.8
 0.2 <= BESS_State_of_Charge_15 <= 0.8
 0.2 <= BESS_State_of_Charge_16 <= 0.8
 0.2 <= BESS_State_of_Charge_17 <= 0.8
 0.2 <= BESS_State_of_Charge_18 <= 0.8
 0.2 <= BESS_State_of_Charge_19 <= 0.8
 0.2 <= BESS_State_of_Charge_2 <= 0.8
 0.2 <= BESS_State_of_Charge_20 <= 0.8
 0.2 <= BESS_State_of_Charge_21 <= 0.8
 0.2 <= BESS_State_of_Charge_22 <= 0.8
 0.2 <= BESS_State_of_Charge_23 <= 0.8
 0.2 <= BESS_State_of_Charge_3 <= 0.8
 0.2 <= BESS_State_of_Charge_4 <= 0.8
 0.2 <= BESS_State_of_Charge_5 <= 0.8
 0.2 <= BESS_State_of_Charge_6 <= 0.8
 0.2 <= BESS_State_of_Charge_7 <= 0.8
 0.2 <= BESS_State_of_Charge_8 <= 0.8
 0.2 <= BESS_State_of_Charge_9 <= 0.8
End
