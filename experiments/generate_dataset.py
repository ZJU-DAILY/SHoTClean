import numpy as np
import pandas as pd
from tqdm import tqdm

from tools.utils import Assist

def porto_process():
    trajectories = pd.read_csv("datasets/Porto/porto.csv", header=0, usecols=['POLYLINE', 'TIMESTAMP'])
    cnt = 0
    for _, (idx, traj) in enumerate(
            tqdm(trajectories.iterrows(), total=len(trajectories), desc="Processing trajectories")):
        polyline = eval(traj["POLYLINE"])
        if len(polyline) > 1200:
            cnt += 1
            if cnt > 10:
                break
            data = pd.DataFrame([{'lon': point[0], 'lat': point[1]} for point in polyline])
            ids = np.arange(1, len(data) + 1).reshape(-1, 1)
            combined_data = np.hstack([ids, data, data])
            output_file_name = f"datasets/Porto/porto_{cnt}.data"
            fmt = '%d,' + '%.4f,' * (combined_data.shape[1] - 1)
            fmt = fmt.rstrip(',')
            np.savetxt(output_file_name, combined_data, delimiter=',', fmt=fmt)
    print(cnt)

def KPI():
    data = pd.read_csv("datasets/KPI/train.csv", header=0, usecols=['timestamp', 'value', 'KPI ID'])
    data = data[data['KPI ID'] == 'a40b1df87e3f1c87']
    data = data[['value']]
    data = data[data['value'] != 0]
    # data = data[:10000]
    ids = np.arange(1, len(data) + 1).reshape(-1, 1)
    combined_data = np.hstack([ids, data, data])
    np.savetxt('../datasets/KPI/KPI.data', combined_data, delimiter=',', fmt='%d')
    print(data)

def FRED():
    data = pd.read_csv("datasets/FRED/TOTALSA.csv", header=0)
    data = data.drop(columns=['observation_date'])
    ids = np.arange(1, len(data) + 1).reshape(-1, 1)
    combined_data = np.hstack([ids, data, data])
    fmt = '%d,' + '%.1f,' * (combined_data.shape[1] - 1)
    fmt = fmt.rstrip(',')
    np.savetxt('../datasets/FRED/TOTALSA.data', combined_data, delimiter=',', fmt=fmt)
    print(data)

def SWaT():
    data = pd.read_excel("datasets/SWaT/SWaT_dataset_Jul 19 v2.xlsx", skiprows=1, header=0)
    data = data[['LIT 101', 'AIT 201', 'AIT 202', 'AIT 203', 'FIT 201', 'AIT 301', 'AIT 302', 'AIT 303', 'DPIT 301',
                 'FIT 301', 'LIT 301', 'AIT 402', 'FIT 401', 'LIT 401', 'AIT 501', 'AIT 502', 'AIT 503', 'AIT 504',
                 'FIT 501', 'FIT 502', 'FIT 503', 'FIT 504', 'PIT 501', 'PIT 502', 'PIT 503', 'FIT 601']]
    data = data[1:1000]
    ids = np.arange(1, len(data) + 1).reshape(-1, 1)
    combined_data = np.hstack([ids, data, data])
    fmt = '%d,' + '%.4f,' * (combined_data.shape[1] - 1)
    fmt = fmt.rstrip(',')
    np.savetxt('../datasets/SWaT/SWaT1k.data', combined_data, delimiter=',', fmt=fmt)
    print(data)

def WADI():
    data = pd.read_csv("datasets/WADI/WADI_14days_new.csv", header=0)
    data = data.drop(columns=['Row', 'Date', 'Time'])
    cols_to_keep = [
        '1_AIT_001_PV', '1_AIT_002_PV', '1_AIT_003_PV', '1_AIT_004_PV', '1_AIT_005_PV',
        '1_FIT_001_PV', '1_LT_001_PV', '2_DPIT_001_PV', '2_FIC_101_CO', '2_FIC_101_PV',
        '2_FIC_101_SP', '2_FIC_201_CO', '2_FIC_201_PV', '2_FIC_201_SP', '2_FIC_301_CO',
        '2_FIC_301_PV', '2_FIC_301_SP', '2_FIC_401_CO', '2_FIC_401_PV', '2_FIC_401_SP',
        '2_FIC_501_CO', '2_FIC_501_PV', '2_FIC_501_SP', '2_FIC_601_CO', '2_FIC_601_PV',
        '2_FIC_601_SP', '2_FIT_001_PV', '2_FIT_002_PV', '2_FIT_003_PV', '2_FQ_101_PV',
        '2_FQ_201_PV', '2_FQ_301_PV', '2_FQ_401_PV', '2_FQ_501_PV', '2_FQ_601_PV',
        '2_LT_001_PV', '2_LT_002_PV', '2_MCV_101_CO', '2_MCV_201_CO', '2_MCV_301_CO',
        '2_MCV_401_CO', '2_MCV_501_CO', '2_MCV_601_CO', '2_P_003_SPEED', '2_P_004_SPEED',
        '2_P_004_STATUS', '2_PIC_003_CO', '2_PIC_003_PV', '2_PIC_003_SP', '2_PIT_001_PV',
        '2_PIT_002_PV', '2_PIT_003_PV', '2A_AIT_001_PV', '2A_AIT_002_PV', '2A_AIT_003_PV',
        '2A_AIT_004_PV', '2B_AIT_001_PV', '2B_AIT_002_PV', '2B_AIT_003_PV', '2B_AIT_004_PV',
        '3_AIT_001_PV', '3_AIT_003_PV', '3_AIT_004_PV', '3_AIT_005_PV', '3_FIT_001_PV',
        '3_LS_001_AL', '3_LT_001_PV', '3_MV_001_STATUS', '3_MV_002_STATUS', '3_MV_003_STATUS',
        '3_P_001_STATUS', '3_P_002_STATUS', '3_P_003_STATUS', '3_P_004_STATUS',
        'LEAK_DIFF_PRESSURE', 'PLANT_START_STOP_LOG', 'TOTAL_CONS_REQUIRED_FLOW'
    ]
    data = data[cols_to_keep]
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna(axis=1, how='any')
    ids = np.arange(1, len(data) + 1).reshape(-1, 1)
    combined_data = np.hstack([ids, data, data])
    fmt = '%d,' + '%.4f,' * (combined_data.shape[1] - 1)
    fmt = fmt.rstrip(',')
    np.savetxt('../datasets/WADI/WADI.data', combined_data, delimiter=',', fmt=fmt)
    print(data)

def UCI():
    filepath = "datasets/UCI/energydata_complete.csv"
    data = pd.read_csv(filepath, header=0)
    data = data.drop(columns=['date'])
    keep_rows = ['Appliances', 'T1', 'RH_1', 'T2', 'RH_2', 'T3', 'RH_3', 'T4', 'RH_4', 'T5', 'T6', 'T7', 'RH_7', 'T8', 'RH_8', 'T9', 'RH_9', 'T_out', 'Press_mm_hg', 'RH_out', 'Windspeed', 'Tdewpoint']
    data = data[keep_rows]
    ids = np.arange(1, len(data) + 1).reshape(-1, 1)
    combined_data = np.hstack([ids, data, data])
    fmt = '%d,' + '%.1f,' * (combined_data.shape[1] - 1)
    fmt = fmt.rstrip(',')
    np.savetxt('datasets/UCI/AEP_new.data', combined_data, delimiter=',', fmt=fmt)
    print(data)

def exchange():
    data = pd.read_csv(
        'datasets/exchange/exchange_rate.txt.gz',
        sep=',',
        compression='gzip',
        header=None,
        encoding='utf-8'
    )
    ids = np.arange(1, len(data) + 1).reshape(-1, 1)
    combined_data = np.hstack([ids, data, data])
    fmt = '%d,' + '%.6f,' * (combined_data.shape[1] - 1)
    fmt = fmt.rstrip(',')
    np.savetxt('../datasets/exchange/exchange.data', combined_data, delimiter=',', fmt=fmt)
    print(data)


if __name__ == "__main__":
    assist = Assist()
    # input_file_name = "FRED/TOTALSA.data"
    # input_file_name = "stock/stock12k.data"
    # input_file_name = "CA/CA.csv"
    # input_file_name = "WADI/WADI.data"
    # input_file_name = "Porto/porto_10.data"
    # input_file_name = "PSM/PSM.data"
    # input_file_name = "UCI/AEP.data"
    # input_file_name = "SWaT/SWaT.data"
    # input_file_name = "exchange/exchange.data"
    # input_file_name = "KPI/KPI.data"
    UCI()