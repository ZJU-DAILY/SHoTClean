import numpy as np
import time

from SHoTClean_B1 import SHoTClean_B1
from SHoTClean_S1 import SHoTClean_S1
from SHoTClean_P1 import SHoTClean_P1
from baselines.SCREEN import SCREEN
from baselines.SpeedAcc import SpeedAcc
from tools.utils import Assist_Single
from baselines.MTCSC import MTCSC
from baselines.LsGreedy import LsGreedy
from baselines.EWMA import EWMA


def main():
    assist = Assist_Single()

    input_file_name = "FRED/TOTALSA.data"
    s_max = 2
    s_min = -2
    TT = 10
    T = 10

    method_num = 11
    total_drate = np.zeros(10)
    total_rms = np.zeros((10, method_num))
    total_cost = np.zeros((10, method_num))
    total_acc = np.zeros((10, method_num))
    total_time = np.zeros((10, method_num))

    for i in range(10):
        # drate = 0.80
        drate = round(0.05 + 0.025 * i, 3)
        total_drate[i] = drate
        print(f"Dirty rate is {drate}")
        total_dirty_rms = 0
        exp_time = 10

        for j in range(exp_time):
            seed = j + 1

            # MTCSC
            dirty_series = assist.read_data(input_file_name, ",")
            dirty_series = assist.add_noise(dirty_series, drate, seed)
            rmsDirty = assist.calc_rms(dirty_series)
            total_dirty_rms += rmsDirty
            mtcsc = MTCSC(dirty_series, s_max, s_min, T)
            start_time = time.time()
            result_series_MTCSC = mtcsc.mainScreen()
            end_time = time.time()
            rms_MTCSC = assist.calc_rms(result_series_MTCSC)
            cost_MTCSC = assist.calc_cost(result_series_MTCSC)
            acc_MTCSC = assist.calc_acc(result_series_MTCSC)

            total_rms[i][0] += rms_MTCSC
            total_cost[i][0] += cost_MTCSC
            total_acc[i][0] += acc_MTCSC
            total_time[i][0] += (end_time - start_time)

            # SCREEN
            dirty_series = assist.read_data(input_file_name, ",")
            dirty_series = assist.add_noise(dirty_series, drate, seed)
            screen = SCREEN(dirty_series, s_max, s_min, TT)
            start_time = time.time()
            result_series_screen = screen.main_screen()
            end_time = time.time()
            rms_screen = assist.calc_rms(result_series_screen)
            cost_screen = assist.calc_cost(result_series_screen)
            acc_screen = assist.calc_acc(result_series_screen)

            total_rms[i][1] += rms_screen
            total_cost[i][1] += cost_screen
            total_acc[i][1] += acc_screen
            total_time[i][1] += (end_time - start_time)

            # SpeedAcc
            dirty_series = assist.read_data(input_file_name, ",")
            dirty_series = assist.add_noise(dirty_series, drate, seed)
            speedacc = SpeedAcc(dirty_series, TT, s_max, s_min, 4000, -4000)
            start_time = time.time()
            result_series_speedacc = speedacc.main_sli_up()
            end_time = time.time()
            rms_speedacc = assist.calc_rms(result_series_speedacc)
            cost_speedacc = assist.calc_cost(result_series_speedacc)
            acc_speedacc = assist.calc_acc(result_series_speedacc)

            total_rms[i][2] += rms_speedacc
            total_cost[i][2] += cost_speedacc
            total_acc[i][2] += acc_speedacc
            total_time[i][2] += (end_time - start_time)

            # EWMA
            dirty_series = assist.read_data(input_file_name, ",")
            dirty_series = assist.add_noise(dirty_series, drate, seed)
            expsmooth = EWMA(dirty_series, 0.042)
            start_time = time.time()
            result_series_expsmooth = expsmooth.main_exp()
            end_time = time.time()
            rms_expsmooth = assist.calc_rms(result_series_expsmooth)
            cost_expsmooth = assist.calc_cost(result_series_expsmooth)
            acc_expsmooth = assist.calc_acc(result_series_expsmooth)

            total_rms[i][3] += rms_expsmooth
            total_cost[i][3] += cost_expsmooth
            total_acc[i][3] += acc_expsmooth
            total_time[i][3] += (end_time - start_time)

            # LsGreedy
            dirty_series = assist.read_data(input_file_name, ",")
            dirty_series = assist.add_noise(dirty_series, drate, seed)
            lsgreedy = LsGreedy(dirty_series)
            start_time = time.time()
            result_series_lsgreedy = lsgreedy.repair()
            end_time = time.time()
            rms_lsgreedy = assist.calc_rms(result_series_lsgreedy)
            cost_lsgreedy = assist.calc_cost(result_series_lsgreedy)
            acc_lsgreedy = assist.calc_acc(result_series_lsgreedy)

            total_rms[i][4] += rms_lsgreedy
            total_cost[i][4] += cost_lsgreedy
            total_acc[i][4] += acc_lsgreedy
            total_time[i][4] += (end_time - start_time)

            # SHoTClean-B WITH SOFT
            dirty_series = assist.read_data(input_file_name, ",")
            dirty_series = assist.add_noise(dirty_series, drate, seed)
            SHoTClean_Soft = SHoTClean_B1(dirty_series, s_max, s_min, is_soft=True)
            start_time = time.time()
            result_series_SHoTClean_Soft = SHoTClean_Soft.clean()
            end_time = time.time()
            rms_SHoTClean_Soft = assist.calc_rms(result_series_SHoTClean_Soft)
            cost_SHoTClean_Soft = assist.calc_cost(result_series_SHoTClean_Soft)
            acc_SHoTClean_Soft = assist.calc_acc(result_series_SHoTClean_Soft)

            total_rms[i][6] += rms_SHoTClean_Soft
            total_cost[i][6] += cost_SHoTClean_Soft
            total_acc[i][6] += acc_SHoTClean_Soft
            total_time[i][6] += (end_time - start_time)

            # SHoTClean-S WITH SOFT
            dirty_series = assist.read_data(input_file_name, ",")
            dirty_series = assist.add_noise(dirty_series, drate, seed)
            SHoTClean_S = SHoTClean_S1(dirty_series, s_max, s_min, 10, is_soft=True)
            start_time = time.time()
            result_series_SHoTClean_S = SHoTClean_S.mainScreen()
            end_time = time.time()
            rms_SHoTClean_S = assist.calc_rms(result_series_SHoTClean_S)
            cost_SHoTClean_S = assist.calc_cost(result_series_SHoTClean_S)
            acc_SHoTClean_S = assist.calc_acc(result_series_SHoTClean_S)

            total_rms[i][8] += rms_SHoTClean_S
            total_cost[i][8] += cost_SHoTClean_S
            total_acc[i][8] += acc_SHoTClean_S
            total_time[i][8] += (end_time - start_time)

            # SHoTClean-P WITH SOFT
            dirty_series = assist.read_data(input_file_name, ",")
            dirty_series = assist.add_noise(dirty_series, drate, seed)
            SHoTClean_P = SHoTClean_P1(dirty_series, s_max, s_min, 10, is_soft=True)
            start_time = time.time()
            result_series_SHoTClean_P = SHoTClean_P.mainScreen()
            end_time = time.time()
            rms_SHoTClean_P = assist.calc_rms(result_series_SHoTClean_P)
            cost_SHoTClean_P = assist.calc_cost(result_series_SHoTClean_P)
            acc_SHoTClean_P = assist.calc_acc(result_series_SHoTClean_P)

            total_rms[i][10] += rms_SHoTClean_P
            total_cost[i][10] += cost_SHoTClean_P
            total_acc[i][10] += acc_SHoTClean_P
            total_time[i][10] += (end_time - start_time)

        total_dirty_rms /= exp_time
        print(f"Dirty RMS error is {round(total_dirty_rms, 3)}")

        # Output results
        for j in range(method_num):
            total_rms[i][j] /= exp_time
            total_cost[i][j] /= exp_time
            total_acc[i][j] /= exp_time
            total_time[i][j] /= exp_time

    print(total_rms)
    print(total_cost)
    print(total_acc)
    print(total_time)
    name = ["Methods" ,"MTCSC", "SCREEN", "SpeedAcc", "EWMA", "LsGreedy", "B-Hard", "B-Soft", "S-Hard", "S-Soft", "P-Hard", "P-Soft"]
    write_file_name = "../results/One/FRED/RMS.csv"
    assist.write_csv(write_file_name, name, total_drate, total_rms)
    write_file_name = "../results/One/FRED/COST.csv"
    assist.write_csv(write_file_name, name, total_drate, total_cost)
    write_file_name = "../results/One/FRED/ACC.csv"
    assist.write_csv(write_file_name, name, total_drate, total_acc)
    write_file_name = "../results/One/FRED/TIME.csv"
    assist.write_csv(write_file_name, name, total_drate, total_time)

if __name__ == "__main__":
    main()
