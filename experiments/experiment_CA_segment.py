import numpy as np
import time

from BASE_P1 import BASE_P1
from BASE_B1 import BASE_B1
from BASE_S1 import BASE_S1
from baselines.SCREEN import SCREEN
from baselines.SpeedAcc import SpeedAcc
from tools.utils import Assist_Single
from baselines.MTCSC_Uni import MTCSC_Uni
from baselines.LsGreedy import LsGreedy
from baselines.EWMA import EWMA
from matplotlib import pyplot as plt


def main():
    assist = Assist_Single()

    input_file_name = "CA/CA.csv"
    s_max = 3000
    s_min = -3000
    TT = 100
    T = 100

    method_num = 11
    total_drate = np.zeros(10)
    total_rms = np.zeros((10, method_num))
    total_cost = np.zeros((10, method_num))
    total_acc = np.zeros((10, method_num))
    total_time = np.zeros((10, method_num))

    for i in range(1):
        # drate = round(0.05 + 0.025 * i, 3)
        drate = 0.20
        total_drate[i] = drate
        print(f"Dirty rate is {drate}")
        total_dirty_rms = 0
        exp_time = 1

        for j in range(exp_time):
            seed = j + 1

            # MTCSC_Uni
            dirty_series = assist.read_data(input_file_name, ",")
            dirty_series = assist.add_noise_continuous(dirty_series, drate, seed)
            rmsDirty = assist.calc_rms(dirty_series)
            total_dirty_rms += rmsDirty
            mtcsc_uni = MTCSC_Uni(dirty_series, s_max, s_min, T)
            start_time = time.time()
            result_series_mtcsc_uni = mtcsc_uni.mainScreen()
            end_time = time.time()
            rms_mtcsc_uni = assist.calc_rms(result_series_mtcsc_uni)
            cost_mtcsc_uni = assist.calc_cost(result_series_mtcsc_uni)
            acc_mtcsc_uni = assist.calc_acc(result_series_mtcsc_uni)

            total_rms[i][0] += rms_mtcsc_uni
            total_cost[i][0] += cost_mtcsc_uni
            total_acc[i][0] += acc_mtcsc_uni
            total_time[i][0] += (end_time - start_time)

            # SCREEN
            dirty_series = assist.read_data(input_file_name, ",")
            dirty_series = assist.add_noise_continuous(dirty_series, drate, seed)
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
            dirty_series = assist.add_noise_continuous(dirty_series, drate, seed)
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
            dirty_series = assist.add_noise_continuous(dirty_series, drate, seed)
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
            dirty_series = assist.add_noise_continuous(dirty_series, drate, seed)
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

            # BASE G HARD
            # dirty_series = assist.read_data(input_file_name, ",")
            # dirty_series = assist.add_noise_continuous(dirty_series, drate, seed)
            # BASE_Hard = BASE_B1(dirty_series, s_max, s_min, is_soft=False)
            # start_time = time.time()
            # result_series_BASE_Hard = BASE_Hard.clean()
            # end_time = time.time()
            # rms_BASE_Hard = assist.calc_rms(result_series_BASE_Hard)
            # cost_BASE_Hard = assist.calc_cost(result_series_BASE_Hard)
            # acc_BASE_Hard = assist.calc_acc(result_series_BASE_Hard)
            #
            # total_rms[i][5] += rms_BASE_Hard
            # total_cost[i][5] += cost_BASE_Hard
            # total_acc[i][5] += acc_BASE_Hard
            # total_time[i][5] += (end_time - start_time)

            # BASE G WITH SOFT
            dirty_series = assist.read_data(input_file_name, ",")
            dirty_series = assist.add_noise_continuous(dirty_series, drate, seed)
            BASE_Soft = BASE_B1(dirty_series, s_max, s_min, is_soft=True)
            start_time = time.time()
            result_series_BASE_Soft = BASE_Soft.clean()
            end_time = time.time()
            rms_BASE_Soft = assist.calc_rms(result_series_BASE_Soft)
            cost_BASE_Soft = assist.calc_cost(result_series_BASE_Soft)
            acc_BASE_Soft = assist.calc_acc(result_series_BASE_Soft)

            total_rms[i][6] += rms_BASE_Soft
            total_cost[i][6] += cost_BASE_Soft
            total_acc[i][6] += acc_BASE_Soft
            total_time[i][6] += (end_time - start_time)

            # BASE L WITH HARD
            # dirty_series = assist.read_data(input_file_name, ",")
            # dirty_series = assist.add_noise_continuous(dirty_series, drate, seed)
            # BASE_L_Hard = BASE_S1(dirty_series, s_max, s_min, T, is_soft=False)
            # start_time = time.time()
            # result_series_BASE_L_Hard = BASE_L_Hard.mainScreen()
            # end_time = time.time()
            # rms_BASE_L_Hard = assist.calc_rms(result_series_BASE_L_Hard)
            # cost_BASE_L_Hard = assist.calc_cost(result_series_BASE_L_Hard)
            # acc_BASE_L_Hard = assist.calc_acc(result_series_BASE_L_Hard)
            #
            # total_rms[i][7] += rms_BASE_L_Hard
            # total_cost[i][7] += cost_BASE_L_Hard
            # total_acc[i][7] += acc_BASE_L_Hard
            # total_time[i][7] += (end_time - start_time)

            # BASE L WITH SOFT
            dirty_series = assist.read_data(input_file_name, ",")
            dirty_series = assist.add_noise_continuous(dirty_series, drate, seed)
            BASE_L_Soft = BASE_S1(dirty_series, s_max, s_min, T, is_soft=True)
            start_time = time.time()
            result_series_BASE_L_Soft = BASE_L_Soft.mainScreen()
            end_time = time.time()
            rms_BASE_L_Soft = assist.calc_rms(result_series_BASE_L_Soft)
            cost_BASE_L_Soft = assist.calc_cost(result_series_BASE_L_Soft)
            acc_BASE_L_Soft = assist.calc_acc(result_series_BASE_L_Soft)

            total_rms[i][8] += rms_BASE_L_Soft
            total_cost[i][8] += cost_BASE_L_Soft
            total_acc[i][8] += acc_BASE_L_Soft
            total_time[i][8] += (end_time - start_time)

            # BASE A WITH HARD
            # dirty_series = assist.read_data(input_file_name, ",")
            # dirty_series = assist.add_noise_continuous(dirty_series, drate, seed)
            # BASE_A_Hard = BASE_P1(dirty_series, s_max, s_min, T, is_soft=False)
            # start_time = time.time()
            # result_series_BASE_A_Hard = BASE_A_Hard.mainScreen()
            # end_time = time.time()
            # rms_BASE_A_Hard = assist.calc_rms(result_series_BASE_A_Hard)
            # cost_BASE_A_Hard = assist.calc_cost(result_series_BASE_A_Hard)
            # acc_BASE_A_Hard = assist.calc_acc(result_series_BASE_A_Hard)
            #
            # total_rms[i][9] += rms_BASE_A_Hard
            # total_cost[i][9] += cost_BASE_A_Hard
            # total_acc[i][9] += acc_BASE_A_Hard
            # total_time[i][9] += (end_time - start_time)

            # BASE A WITH SOFT
            dirty_series = assist.read_data(input_file_name, ",")
            dirty_series = assist.add_noise_continuous(dirty_series, drate, seed)
            BASE_A_Soft = BASE_P1(dirty_series, s_max, s_min, T, is_soft=True)
            start_time = time.time()
            result_series_BASE_A_Soft = BASE_A_Soft.mainScreen()
            end_time = time.time()
            rms_BASE_A_Soft = assist.calc_rms(result_series_BASE_A_Soft)
            cost_BASE_A_Soft = assist.calc_cost(result_series_BASE_A_Soft)
            acc_BASE_A_Soft = assist.calc_acc(result_series_BASE_A_Soft)

            total_rms[i][10] += rms_BASE_A_Soft
            total_cost[i][10] += cost_BASE_A_Soft
            total_acc[i][10] += acc_BASE_A_Soft
            total_time[i][10] += (end_time - start_time)

            # plt.figure(figsize=(12, 6))
            # dirty_series = assist.read_data(input_file_name, ",")
            # dirty_series = assist.add_noise_continuous(dirty_series, drate, seed)
            # ori = [p.get_value() for p in dirty_series.get_timeseries()]
            # truth = [p.get_truth() for p in dirty_series.get_timeseries()]
            # hard = [p.get_value() for p in result_series_BASE_Hard.get_timeseries()]
            # soft = [p.get_value() for p in result_series_BASE_Soft.get_timeseries()]
            # mtcsc = [p.get_value() for p in result_series_mtcsc_uni.get_timeseries()]
            # lsgreedy = [p.get_value() for p in result_series_lsgreedy.get_timeseries()]
            # plt.plot(truth[:5000], 'r-', label='Truth')
            # plt.plot(ori[:5000], 'g--', label='Noise')
            # # plt.plot(hard, 'b-', label='Hard')
            # # plt.plot(soft[:5000], 'r-', alpha=0.7, label='Soft')
            # # plt.plot(mtcsc, 'purple', label='MTCSC_Uni')
            # # plt.plot(lsgreedy, 'b-', label='LsGreedy')
            # plt.legend()
            # plt.show()
            # exit()

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
    name = ["Methods" ,"MTCSC_Uni", "SCREEN", "SpeedAcc", "EWMA", "LsGreedy", "B-Hard", "B-Soft", "S-Hard", "S-Soft", "P-Hard", "P-Soft"]
    write_file_name = "../results/One/CA/RMS_segment.csv"
    assist.write_csv(write_file_name, name, total_drate, total_rms)
    write_file_name = "../results/One/CA/COST_segment.csv"
    assist.write_csv(write_file_name, name, total_drate, total_cost)
    write_file_name = "../results/One/CA/ACC_segment.csv"
    assist.write_csv(write_file_name, name, total_drate, total_acc)
    write_file_name = "../results/One/CA/TIME_segment.csv"
    assist.write_csv(write_file_name, name, total_drate, total_time)

if __name__ == "__main__":
    main()
