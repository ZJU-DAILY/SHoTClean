import numpy as np
import time

from BASE_B import BASE_B
from BASE_B1 import BASE_B1
from BASE_S import BASE_S
from BASE_S1 import BASE_S1
from BASE_P1 import BASE_P1
from BASE_C import BASE_C
from baselines.SCREEN import SCREEN
from baselines.SpeedAcc import SpeedAcc
from tools.entity import TimeSeries, TimePoint
from tools.utils import Assist
from baselines.MTCSC_Uni import MTCSC_Uni
from baselines.LsGreedy import LsGreedy
from baselines.EWMA import EWMA
# from matplotlib import pyplot as plt


def main():
    assist = Assist()

    input_file_name = "SWaT/SWaT.data"
    s_max = 13.0
    s_min = -13.0
    TT = 10

    method_num = 9
    total_drate = np.zeros(10)
    total_rms = np.zeros((10, method_num))
    total_cost = np.zeros((10, method_num))
    total_acc = np.zeros((10, method_num))
    total_time = np.zeros((10, method_num))

    for i in range(1):
        # drate = round(0.05 + 0.025 * i, 3)
        drate = 0.2
        total_drate[i] = drate
        print(f"Dirty rate is {drate}")
        total_dirty_rms = 0
        exp_time = 1

        # sizes = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        size = 1
        # for size in sizes:
        # total_rms = np.zeros((10, method_num))
        # total_cost = np.zeros((10, method_num))
        # total_acc = np.zeros((10, method_num))
        # total_time = np.zeros((10, method_num))
        for j in range(exp_time):
            seed = j + 1

            # MTCSC_Uni
            dirty_series = assist.read_data(input_file_name, ",", size)
            dirty_series = assist.add_noise_random_dimension(dirty_series, drate, seed)
            rmsDirty = assist.calc_rms(dirty_series)
            total_dirty_rms += rmsDirty
            uni_dirty_list = assist.split_by_dimension(dirty_series)
            D = len(uni_dirty_list)
            uni_result_list = []
            time_cost = 0
            for d in range(D):
                uni_dirty = uni_dirty_list[d]
                mtcsc_uni = MTCSC_Uni(uni_dirty, s_max, s_min, TT)
                start_time = time.time()
                result_uni_d = mtcsc_uni.mainScreen()
                end_time = time.time()
                time_cost += (end_time - start_time)
                uni_result_list.append(result_uni_d)
            result_series_mtcsc_uni = TimeSeries()
            timestamps_0 = [tp.get_timestamp() for tp in uni_result_list[0].get_timeseries()]
            for idx, ts in enumerate(timestamps_0):
                val_dims = [uni_result_list[d].get_timeseries()[idx].get_value() for d in range(D)]
                truth_dims = [uni_result_list[d].get_timeseries()[idx].get_truth() for d in range(D)]
                noise_dims = [uni_result_list[d].get_timeseries()[idx].get_noise() for d in range(D)]
                result_series_mtcsc_uni.add_point(TimePoint(ts, val_dims, truth_dims, noise_dims))
            rms_mtcsc_uni = assist.calc_rms(result_series_mtcsc_uni)
            cost_mtcsc_uni = assist.calc_cost(result_series_mtcsc_uni)
            acc_mtcsc_uni = assist.calc_acc(result_series_mtcsc_uni)

            total_rms[i][0] += rms_mtcsc_uni
            total_cost[i][0] += cost_mtcsc_uni
            total_acc[i][0] += acc_mtcsc_uni
            total_time[i][0] += time_cost

            print(f"Total_dirty RMS: {total_dirty_rms}")
            print(f"MTCSC_Uni RMS: {rms_mtcsc_uni}, Cost: {cost_mtcsc_uni}, Acc: {acc_mtcsc_uni}, Time: {time_cost}")

            # SCREEN
            dirty_series = assist.read_data(input_file_name, ",", size)
            dirty_series = assist.add_noise_random_dimension(dirty_series, drate, seed)
            uni_dirty_list = assist.split_by_dimension(dirty_series)
            D = len(uni_dirty_list)
            uni_result_list = []
            time_cost = 0
            for d in range(D):
                uni_dirty = uni_dirty_list[d]
                screen = SCREEN(uni_dirty, s_max, s_min, TT)
                start_time = time.time()
                result_uni_d = screen.main_screen()
                end_time = time.time()
                time_cost += (end_time - start_time)
                uni_result_list.append(result_uni_d)
            result_series_screen = TimeSeries()
            timestamps_0 = [tp.get_timestamp() for tp in uni_result_list[0].get_timeseries()]
            for idx, ts in enumerate(timestamps_0):
                val_dims = [uni_result_list[d].get_timeseries()[idx].get_value() for d in range(D)]
                truth_dims = [uni_result_list[d].get_timeseries()[idx].get_truth() for d in range(D)]
                noise_dims = [uni_result_list[d].get_timeseries()[idx].get_noise() for d in range(D)]
                result_series_screen.add_point(TimePoint(ts, val_dims, truth_dims, noise_dims))
            rms_screen = assist.calc_rms(result_series_screen)
            cost_screen = assist.calc_cost(result_series_screen)
            acc_screen = assist.calc_acc(result_series_screen)

            total_rms[i][1] += rms_screen
            total_cost[i][1] += cost_screen
            total_acc[i][1] += acc_screen
            total_time[i][1] += time_cost

            print(f"SCREEN RMS: {rms_screen}, Cost: {cost_screen}, Acc: {acc_screen}, Time: {time_cost}")

            # SpeedAcc
            dirty_series = assist.read_data(input_file_name, ",", size)
            dirty_series = assist.add_noise_random_dimension(dirty_series, drate, seed)
            uni_dirty_list = assist.split_by_dimension(dirty_series)
            D = len(uni_dirty_list)
            uni_result_list = []
            time_cost = 0
            for d in range(D):
                uni_dirty = uni_dirty_list[d]
                speedacc = SpeedAcc(uni_dirty, TT, s_max, s_min, 4000, -4000)
                start_time = time.time()
                result_uni_d = speedacc.main_sli_up()
                end_time = time.time()
                time_cost += (end_time - start_time)
                uni_result_list.append(result_uni_d)
            result_series_speedacc = TimeSeries()
            timestamps_0 = [tp.get_timestamp() for tp in uni_result_list[0].get_timeseries()]
            for idx, ts in enumerate(timestamps_0):
                val_dims = [uni_result_list[d].get_timeseries()[idx].get_value() for d in range(D)]
                truth_dims = [uni_result_list[d].get_timeseries()[idx].get_truth() for d in range(D)]
                noise_dims = [uni_result_list[d].get_timeseries()[idx].get_noise() for d in range(D)]
                result_series_speedacc.add_point(TimePoint(ts, val_dims, truth_dims, noise_dims))
            rms_speedacc = assist.calc_rms(result_series_speedacc)
            cost_speedacc = assist.calc_cost(result_series_speedacc)
            acc_speedacc = assist.calc_acc(result_series_speedacc)

            total_rms[i][2] += rms_speedacc
            total_cost[i][2] += cost_speedacc
            total_acc[i][2] += acc_speedacc
            total_time[i][2] += time_cost

            print(f"SpeedAcc RMS: {rms_speedacc}, Cost: {cost_speedacc}, Acc: {acc_speedacc}, Time: {time_cost}")

            # EWMA
            dirty_series = assist.read_data(input_file_name, ",", size)
            dirty_series = assist.add_noise_random_dimension(dirty_series, drate, seed)
            uni_dirty_list = assist.split_by_dimension(dirty_series)
            D = len(uni_dirty_list)
            uni_result_list = []
            time_cost = 0
            for d in range(D):
                uni_dirty = uni_dirty_list[d]
                expsmooth = EWMA(uni_dirty, 0.042)
                start_time = time.time()
                result_uni_d = expsmooth.main_exp()
                end_time = time.time()
                time_cost += (end_time - start_time)
                uni_result_list.append(result_uni_d)
            result_series_expsmooth = TimeSeries()
            timestamps_0 = [tp.get_timestamp() for tp in uni_result_list[0].get_timeseries()]
            for idx, ts in enumerate(timestamps_0):
                val_dims = [uni_result_list[d].get_timeseries()[idx].get_value() for d in range(D)]
                truth_dims = [uni_result_list[d].get_timeseries()[idx].get_truth() for d in range(D)]
                noise_dims = [uni_result_list[d].get_timeseries()[idx].get_noise() for d in range(D)]
                result_series_expsmooth.add_point(TimePoint(ts, val_dims, truth_dims, noise_dims))
            rms_expsmooth = assist.calc_rms(result_series_expsmooth)
            cost_expsmooth = assist.calc_cost(result_series_expsmooth)
            acc_expsmooth = assist.calc_acc(result_series_expsmooth)

            total_rms[i][3] += rms_expsmooth
            total_cost[i][3] += cost_expsmooth
            total_acc[i][3] += acc_expsmooth
            total_time[i][3] += time_cost

            print(f"EWMA RMS: {rms_expsmooth}, Cost: {cost_expsmooth}, Acc: {acc_expsmooth}, Time: {time_cost}")

            # LsGreedy
            dirty_series = assist.read_data(input_file_name, ",", size)
            dirty_series = assist.add_noise_random_dimension(dirty_series, drate, seed)
            uni_dirty_list = assist.split_by_dimension(dirty_series)
            D = len(uni_dirty_list)
            uni_result_list = []
            time_cost = 0
            for d in range(D):
                uni_dirty = uni_dirty_list[d]
                lsgreedy = LsGreedy(uni_dirty)
                start_time = time.time()
                result_uni_d = lsgreedy.repair()
                end_time = time.time()
                time_cost += (end_time - start_time)
                uni_result_list.append(result_uni_d)
            result_series_lsgreedy = TimeSeries()
            timestamps_0 = [tp.get_timestamp() for tp in uni_result_list[0].get_timeseries()]
            for idx, ts in enumerate(timestamps_0):
                val_dims = [uni_result_list[d].get_timeseries()[idx].get_value() for d in range(D)]
                truth_dims = [uni_result_list[d].get_timeseries()[idx].get_truth() for d in range(D)]
                noise_dims = [uni_result_list[d].get_timeseries()[idx].get_noise() for d in range(D)]
                result_series_lsgreedy.add_point(TimePoint(ts, val_dims, truth_dims, noise_dims))
            rms_lsgreedy = assist.calc_rms(result_series_lsgreedy)
            cost_lsgreedy = assist.calc_cost(result_series_lsgreedy)
            acc_lsgreedy = assist.calc_acc(result_series_lsgreedy)

            total_rms[i][4] += rms_lsgreedy
            total_cost[i][4] += cost_lsgreedy
            total_acc[i][4] += acc_lsgreedy
            total_time[i][4] += time_cost

            print(f"LsGreedy RMS: {rms_lsgreedy}, Cost: {cost_lsgreedy}, Acc: {acc_lsgreedy}, Time: {time_cost}")

            # BASE B
            dirty_series = assist.read_data(input_file_name, ",", size)
            dirty_series = assist.add_noise_random_dimension(dirty_series, drate, seed)
            BASE_B_Soft = BASE_B(dirty_series, s_max, s_min)
            start_time = time.time()
            result_series_BASE_B_Soft = BASE_B_Soft.clean()
            end_time = time.time()
            rms_BASE_B_Soft = assist.calc_rms(result_series_BASE_B_Soft)
            cost_BASE_B_Soft = assist.calc_cost(result_series_BASE_B_Soft)
            acc_BASE_B_Soft = assist.calc_acc(result_series_BASE_B_Soft)

            total_rms[i][5] += rms_BASE_B_Soft
            total_cost[i][5] += cost_BASE_B_Soft
            total_acc[i][5] += acc_BASE_B_Soft
            total_time[i][5] += end_time - start_time

            print(
                f"BASE B RMS: {rms_BASE_B_Soft}, Cost: {cost_BASE_B_Soft}, Acc: {acc_BASE_B_Soft}, Time: {end_time - start_time}")

            # BASE S-Multi
            dirty_series = assist.read_data(input_file_name, ",", size)
            dirty_series = assist.add_noise_random_dimension(dirty_series, drate, seed)
            BASE_S_Multi_Soft = BASE_S(dirty_series, s_max, s_min, 10, is_soft=True)
            start_time = time.time()
            result_series_BASE_S_M_Soft = BASE_S_Multi_Soft.mainScreen()
            end_time = time.time()
            rms_BASE_S_Soft = assist.calc_rms(result_series_BASE_S_M_Soft)
            cost_BASE_S_Soft = assist.calc_cost(result_series_BASE_S_M_Soft)
            acc_BASE_S_Soft = assist.calc_acc(result_series_BASE_S_M_Soft)

            total_rms[i][6] += rms_BASE_S_Soft
            total_cost[i][6] += cost_BASE_S_Soft
            total_acc[i][6] += acc_BASE_S_Soft
            total_time[i][6] += end_time - start_time

            print(f"BASE S-Multi RMS: {rms_BASE_S_Soft}, Cost: {cost_BASE_S_Soft}, Acc: {acc_BASE_S_Soft}, Time: {end_time - start_time}")

            # BASE P-Single
            dirty_series = assist.read_data(input_file_name, ",", size)
            dirty_series = assist.add_noise_random_dimension(dirty_series, drate, seed)
            uni_dirty_list = assist.split_by_dimension(dirty_series)
            D = len(uni_dirty_list)
            uni_result_list = []
            time_cost = 0
            for d in range(D):
                uni_dirty = uni_dirty_list[d]
                BASE_P_Soft = BASE_P1(uni_dirty, s_max, s_min, 10, is_soft=True)
                start_time = time.time()
                result_uni_d = BASE_P_Soft.mainScreen()
                end_time = time.time()
                time_cost += (end_time - start_time)
                uni_result_list.append(result_uni_d)
            result_series_BASE_P_Soft = TimeSeries()
            timestamps_0 = [tp.get_timestamp() for tp in uni_result_list[0].get_timeseries()]
            for idx, ts in enumerate(timestamps_0):
                val_dims = [uni_result_list[d].get_timeseries()[idx].get_value() for d in range(D)]
                truth_dims = [uni_result_list[d].get_timeseries()[idx].get_truth() for d in range(D)]
                noise_dims = [uni_result_list[d].get_timeseries()[idx].get_noise() for d in range(D)]
                result_series_BASE_P_Soft.add_point(TimePoint(ts, val_dims, truth_dims, noise_dims))

            rms_BASE_P_Soft = assist.calc_rms(result_series_BASE_P_Soft)
            cost_BASE_P_Soft = assist.calc_cost(result_series_BASE_P_Soft)
            acc_BASE_P_Soft = assist.calc_acc(result_series_BASE_P_Soft)
            total_rms[i][7] += rms_BASE_P_Soft
            total_cost[i][7] += cost_BASE_P_Soft
            total_acc[i][7] += acc_BASE_P_Soft
            total_time[i][7] += time_cost

            print(
                f"BASE P-Single RMS: {rms_BASE_P_Soft}, Cost: {cost_BASE_P_Soft}, Acc: {acc_BASE_P_Soft}, Time: {time_cost}")


            # BASE C
            dirty_series = assist.read_data(input_file_name, ",", size)
            dirty_series = assist.add_noise_random_dimension(dirty_series, drate, seed)
            BASE_C_Soft = BASE_C(dirty_series, s_max, s_min, 10, is_soft=True)
            start_time = time.time()
            result_series_BASE_C_Soft = BASE_C_Soft.mainScreen()
            end_time = time.time()
            rms_BASE_C_Soft = assist.calc_rms(result_series_BASE_C_Soft)
            cost_BASE_C_Soft = assist.calc_cost(result_series_BASE_C_Soft)
            acc_BASE_C_Soft = assist.calc_acc(result_series_BASE_C_Soft)

            total_rms[i][8] += rms_BASE_C_Soft
            total_cost[i][8] += cost_BASE_C_Soft
            total_acc[i][8] += acc_BASE_C_Soft
            total_time[i][8] += end_time - start_time

            print(
                f"BASE C RMS: {rms_BASE_C_Soft}, Cost: {cost_BASE_C_Soft}, Acc: {acc_BASE_C_Soft}, Time: {end_time - start_time}")

                # dirty_series = assist.read_data(input_file_name, ",")
                # dirty_series = assist.add_noise_random_dimension(dirty_series, drate, seed)
                # assist.save_data(dirty_series, f"../results/Multi/PSM/Dirty_{drate}.data")
                # assist.save_data(result_series_mtcsc_uni, f"../results/Multi/PSM/MTCSC_{drate}.data")
                # assist.save_data(result_series_screen, f"../results/Multi/PSM/SCREEN_{drate}.data")
                # assist.save_data(result_series_speedacc, f"../results/Multi/PSM/SpeedAcc_{drate}.data")
                # assist.save_data(result_series_expsmooth, f"../results/Multi/PSM/EWMA_{drate}.data")
                # assist.save_data(result_series_lsgreedy, f"../results/Multi/PSM/LsGreedy_{drate}.data")
                # assist.save_data(result_series_BASE_B_Soft, f"../results/Multi/PSM/B-Soft_{drate}.data")
                # assist.save_data(result_series_BASE_C_Soft, f"../results/Multi/PSM/C-Soft_{drate}.data")
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
    name = ["Methods" ,"MTCSC_Uni", "SCREEN", "SpeedAcc", "EWMA", "LsGreedy",  "B-Soft", "S-Multi-Soft", "P-Single-Soft", "C-Soft"]
    write_file_name = f"../results/Multi/SWaT/RMS_Partial.csv"
    assist.write_csv(write_file_name, name, total_drate, total_rms)
    write_file_name = f"../results/Multi/SWaT/COST_Partial.csv"
    assist.write_csv(write_file_name, name, total_drate, total_cost)
    write_file_name = f"../results/Multi/SWaT/ACC_Partial.csv"
    assist.write_csv(write_file_name, name, total_drate, total_acc)
    write_file_name = f"../results/Multi/SWaT/TIME_Partial.csv"
    assist.write_csv(write_file_name, name, total_drate, total_time)

if __name__ == "__main__":
    main()
