import numpy as np
import time

from SHoTClean_B import SHoTClean_B
from SHoTClean_C import SHoTClean_C
from SHoTClean_S import SHoTClean_S
from SHoTClean_P1 import SHoTClean_P1
from baselines.SCREEN import SCREEN
from baselines.SpeedAcc import SpeedAcc
from tools.entity import TimeSeries, TimePoint
from tools.utils import Assist
from baselines.MTCSC import MTCSC
from baselines.LsGreedy import LsGreedy
from baselines.EWMA import EWMA


def main():
    assist = Assist()

    input_file_name = "exchange/exchange.data"
    s_max = 0.1
    s_min = -0.1
    TT = 10

    method_num = 9
    total_drate = np.zeros(10)
    total_rms = np.zeros((10, method_num))
    total_cost = np.zeros((10, method_num))
    total_acc = np.zeros((10, method_num))
    total_time = np.zeros((10, method_num))

    for i in range(10):
        drate = round(0.05 + 0.025 * i, 3)
        # drate = 0.80
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
            uni_dirty_list = assist.split_by_dimension(dirty_series)
            D = len(uni_dirty_list)  
            uni_result_list = []
            time_cost = 0
            for d in range(D):
                uni_dirty = uni_dirty_list[d]
                mtcsc = MTCSC(uni_dirty, s_max, s_min, TT)
                start_time = time.time()
                result_uni_d = mtcsc.mainScreen()
                end_time = time.time()
                time_cost += (end_time - start_time)
                uni_result_list.append(result_uni_d)
            result_series_MTCSC = TimeSeries()
            timestamps_0 = [tp.get_timestamp() for tp in uni_result_list[0].get_timeseries()]
            for idx, ts in enumerate(timestamps_0):
                val_dims = [uni_result_list[d].get_timeseries()[idx].get_value() for d in range(D)]
                truth_dims = [uni_result_list[d].get_timeseries()[idx].get_truth() for d in range(D)]
                noise_dims = [uni_result_list[d].get_timeseries()[idx].get_noise() for d in range(D)]
                result_series_MTCSC.add_point(TimePoint(ts, val_dims, truth_dims, noise_dims))
            rms_MTCSC = assist.calc_rms(result_series_MTCSC)
            cost_MTCSC = assist.calc_cost(result_series_MTCSC)
            acc_MTCSC = assist.calc_acc(result_series_MTCSC)

            total_rms[i][0] += rms_MTCSC
            total_cost[i][0] += cost_MTCSC
            total_acc[i][0] += acc_MTCSC
            total_time[i][0] += time_cost

            print(f"Total_dirty RMS: {total_dirty_rms}")
            print(f"MTCSC RMS: {rms_MTCSC}, Cost: {cost_MTCSC}, Acc: {acc_MTCSC}, Time: {time_cost}")
            # exit()

            # SCREEN
            dirty_series = assist.read_data(input_file_name, ",")
            dirty_series = assist.add_noise(dirty_series, drate, seed)
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
            dirty_series = assist.read_data(input_file_name, ",")
            dirty_series = assist.add_noise(dirty_series, drate, seed)
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
            dirty_series = assist.read_data(input_file_name, ",")
            dirty_series = assist.add_noise(dirty_series, drate, seed)
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
            dirty_series = assist.read_data(input_file_name, ",")
            dirty_series = assist.add_noise(dirty_series, drate, seed)
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
            dirty_series = assist.read_data(input_file_name, ",")
            dirty_series = assist.add_noise(dirty_series, drate, seed)
            SHoTClean_B_Soft = SHoTClean_B(dirty_series, s_max, s_min)
            start_time = time.time()
            result_series_SHoTClean_B_Soft = SHoTClean_B_Soft.clean()
            end_time = time.time()
            rms_SHoTClean_B_Soft = assist.calc_rms(result_series_SHoTClean_B_Soft)
            cost_SHoTClean_B_Soft = assist.calc_cost(result_series_SHoTClean_B_Soft)
            acc_SHoTClean_B_Soft = assist.calc_acc(result_series_SHoTClean_B_Soft)

            total_rms[i][5] += rms_SHoTClean_B_Soft
            total_cost[i][5] += cost_SHoTClean_B_Soft
            total_acc[i][5] += acc_SHoTClean_B_Soft
            total_time[i][5] += end_time - start_time
            print(
                f"BASE B RMS: {rms_SHoTClean_B_Soft}, Cost: {cost_SHoTClean_B_Soft}, Acc: {acc_SHoTClean_B_Soft}, Time: {end_time - start_time}")

            # BASE S-Multi
            dirty_series = assist.read_data(input_file_name, ",")
            dirty_series = assist.add_noise(dirty_series, drate, seed)
            SHoTClean_S_Multi_Soft = SHoTClean_S(dirty_series, s_max, s_min, 10, is_soft=True)
            start_time = time.time()
            result_series_SHoTClean_S_M_Soft = SHoTClean_S_Multi_Soft.mainScreen()
            end_time = time.time()
            rms_SHoTClean_S_Soft = assist.calc_rms(result_series_SHoTClean_S_M_Soft)
            cost_SHoTClean_S_Soft = assist.calc_cost(result_series_SHoTClean_S_M_Soft)
            acc_SHoTClean_S_Soft = assist.calc_acc(result_series_SHoTClean_S_M_Soft)

            total_rms[i][6] += rms_SHoTClean_S_Soft
            total_cost[i][6] += cost_SHoTClean_S_Soft
            total_acc[i][6] += acc_SHoTClean_S_Soft
            total_time[i][6] += end_time - start_time

            print(
                f"BASE S-Multi RMS: {rms_SHoTClean_S_Soft}, Cost: {cost_SHoTClean_S_Soft}, Acc: {acc_SHoTClean_S_Soft}, Time: {end_time - start_time}")

            # BASE P-Single
            dirty_series = assist.read_data(input_file_name, ",")
            dirty_series = assist.add_noise(dirty_series, drate, seed)
            uni_dirty_list = assist.split_by_dimension(dirty_series)
            D = len(uni_dirty_list)  
            uni_result_list = []
            time_cost = 0
            for d in range(D):
                uni_dirty = uni_dirty_list[d]
                SHoTClean_P_Soft = SHoTClean_P1(uni_dirty, s_max, s_min, 10, is_soft=True)
                start_time = time.time()
                result_uni_d = SHoTClean_P_Soft.mainScreen()
                end_time = time.time()
                time_cost += (end_time - start_time)
                uni_result_list.append(result_uni_d)
            result_series_SHoTClean_P_Soft = TimeSeries()
            timestamps_0 = [tp.get_timestamp() for tp in uni_result_list[0].get_timeseries()]
            for idx, ts in enumerate(timestamps_0):
                val_dims = [uni_result_list[d].get_timeseries()[idx].get_value() for d in range(D)]
                truth_dims = [uni_result_list[d].get_timeseries()[idx].get_truth() for d in range(D)]
                noise_dims = [uni_result_list[d].get_timeseries()[idx].get_noise() for d in range(D)]
                result_series_SHoTClean_P_Soft.add_point(TimePoint(ts, val_dims, truth_dims, noise_dims))

            rms_SHoTClean_P_Soft = assist.calc_rms(result_series_SHoTClean_P_Soft)
            cost_SHoTClean_P_Soft = assist.calc_cost(result_series_SHoTClean_P_Soft)
            acc_SHoTClean_P_Soft = assist.calc_acc(result_series_SHoTClean_P_Soft)
            total_rms[i][7] += rms_SHoTClean_P_Soft
            total_cost[i][7] += cost_SHoTClean_P_Soft
            total_acc[i][7] += acc_SHoTClean_P_Soft
            total_time[i][7] += time_cost

            print(f"BASE P-Single RMS: {rms_SHoTClean_P_Soft}, Cost: {cost_SHoTClean_P_Soft}, Acc: {acc_SHoTClean_P_Soft}, Time: {time_cost}")

            # BASE C
            dirty_series = assist.read_data(input_file_name, ",")
            dirty_series = assist.add_noise(dirty_series, drate, seed)
            SHoTClean_C_Soft = SHoTClean_C(dirty_series, s_max, s_min, 10, is_soft=True)
            start_time = time.time()
            result_series_SHoTClean_C_Soft = SHoTClean_C_Soft.mainScreen()
            end_time = time.time()
            rms_SHoTClean_C_Soft = assist.calc_rms(result_series_SHoTClean_C_Soft)
            cost_SHoTClean_C_Soft = assist.calc_cost(result_series_SHoTClean_C_Soft)
            acc_SHoTClean_C_Soft = assist.calc_acc(result_series_SHoTClean_C_Soft)

            total_rms[i][8] += rms_SHoTClean_C_Soft
            total_cost[i][8] += cost_SHoTClean_C_Soft
            total_acc[i][8] += acc_SHoTClean_C_Soft
            total_time[i][8] += end_time - start_time
            print(f"BASE C RMS: {rms_SHoTClean_C_Soft}, Cost: {cost_SHoTClean_C_Soft}, Acc: {acc_SHoTClean_C_Soft}, Time: {end_time - start_time}")

            # Save Data
            # dirty_series = assist.read_data(input_file_name, ",")
            # dirty_series = assist.add_noise(dirty_series, drate, seed)
            # assist.save_data(dirty_series, f"../results/Multi/exchange/Dirty_{drate}.data")
            # assist.save_data(result_series_MTCSC, f"../results/Multi/exchange/MTCSC_{drate}.data")
            # assist.save_data(result_series_screen, f"../results/Multi/exchange/SCREEN_{drate}.data")
            # assist.save_data(result_series_speedacc, f"../results/Multi/exchange/SpeedAcc_{drate}.data")
            # assist.save_data(result_series_expsmooth, f"../results/Multi/exchange/EWMA_{drate}.data")
            # assist.save_data(result_series_lsgreedy, f"../results/Multi/exchange/LsGreedy_{drate}.data")
            # assist.save_data(result_series_SHoTClean_B_Soft, f"../results/Multi/exchange/B-Soft_{drate}.data")
            # assist.save_data(result_series_SHoTClean_C_Soft, f"../results/Multi/exchange/C-Soft_{drate}.data")
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
    name = ["Methods" ,"MTCSC", "SCREEN", "SpeedAcc", "EWMA", "LsGreedy", "B-Soft", "S-Multi-Soft", "P-Single-Soft", "C-Soft"]
    write_file_name = "../results/Multi/exchange/RMS.csv"
    assist.write_csv(write_file_name, name, total_drate, total_rms)
    write_file_name = "../results/Multi/exchange/COST.csv"
    assist.write_csv(write_file_name, name, total_drate, total_cost)
    write_file_name = "../results/Multi/exchange/ACC.csv"
    assist.write_csv(write_file_name, name, total_drate, total_acc)
    write_file_name = "../results/Multi/exchange/TIME.csv"
    assist.write_csv(write_file_name, name, total_drate, total_time)

if __name__ == "__main__":
    main()
