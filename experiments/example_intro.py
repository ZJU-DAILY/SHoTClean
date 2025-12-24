import numpy as np
import time

from SHoTClean_B1 import SHoTClean_B1
from baselines.SCREEN import SCREEN
from baselines.SpeedAcc import SpeedAcc
from tools.utils import Assist_Single
from baselines.MTCSC import MTCSC
from baselines.LsGreedy import LsGreedy
from baselines.EWMA import EWMA
from matplotlib import pyplot as plt
import matplotlib as mpl
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif']  = ['Times New Roman']


def main():
    assist = Assist_Single()

    input_file_name = "CA/CA.csv"
    s_max = 3500
    s_min = -3500
    TT = 10
    T = 10

    method_num = 11
    total_drate = np.zeros(10)
    total_rms = np.zeros((10, method_num))
    total_cost = np.zeros((10, method_num))
    total_num = np.zeros((10, method_num))
    total_time = np.zeros((10, method_num))

    for i in range(1):
        drate = round(0.05 + 0.025 * i, 3)
        total_drate[i] = drate
        print(f"Dirty rate is {drate}")
        total_dirty_rms = 0
        exp_time = 1

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
            assist.save_data(result_series_MTCSC, f"../results/One/CA/CA_MTCSC.data")
            end_time = time.time()
            rms_MTCSC = assist.calc_rms(result_series_MTCSC)
            cost_MTCSC = assist.calc_cost(result_series_MTCSC)
            num_MTCSC = assist.point_num1(result_series_MTCSC)

            total_rms[i][0] += rms_MTCSC
            total_cost[i][0] += cost_MTCSC
            total_num[i][0] += num_MTCSC
            total_time[i][0] += (end_time - start_time)

            # SCREEN
            dirty_series = assist.read_data(input_file_name, ",")
            dirty_series = assist.add_noise(dirty_series, drate, seed)
            screen = SCREEN(dirty_series, s_max, s_min, TT)
            start_time = time.time()
            result_series_screen = screen.main_screen()
            assist.save_data(result_series_screen, f"../results/One/CA/CA_SCREEN.data")
            end_time = time.time()
            rms_screen = assist.calc_rms(result_series_screen)
            cost_screen = assist.calc_cost(result_series_screen)
            num_screen = assist.point_num1(result_series_screen)

            total_rms[i][1] += rms_screen
            total_cost[i][1] += cost_screen
            total_num[i][1] += num_screen
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
            num_speedacc = assist.point_num1(result_series_speedacc)

            total_rms[i][2] += rms_speedacc
            total_cost[i][2] += cost_speedacc
            total_num[i][2] += num_speedacc
            total_time[i][2] += (end_time - start_time)

            # EWMA
            dirty_series = assist.read_data(input_file_name, ",")
            dirty_series = assist.add_noise(dirty_series, drate, seed)
            expsmooth = EWMA(dirty_series, 0.042)
            start_time = time.time()
            result_series_expsmooth = expsmooth.main_exp()
            assist.save_data(result_series_expsmooth, f"../results/One/CA/CA_EWMA.data")
            end_time = time.time()
            rms_expsmooth = assist.calc_rms(result_series_expsmooth)
            cost_expsmooth = assist.calc_cost(result_series_expsmooth)
            num_expsmooth = assist.point_num1(result_series_expsmooth)

            total_rms[i][3] += rms_expsmooth
            total_cost[i][3] += cost_expsmooth
            total_num[i][3] += num_expsmooth
            total_time[i][3] += (end_time - start_time)

            # LsGreedy
            dirty_series = assist.read_data(input_file_name, ",")
            dirty_series = assist.add_noise(dirty_series, drate, seed)
            lsgreedy = LsGreedy(dirty_series)
            start_time = time.time()
            result_series_lsgreedy = lsgreedy.repair()
            assist.save_data(result_series_lsgreedy, f"../results/One/CA/CA_LsGreedy.data")
            end_time = time.time()
            rms_lsgreedy = assist.calc_rms(result_series_lsgreedy)
            cost_lsgreedy = assist.calc_cost(result_series_lsgreedy)
            num_lsgreedy = assist.point_num1(result_series_lsgreedy)

            total_rms[i][4] += rms_lsgreedy
            total_cost[i][4] += cost_lsgreedy
            total_num[i][4] += num_lsgreedy
            total_time[i][4] += (end_time - start_time)

            # BASE G WITH SOFT
            dirty_series = assist.read_data(input_file_name, ",")
            dirty_series = assist.add_noise(dirty_series, drate, seed)
            SHoTClean_Soft = SHoTClean_B1(dirty_series, s_max, s_min, is_soft=True)
            start_time = time.time()
            result_series_SHoTClean_Soft = SHoTClean_Soft.clean()
            assist.save_data(result_series_SHoTClean_Soft, f"../results/One/CA/CA_SHoTCleanB.data")
            end_time = time.time()
            rms_SHoTClean_Soft = assist.calc_rms(result_series_SHoTClean_Soft)
            cost_SHoTClean_Soft = assist.calc_cost(result_series_SHoTClean_Soft)
            num_SHoTClean_Soft = assist.point_num1(result_series_SHoTClean_Soft)

            total_rms[i][6] += rms_SHoTClean_Soft
            total_cost[i][6] += cost_SHoTClean_Soft
            total_num[i][6] += num_SHoTClean_Soft
            total_time[i][6] += (end_time - start_time)

            dirty_series = assist.read_data(input_file_name, ",")
            dirty_series = assist.add_noise(dirty_series, drate, seed)
            assist.save_data(dirty_series, f"../results/One/CA/CA_Dirty.data")
            # exit()

            dirty = [p.get_value() for p in dirty_series.get_timeseries()]
            truth = [p.get_truth() for p in dirty_series.get_timeseries()]
            SHoT = [p.get_value() for p in result_series_SHoTClean_Soft.get_timeseries()]
            mtcsc = [p.get_value() for p in result_series_MTCSC.get_timeseries()]
            lsgreedy = [p.get_value() for p in result_series_lsgreedy.get_timeseries()]
            expsmooth = [p.get_value() for p in result_series_expsmooth.get_timeseries()]
            screen = [p.get_value() for p in result_series_screen.get_timeseries()]
            Akane_read = assist.read_data("../results/One/CA/CA_Akane.data", ",")
            Akane = [p.get_value() for p in Akane_read.get_timeseries()]

            # window = 50
            # found = False
            # cnt = 0
            # start = 1000
            # while start < len(truth) - window + 1 and cnt < 3:
            #     end = start + window
            #     new_start = start - 10
            #     new_end = end + 10
            #     cond_shot = np.all(SHoT[new_start:new_end] == truth[new_start:new_end])
            #     cond_mtcsc = np.all(mtcsc[new_start:new_end] != truth[new_start:new_end])
            #     cond_akane = np.all(Akane[new_start:new_end] != truth[new_start:new_end])
            #     if cond_shot and cond_mtcsc and cond_akane:
            #         print(f"找到满足条件的区间：[{new_start}, {new_end})")
            #         cnt += 1
            #         start += window
            #     start += 1

            # Detailed One
            # plt.figure(figsize=(12, 6))
            # start = 18680
            # end = 18701
            # x = range(start, end)
            # plt.plot(x, truth[start:end], color='black', marker='o', lw=5, markersize=15, label='Truth')
            # plt.plot(x, dirty[start:end], color='green', marker='s', lw=5, markersize=15, label='Dirty')
            # plt.plot(x, expsmooth[start:end], color='magenta', marker='^', lw=5, markersize=15, label='EWMA')
            # plt.plot(x, screen[start:end], color='gold', marker='d', lw=5, markersize=15, label='SCREEN')
            # plt.plot(x, mtcsc[start:end], color='blue', marker='*', lw=5, markersize=15, label='MTCSC')
            # plt.plot(x, lsgreedy[start:end], color='cyan', marker='p', lw=5, markersize=15, label='LsGreedy')
            # plt.plot(x, Akane[start:end], color='orange', marker='h', lw=5, markersize=15, label='Akane')
            # plt.plot(x, SHoT[start:end], color='red', marker='+', lw=5, markersize=15, label='SHoTClean')
            #
            # plt.xticks(ticks=[])
            # plt.yticks(ticks=[])
            # ax = plt.gca()
            # for spine in ax.spines.values():
            #     spine.set_visible(False)
            #
            # plt.tight_layout()
            # # plt.show()
            # plt.savefig(f"../results/Example/intro_{start}_{end}.svg")
            # exit()

            # Large One
            plt.figure(figsize=(12, 6))
            start = 18615
            end = 18710
            x = range(start, end)
            plt.plot(x, truth[start:end], color='black', marker='o', markersize=5, label='Truth')
            plt.plot(x, dirty[start:end], color='green', marker='s', markersize=5, label='Dirty')
            plt.plot(x, expsmooth[start:end], color='magenta', marker='^', markersize=5, label='EWMA')
            plt.plot(x, screen[start:end], color='gold', marker='d', markersize=5, label='SCREEN')
            plt.plot(x, mtcsc[start:end], color='blue', marker='*', markersize=6, label='MTCSC')
            plt.plot(x, lsgreedy[start:end], color='cyan', marker='p', markersize=5, label='LsGreedy')
            plt.plot(x, Akane[start:end], color='orange', marker='h', markersize=5, label='Akane')
            plt.plot(x, SHoT[start:end], color='red', marker='+', markersize=6, label='SHoTClean')

            plt.xlabel('Timestamp', fontsize=20)  # X 轴标签
            plt.ylabel('MWh', fontsize=20)  # Y 轴标签
            plt.xticks(ticks=range(start, end, 10), fontsize=16)  # X 轴刻度
            plt.yticks(fontsize=16)  # Y 轴刻度

            plt.legend(loc='upper left', fontsize=12)
            plt.tight_layout()
            # plt.show()
            plt.savefig(f"../results/Example/intro_{start}_{end}.svg")

if __name__ == "__main__":
    main()
