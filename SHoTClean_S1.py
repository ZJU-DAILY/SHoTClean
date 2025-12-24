import time
from collections import defaultdict, deque

from tools.utils import Assist_Single
from tools.entity import TimePoint_Single, TimeSeries_Single

import numpy as np
from numba import njit


@njit
def fast_median_mad(values):
    values = values[:-1]
    values = np.sort(values)
    n = len(values)

    if n % 2 == 1:
        median = values[n // 2]
    else:
        median = (values[(n // 2) - 1] + values[n // 2]) / 2.0

    deviations = np.abs(values - median)
    deviations = np.sort(deviations)

    if len(deviations) % 2 == 1:
        mad = deviations[len(deviations) // 2]
    else:
        mad = (deviations[(len(deviations) // 2) - 1] +
               deviations[len(deviations) // 2]) / 2.0

    mad_std = 1.4826 * mad + 1e-6
    return median, mad_std


"""Streaming Algorithm only for Single-dimensional Data"""
class SHoTClean_S1:
    def __init__(self, timeseries, sMax, sMin, t, alpha=0.01, is_soft=True):
        self.timeseries = timeseries
        self.kp = None
        self.T = t
        self.SMAX = sMax
        self.SMIN = sMin
        self.is_soft = is_soft
        self.alpha = alpha
        self.window_stats = {}

    def mainScreen(self):
        totalList = self.timeseries.get_timeseries()
        size = len(totalList)
        preEnd = -1
        prePoint = None
        tempSeries = TimeSeries_Single()
        readIndex = 1
        tp = totalList[0]
        tempSeries.add_point(tp)

        window_deque = deque()
        window_deque.append(tp)

        wStartTime = tp.get_timestamp()
        wEndTime = wStartTime
        wGoalTime = wStartTime + self.T

        while readIndex < size:
            tp = totalList[readIndex]
            curTime = tp.get_timestamp()

            if curTime > wGoalTime:
                while True:
                    if len(window_deque) == 0:
                        window_deque.append(tp)
                        wGoalTime = curTime + self.T
                        wEndTime = curTime
                        break

                    self.kp = window_deque[0]
                    wStartTime = self.kp.get_timestamp()
                    wGoalTime = wStartTime + self.T

                    if curTime <= wGoalTime:
                        window_deque.append(tp)
                        wEndTime = curTime
                        break

                    curEnd = wEndTime
                    if preEnd == -1:
                        prePoint = self.kp
                    if self.is_soft:
                        self._update_window_stats_deque(window_deque)
                    self.local_deque(window_deque, prePoint)
                    prePoint = self.kp
                    preEnd = curEnd
                    window_deque.popleft()
            else:
                if curTime > wEndTime:
                    window_deque.append(tp)
                    wEndTime = curTime
            readIndex += 1

        while len(window_deque) > 0:
            self.kp = window_deque[0]
            if prePoint is None:
                prePoint = self.kp
            if self.is_soft:
                self._update_window_stats_deque(window_deque)
            self.local_deque(window_deque, prePoint)
            prePoint = self.kp
            window_deque.popleft()

        return self.timeseries

    def _update_window_stats_deque(self, window_deque):
        values = np.array([p.get_value() for p in window_deque])
        window_id = id(window_deque)

        if len(values) < self.T:
            self.window_stats[window_id] = {'mu': 0, 'std': 1e-6}
            return
        mu, std = fast_median_mad(values)

        self.window_stats[window_id] = {
            'mu': mu,
            'std': std,
        }

    def _get_window_stats_deque(self, window_deque):
        if len(window_deque) == 0:
            return 0, 1e-6
        return self.window_stats.get(id(window_deque), {'mu': 0, 'std': 1e-6}).values()

    def _compute_soft_score(self, value, window_deque):
        mu, std = self._get_window_stats_deque(window_deque)
        deviation = abs(value - mu) / std
        return np.exp(-self.alpha * deviation)

    def distance(self, prePoint, kp):
        return kp.get_value() - prePoint.get_value()

    def _temporal_decay(self, gap):
        return np.exp(-0.01 * gap)

    def local_deque(self, window_deque, prePoint):
        preTime = prePoint.get_timestamp()
        preVal = prePoint.get_value()
        kpTime = self.kp.get_timestamp()
        kpVal = self.kp.get_value()

        lowerBound = preVal + self.SMIN * (kpTime - preTime)
        upperBound = preVal + self.SMAX * (kpTime - preTime)
        length = len(window_deque)

        if length == 1 and (lowerBound > kpVal or upperBound < kpVal):
            self.kp.set_value(preVal)
            return

        times = [p.get_timestamp() for p in window_deque]
        values = [p.get_value() for p in window_deque]
        dp = [{'length': 0, 'score': -np.inf, 'prev': -1} for _ in range(length)]
        soft_scores = [self._compute_soft_score(values[i], window_deque) for i in range(length)] if self.is_soft else None

        for i in range(length):
            t1 = times[i]
            v1 = values[i]

            if ((t1 - preTime) * self.SMAX) >= (preVal - v1) >= ((t1 - preTime) * self.SMIN):
                dp[i]['length'] = 1
                dp[i]['prev'] = -1
                if self.is_soft:
                    dp[i]['score'] = soft_scores[i]

            for j in range(i):
                if dp[j]['prev'] == -1:
                    t2 = times[j]
                    v2 = values[j]

                    if ((t1 - t2) * self.SMAX) >= (v1 - v2) >= ((t1 - t2) * self.SMIN):
                        if not self.is_soft:
                            candidate_length = dp[j]['length'] + 1
                            if candidate_length > dp[i]['length']:
                                dp[i]['length'] = candidate_length
                                dp[i]['prev'] = j
                        else:
                            candidate_score = dp[j]['score'] + soft_scores[i] * self._temporal_decay(t1 - t2)
                            if candidate_score > dp[i]['score']:
                                dp[i]['score'] = candidate_score
                                dp[i]['prev'] = j

        best_end = 0
        if self.is_soft:
            for i in range(1, length):
                if dp[i]['score'] > dp[best_end]['score']:
                    best_end = i
        else:
            for i in range(1, length):
                if dp[i]['length'] > dp[best_end]['length']:
                    best_end = i

        best_start = best_end
        while dp[best_start]['prev'] != -1:
            best_start = dp[best_start]['prev']

        maxPoint = window_deque[best_start]
        maxTime = maxPoint.get_timestamp()
        maxVal = maxPoint.get_value()

        lowerBound_max = maxVal + self.SMAX * (kpTime - maxTime)
        upperBound_max = maxVal + self.SMIN * (kpTime - maxTime)

        lowerBound = max(lowerBound, lowerBound_max)
        upperBound = min(upperBound, upperBound_max)

        if upperBound < kpVal or lowerBound > kpVal:
            pre_dis = kpTime - preTime
            pre_next_dis = maxTime - preTime
            rate = pre_dis / pre_next_dis
            modify = (maxVal - preVal) * rate + preVal
            self.kp.set_value(modify)


if __name__ == "__main__":
    assist = Assist_Single()
    input_file_name = "stock/stock12k.data"
    # input_file_name = "CA/CA.csv"
    # input_file_name = "exchange/exchange.data"
    if 'stock' in input_file_name:
        s_max = 3
        s_min = -3
    elif 'CA' in input_file_name:
        s_max = 3000
        s_min = -3000
    elif 'exchange' in input_file_name:
        s_max = 0.1
        s_min = -0.1
    t = 10

    method_num = 2
    total_drate = np.zeros(10)
    total_rms = np.zeros((10, method_num))
    total_cost = np.zeros((10, method_num))
    total_cossim = np.zeros((10, method_num))
    total_time = np.zeros((10, method_num))

    for i in range(10):
        drate = round(0.05 + 0.025 * i, 3)
        # drate = 0.20
        total_drate[i] = drate
        print(f"Dirty rate is {drate}")
        total_dirty_rms = 0
        exp_time = 10

        for j in range(exp_time):
            seed = j + 1

            dirty_series = assist.read_data(input_file_name, ",")
            dirty_series = assist.add_noise(dirty_series, drate, seed)
            rmsDirty = assist.calc_rms(dirty_series)
            total_dirty_rms += rmsDirty
            SHoTClean_Soft = SHoTClean_S1(dirty_series, s_max, s_min, t, is_soft=True)
            start_time = time.time()
            result_series_SHoTClean_Soft = SHoTClean_Soft.mainScreen()
            end_time = time.time()
            rms_SHoTClean_Soft = assist.calc_rms(result_series_SHoTClean_Soft)
            cost_SHoTClean_Soft = assist.calc_cost(result_series_SHoTClean_Soft)
            cossim_SHoTClean_Soft = assist.calc_acc(result_series_SHoTClean_Soft)

            total_rms[i][1] += rms_SHoTClean_Soft
            total_cost[i][1] += cost_SHoTClean_Soft
            total_cossim[i][1] += cossim_SHoTClean_Soft
            total_time[i][1] += (end_time - start_time)

        total_dirty_rms /= exp_time
        print(f"Dirty RMS error is {round(total_dirty_rms, 3)}")

        # Output results
        for j in range(method_num):
            total_rms[i][j] /= exp_time
            total_cost[i][j] /= exp_time
            total_cossim[i][j] /= exp_time
            total_time[i][j] /= exp_time

    print(total_rms)
    print(total_cost)
    print(total_cossim)
    print(total_time)
    # name = ["Methods", "HARD", "SOFT"]
    # write_file_name = "results/One/test/RMS.csv"
    # assist.write_csv(write_file_name, name, total_drate, total_rms)
    # write_file_name = "results/One/test/COST.csv"
    # assist.write_csv(write_file_name, name, total_drate, total_cost)
    # write_file_name = "results/One/test/NUM.csv"
    # assist.write_csv(write_file_name, name, total_drate, total_num)
    # write_file_name = "results/One/test/TIME.csv"
    # assist.write_csv(write_file_name, name, total_drate, total_time)