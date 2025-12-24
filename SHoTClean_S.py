import time
from collections import defaultdict, deque

from tools.utils import Assist
from tools.entity import TimePoint, TimeSeries

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


"""Streaming Algorithm for Multi(Single)-dimensional Data"""
class SHoTClean_S:
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
        tempSeries = TimeSeries()
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
                    wStartTime  = self.kp.get_timestamp()
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
        diff = value - mu
        std_scalar = np.linalg.norm(std) if np.any(std != 0) else 1e-6
        deviation = np.linalg.norm(diff) / std_scalar
        return np.exp(-self.alpha * (deviation))

    def distance(self, prePoint, kp):
        return kp.get_value() - prePoint.get_value()

    def _temporal_decay(self, gap):
        return np.exp(-0.01 * gap)

    def _speed_ok(self, v_from, v_to, dt):
        delta = np.array(v_to) - np.array(v_from)
        return np.all(delta >= self.SMIN * dt) and np.all(delta <= self.SMAX * dt)

    def _judge_repair(self, preVal, maxVal, kpVal, preTime, maxTime, kpTime):
        return not (
            self._speed_ok(preVal, kpVal, kpTime - preTime) and
            self._speed_ok(maxVal, kpVal, maxTime - kpTime)
        )


    def local_deque(self, window_deque, prePoint):
        preTime = prePoint.get_timestamp()
        preVal = prePoint.get_value()
        kpTime = self.kp.get_timestamp()
        kpVal = self.kp.get_value()

        if len(window_deque) == 1:
            if self._judge_repair(preVal, kpVal, kpVal, preTime, kpTime, kpTime):
                self.kp.set_value(preVal)
            return

        times = [p.get_timestamp() for p in window_deque]
        values = [p.get_value() for p in window_deque]
        length = len(window_deque)

        dp = [{'length': 0, 'score': -np.inf, 'prev': -1} for _ in range(length)]
        soft_scores = None
        if self.is_soft:
            soft_scores = [self._compute_soft_score(values[i], window_deque) for i in range(length)]

        for i in range(length):
            t1 = times[i]
            v1 = values[i]
            if self._speed_ok(preVal, v1, t1 - preTime):
                dp[i]['length'] = 1
                dp[i]['prev'] = -1
                if self.is_soft:
                    dp[i]['score'] = soft_scores[i]

            for j in range(i):
                if dp[j]['prev'] == -1:
                    t2 = times[j]
                    v2 = values[j]
                    if self._speed_ok(v2, v1, t1 - t2):
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

        maxPoint = window_deque[best_end]
        maxTime = maxPoint.get_timestamp()
        maxVal = maxPoint.get_value()

        if self._judge_repair(preVal, maxVal, kpVal, preTime, maxTime, kpTime):
            rate = (kpTime - preTime) / (maxTime - preTime)
            lb = preVal + self.SMIN * (kpTime - preTime)
            ub = preVal + self.SMAX * (kpTime - preTime)
            violation = (kpVal < lb) | (kpVal > ub)
            modify = kpVal.copy()
            modify[violation] = preVal[violation] + rate * (maxVal[violation] - preVal[violation])
            self.kp.set_value(modify)

if __name__ == "__main__":
    assist = Assist()
    # input_file_name = "stock/stock12k.data"
    # input_file_name = "exchange/exchange.data"
    input_file_name = "SWaT/SWaT.data"
    if 'stock' in input_file_name:
        s_max = 3
        s_min = -3
    elif 'exchange' in input_file_name:
        s_max = 0.1
        s_min = -0.1
        # pass
    elif 'SWaT' in input_file_name:
        s_max = 13
        s_min = -13
    else:
        raise ValueError("Unsupported input file type")
    t = 10

    method_num = 2
    total_drate = np.zeros(10)
    total_rms = np.zeros((10, method_num))
    total_cost = np.zeros((10, method_num))
    total_acc = np.zeros((10, method_num))
    total_time = np.zeros((10, method_num))

    for i in range(10):
        drate = round(0.05 + 0.025 * i, 3)
        # drate = 0.2
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
            SHoTClean_Soft = SHoTClean_S(dirty_series, s_max, s_min, t, is_soft=True)
            start_time = time.time()
            result_series_SHoTClean_Soft = SHoTClean_Soft.mainScreen()
            end_time = time.time()
            rms_SHoTClean_Soft = assist.calc_rms(result_series_SHoTClean_Soft)
            cost_SHoTClean_Soft = assist.calc_cost(result_series_SHoTClean_Soft)
            acc_SHoTClean_Soft = assist.calc_acc(result_series_SHoTClean_Soft)

            total_rms[i][1] += rms_SHoTClean_Soft
            total_cost[i][1] += cost_SHoTClean_Soft
            total_acc[i][1] += acc_SHoTClean_Soft
            total_time[i][1] += (end_time - start_time)

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
    # name = ["Methods", "HARD", "SOFT"]
    # write_file_name = "results/One/test/RMS.csv"
    # assist.write_csv(write_file_name, name, total_drate, total_rms)
    # write_file_name = "results/One/test/COST.csv"
    # assist.write_csv(write_file_name, name, total_drate, total_cost)
    # write_file_name = "results/One/test/NUM.csv"
    # assist.write_csv(write_file_name, name, total_drate, total_acc)
    # write_file_name = "results/One/test/TIME.csv"
    # assist.write_csv(write_file_name, name, total_drate, total_time)