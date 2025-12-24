import bisect
import math
import time
from collections import defaultdict, deque

from tools.utils import Assist_Single
from tools.entity import TimePoint_Single, TimeSeries_Single

import numpy as np
from numba import njit

@njit(cache=True, fastmath=True)
def fast_median_mad(values):
    values = values[:-1]
    n = len(values)
    if n == 0:
        return 0.0, 1e-6

    if n % 2 == 1:
        k = n // 2
        partitioned = np.partition(values, k)
        median = partitioned[k]
    else:
        k1 = (n // 2) - 1
        k2 = n // 2
        partitioned = np.partition(values, [k1, k2])
        median = (partitioned[k1] + partitioned[k2]) / 2.0

    deviations = np.abs(values - median)
    m = len(deviations)
    if m == 0:
        mad = 0.0
    else:
        if m % 2 == 1:
            k_m = m // 2
            partitioned_dev = np.partition(deviations, k_m)
            mad = partitioned_dev[k_m]
        else:
            k_m1 = (m // 2) - 1
            k_m2 = m // 2
            partitioned_dev = np.partition(deviations, [k_m1, k_m2])
            mad = (partitioned_dev[k_m1] + partitioned_dev[k_m2]) / 2.0

    mad_std = 1.4826 * mad + 1e-6
    return median, mad_std


"""CDQ Partitioning Algorithm only for Single-dimensional Data"""
class SHoTClean_P1:
    def __init__(self, timeseries, sMax, sMin, t, alpha=0.05, is_soft=True):
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
        mu, std = fast_median_mad(values)
        self.window_stats[window_id] = {'mu': mu, 'std': std}

    def _get_window_stats_deque(self, window_deque):
        if len(window_deque) == 0:
            return 0, 1e-6
        return self.window_stats.get(id(window_deque), {'mu': 0, 'std': 1e-6}).values()

    def _compute_soft_score(self, value, window_deque):
        mu, std = self._get_window_stats_deque(window_deque)
        deviation = abs(value - mu) / std
        return math.exp(- (self.alpha * (deviation)))

    def distance(self, prePoint, kp):
        return kp.get_value() - prePoint.get_value()

    def _temporal_decay(self, gap):
        return math.exp(-0.01 * gap)

    class _Fenwick1D:
        def __init__(self, n):
            self.n = n
            self.tree = [(0.0, -1)] * (n + 1)

        def update(self, i, val, j_idx):
            while i <= self.n:
                if val > self.tree[i][0]:
                    self.tree[i] = (val, j_idx)
                i += i & -i

        def query(self, i):
            res_val = 0.0
            res_j = -1
            while i > 0:
                if self.tree[i][0] > res_val:
                    res_val = self.tree[i][0]
                    res_j = self.tree[i][1]
                i -= i & -i
            return res_val, res_j

        def clear(self):
            for idx in range(1, self.n + 1):
                self.tree[idx] = (0.0, -1)

    def local_deque(self, window_deque, prePoint):
        preTime = prePoint.get_timestamp()
        preVal = prePoint.get_value()
        kpTime = self.kp.get_timestamp()
        kpVal = self.kp.get_value()

        lowerBound = preVal + self.SMIN * (kpTime - preTime)
        upperBound = preVal + self.SMAX * (kpTime - preTime)
        length = len(window_deque)

        if length == 1:
            if (lowerBound > kpVal) or (upperBound < kpVal):
                self.kp.set_value(preVal)
            return

        times  = [p.get_timestamp() for p in window_deque]
        values = [p.get_value()    for p in window_deque]
        rel_times = [times[i] - preTime for i in range(length)]

        A_list = [ values[i] - self.SMIN * times[i] for i in range(length) ]
        B_list = [ values[i] - self.SMAX * times[i] for i in range(length) ]

        if self.is_soft:
            soft_scores = [ self._compute_soft_score(values[i], window_deque) for i in range(length) ]
            C_list = [soft_scores[i] * math.exp(-0.01 * rel_times[i]) for i in range(length) ]
        else:
            C_list = [1.0] * length

        B_tuples = []
        for i in range(length):
            x = B_list[i]
            arr = np.atleast_1d(x)
            if arr.shape == ():
                B_tuples.append((float(arr),))
            else:
                B_tuples.append(tuple(arr.tolist()))

        B_sorted = sorted(set(B_tuples))
        N_B = len(B_sorted)

        def getBidx(x):
            arr = np.atleast_1d(x)
            if arr.shape == ():
                xt = (float(arr),)
            else:
                xt = tuple(arr.tolist())
            return bisect.bisect_left(B_sorted, xt) + 1

        B_hat = [N_B - getBidx(B_list[i]) + 1 for i in range(length)]

        dp = [0.0] * length
        prev = [-1] * length
        for i in range(length):
            dt = times[i] - preTime
            if dt > 0:
                dv = preVal - values[i]
                if (dv >= self.SMIN * dt) and (dv <= self.SMAX * dt):
                    dp[i] = C_list[i]

        pts = []
        for i in range(length):
            pts.append({
                'i': i,
                'A': A_list[i],
                'B_hat': B_hat[i],
                'C': C_list[i],
                't': rel_times[i]
            })

        fenw = SHoTClean_P1._Fenwick1D(N_B)

        def cdq_solve(l, r):
            if l == r:
                return
            m = (l + r) // 2

            cdq_solve(l, m)

            left_part  = sorted(pts[l:m+1],    key=lambda P: (P['A'], P['i']))
            right_part = sorted(pts[m+1:r+1],  key=lambda P: (P['A'], P['i']))

            iL = 0
            for PR in right_part:
                while iL < len(left_part) and left_part[iL]['A'] <= PR['A']:
                    j_idx = left_part[iL]['i']
                    if self.is_soft:
                        val = dp[j_idx] * math.exp(0.01 * left_part[iL]['t'])
                    else:
                        val = dp[j_idx]
                    fenw.update(left_part[iL]['B_hat'], val, j_idx)
                    iL += 1

                best_val, arg_j = fenw.query(PR['B_hat'])
                if best_val > 0.0:
                    i_idx = PR['i']
                    if self.is_soft:
                        cand = best_val * math.exp(-0.01 * PR['t']) + PR['C']
                    else:
                        cand = best_val + 1.0

                    if cand > dp[i_idx]:
                        dp[i_idx] = cand
                        prev[i_idx] = arg_j

            fenw.clear()

            cdq_solve(m+1, r)

        cdq_solve(0, length-1)

        best_end = max(range(length), key=lambda k: dp[k])
        path = []
        cur = best_end
        while cur != -1:
            path.append(cur)
            cur = prev[cur]

        max_idx = path[-1]
        maxPoint = window_deque[max_idx]
        maxTime  = maxPoint.get_timestamp()
        maxVal   = maxPoint.get_value()

        lowerBound_max = maxVal + self.SMAX * (kpTime - maxTime)
        upperBound_max = maxVal + self.SMIN * (kpTime - maxTime)

        lowerBound = max(lowerBound, lowerBound_max)
        upperBound = min(upperBound, upperBound_max)

        if (upperBound < kpVal) or (lowerBound > kpVal):
            pre_dis = kpTime - preTime
            pre_next_dis = maxTime - preTime
            rate = pre_dis / pre_next_dis
            modify = (maxVal - preVal) * rate + preVal
            self.kp.set_value(modify)

if __name__ == "__main__":
    assist = Assist_Single()
    input_file_name = "stock/stock12k.data"
    # input_file_name = "CA/CA.csv"
    if 'stock' in input_file_name:
        s_max = 3
        s_min = -3
    else:
        raise ValueError("Please set s_max and s_min for your dataset.")
    t = 10

    method_num = 2
    total_drate = np.zeros(10)
    total_rms = np.zeros((10, method_num))
    total_cost = np.zeros((10, method_num))
    total_num = np.zeros((10, method_num))
    total_time = np.zeros((10, method_num))

    for i in range(10):
        drate = round(0.05 + 0.025 * i, 3)
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
            SHoTClean_Soft = SHoTClean_P1(dirty_series, s_max, s_min, t, is_soft=True)
            start_time = time.time()
            result_series_SHoTClean_Soft = SHoTClean_Soft.mainScreen()
            end_time = time.time()
            rms_SHoTClean_Soft = assist.calc_rms(result_series_SHoTClean_Soft)
            cost_SHoTClean_Soft = assist.calc_cost(result_series_SHoTClean_Soft)
            num_SHoTClean_Soft = assist.point_num1(result_series_SHoTClean_Soft)

            total_rms[i][1] += rms_SHoTClean_Soft
            total_cost[i][1] += cost_SHoTClean_Soft
            total_num[i][1] += num_SHoTClean_Soft
            total_time[i][1] += (end_time - start_time)


        total_dirty_rms /= exp_time
        print(f"Dirty RMS error is {round(total_dirty_rms, 3)}")

        # Output results
        for j in range(method_num):
            total_rms[i][j] /= exp_time
            total_cost[i][j] /= exp_time
            total_num[i][j] /= exp_time
            total_time[i][j] /= exp_time

    print(total_rms)
    print(total_cost)
    print(total_num)
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