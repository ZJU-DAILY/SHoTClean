import bisect
import math
from collections import deque

import numpy as np

from SHoTClean_P1 import fast_median_mad
from tools.entity import TimeSeries_Single, TimePoint, TimePoint_Single


class SHoTClean_P1_example:
    def __init__(self, timeseries, sMax, sMin, t, alpha=0.05, is_soft=True):
        self.timeseries = timeseries
        self.kp = None
        self.T = t
        self.SMAX = sMax
        self.SMIN = sMin
        self.is_soft = is_soft
        self.alpha = alpha
        self.window_stats = {}
        self.cnt = 0

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
        preVal  = prePoint.get_value()
        kpTime  = self.kp.get_timestamp()
        kpVal   = self.kp.get_value()

        lowerBound = preVal + self.SMIN * (kpTime - preTime)
        upperBound = preVal + self.SMAX * (kpTime - preTime)
        length = len(window_deque)

        if length == 1:
            if (lowerBound > kpVal) or (upperBound < kpVal):
                self.kp.set_value(preVal)
            return

        # 1) 提取 times 与 values
        times  = [p.get_timestamp() for p in window_deque]
        values = [p.get_value()    for p in window_deque]
        rel_times = [times[i] - preTime for i in range(length)]

        # 2) 计算 A_list, B_list, C_list
        A_list = [ values[i] - self.SMIN * times[i] for i in range(length) ]
        B_list = [ values[i] - self.SMAX * times[i] for i in range(length) ]

        if self.is_soft:
            soft_scores = [ self._compute_soft_score(values[i], window_deque) for i in range(length) ]
            C_list = [soft_scores[i] * math.exp(-0.01 * rel_times[i]) for i in range(length) ]
        else:
            C_list = [1.0] * length

        print("----- new point coming -----")
        print(f"preTime = {preTime},  kpTime = {kpTime}")
        print(f"soft_scores = {soft_scores}")
        print(f"A_list = {A_list}")
        print(f"B_list = {B_list}")
        print(f"C_list = {C_list}")

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
        print(f"B_hat = {B_hat}")

        dp = [0.0] * length
        prev = [-1] * length
        for i in range(length):
            dt = times[i] - preTime
            if dt > 0:
                dv = preVal - values[i]
                if (dv >= self.SMIN * dt) and (dv <= self.SMAX * dt):
                    dp[i] = C_list[i]
        print(f"dp = {dp}")

        pts = []
        for i in range(length):
            pts.append({
                'i': i,
                'A': A_list[i],
                'B_hat': B_hat[i],
                'C': C_list[i],
                't': rel_times[i]
            })

        fenw = SHoTClean_P1_example._Fenwick1D(N_B)

        def cdq_solve(l, r):
            if kpTime == 2:
                print(f"cdq_solve({l}, {r}) called with pts = {pts[l:r+1]}")
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

        if kpTime == 2:
            print(f"Best end index: {best_end}, path: {path}")

        max_idx = path[-1]
        maxPoint = window_deque[max_idx]
        maxTime  = maxPoint.get_timestamp()
        maxVal   = maxPoint.get_value()

        print(f"maxtime = {maxTime}")
        print("--------------------------------------\n")

        lowerBound_max = maxVal + self.SMAX * (kpTime - maxTime)
        upperBound_max = maxVal + self.SMIN * (kpTime - maxTime)

        lowerBound = max(lowerBound, lowerBound_max)
        upperBound = min(upperBound, upperBound_max)

        if (upperBound < kpVal) or (lowerBound > kpVal):
            print(A_list, B_list, C_list, rel_times)
            print(f"kpTime: {kpTime}, preTime: {preTime}, maxTime: {maxTime}, kpVal: {kpVal}, preVal: {preVal}, maxVal: {maxVal}")

            pre_dis = kpTime - preTime
            pre_next_dis = maxTime - preTime
            rate = pre_dis / pre_next_dis
            modify = (maxVal - preVal) * rate + preVal
            self.kp.set_value(modify)


if __name__ == "__main__":

    point_list = [
        TimePoint_Single(1, 2),
        TimePoint_Single(2, 5),
        TimePoint_Single(3, 3),
        TimePoint_Single(4, 2),
        TimePoint_Single(5, 3),
        TimePoint_Single(6, 1),
        TimePoint_Single(7, 4),
    ]
    timeseries = TimeSeries_Single()
    for tp in point_list:
        timeseries.add_point(tp)
    sMax = 2.0000
    sMin = -2.0000
    t = 6
    example = SHoTClean_P1_example(timeseries, sMax, sMin, t, alpha=0.05, is_soft=True)
    result = example.mainScreen()
    print("Corrected Time Series:")
    for point in result.get_timeseries():
        print(f"Timestamp: {point.get_timestamp()}, Value: {point.get_value()}")


