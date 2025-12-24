import time
from collections import defaultdict, deque

from tigramite.data_processing import DataFrame as TgmDataFrame
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.pcmci import PCMCI
from sklearn.linear_model import LinearRegression

from tools.utils import Assist
from tools.entity import TimePoint, TimeSeries

import numpy as np
from numba import njit
import warnings
warnings.filterwarnings("ignore")


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


"""Causal Algorithm for Multi-dimensional Data"""
class SHoTClean_C:
    def __init__(self, timeseries, sMax, sMin, t, alpha=0.01, is_soft=True, w_speed=0.5, w_causal=0.5, p_max=2):
        self.timeseries = timeseries
        self.kp = None
        self.T = t
        self.SMAX = sMax
        self.SMIN = sMin
        self.is_soft = is_soft
        self.alpha = alpha
        self.window_stats = {}
        self.cnt = 0
        self.w_speed = w_speed
        self.w_causal = w_causal
        self.p_max = p_max
        self.causal_models = None
        self.W = None
        self.b = None
        self._fit_causal_models()

    def _fit_causal_models(self):
        length = int(0.01 * len(self.timeseries.get_timeseries()))
        rows = self.timeseries.get_timeseries()[:length]
        values = np.stack([p.get_truth() for p in rows], axis=0)
        T0, k = values.shape

        tgm_df = TgmDataFrame(data=values)
        pcmci = PCMCI(dataframe=tgm_df,
                      cond_ind_test=ParCorr(), verbosity=0)
        results = pcmci.run_pcmci(tau_max=self.p_max, pc_alpha=0.05)
        q_matrix = pcmci.get_corrected_pvalues(
            p_matrix=results['p_matrix'],
            fdr_method='fdr_bh'
        )
        signif = (q_matrix < 0.1)

        causal_models = {}
        for target in range(k):
            parents = [src for src in range(k)
                       if signif[src, target, 0]]
            if not parents:
                continue
            X = values[1:, parents]
            y = values[1:, target]
            reg = LinearRegression().fit(X, y)
            causal_models[target] = (parents, reg)

        self.causal_models = causal_models

        W = np.zeros((k, k), dtype=float)
        b = np.zeros(k, dtype=float)
        for dim, (parents, reg) in causal_models.items():
            W[dim, parents] = reg.coef_
            b[dim] = reg.intercept_
        self.W = W
        self.b = b


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

        self.window_stats[window_id] = {
            'mu': mu,
            'std': std,
        }

    def _get_window_stats_deque(self, window_deque):
        if len(window_deque) == 0:
            return 0, 1e-6
        return self.window_stats.get(id(window_deque), {'mu': 0, 'std': 1e-6}).values()

    def _compute_soft_score(self, values, window_deque):
        mu, std = self._get_window_stats_deque(window_deque)
        l2_dev = np.linalg.norm(values - mu, axis=1)
        sigma = np.linalg.norm(std) or 1e-6
        s_speed = np.exp(-self.alpha * (l2_dev / sigma))
        s_causal = np.zeros(values.shape[0])
        if self.causal_models and self.w_causal > 0:
            pred_all = values @ self.W.T + self.b  # (T,k)
            resid =  10 * np.abs(values - pred_all)  # (T,k)
            s_causal = np.exp(-self.alpha * (resid ** 2)).mean(axis=1)  # (T,)

        soft_scores = (self.w_speed * s_speed + self.w_causal * s_causal)

        return soft_scores

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
        length = len(window_deque)
        values = np.stack([p.get_value() for p in window_deque])
        T, k = values.shape

        dp = [{'length': 0, 'score': -np.inf, 'prev': -1} for _ in range(length)]
        soft_scores = np.zeros(T)
        if self.is_soft:
            soft_scores = self._compute_soft_score(values, window_deque)

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

        best_end = max(range(length), key = lambda idx: dp[idx]['score'] if self.is_soft else dp[idx]['length'])
        if best_end == 0:
            return

        maxPoint = window_deque[best_end]
        maxTime = maxPoint.get_timestamp()
        maxVal = maxPoint.get_value()

        if self._judge_repair(preVal, maxVal, kpVal, preTime, maxTime, kpTime):
            pre_dis = kpTime - preTime
            pre_next_dis = maxTime - preTime
            rate = pre_dis / pre_next_dis
            modify = (maxVal - preVal) * rate + preVal
            self.kp.set_value(modify)
            self.cnt += 1

if __name__ == "__main__":
    assist = Assist()
    # input_file_name = "stock/stock12k.data"
    # input_file_name = "PSM/PSM.data"
    input_file_name = "exchange/exchange.data"
    # input_file_name = "UCI/AEP.data"
    # input_file_name = "SWaT/SWaT.data"
    # input_file_name = "WADI/WADI.data"
    if 'stock' in input_file_name:
        s_max = 3
        s_min = -3
    elif 'PSM' in input_file_name:
        s_max = 0.1
        s_min = -0.1
        w_speed = 0.5
        w_causal = 0.5
        p_max = 2
    elif 'UCI' in input_file_name:
        s_max = 10
        s_min = -10
        w_speed = 0.5
        w_causal = 0.5
        p_max = 2
    elif 'exchange' in input_file_name:
        s_max = 0.1
        s_min = -0.1
        w_speed = 0.5
        w_causal = 0.5
        p_max = 2
    elif 'SWaT' in input_file_name:
        s_max = 13
        s_min = -13
        w_speed = 0.5
        w_causal = 0.5
        p_max = 2
    elif 'WADI' in input_file_name:
        s_max = 3
        s_min = -3
        w_speed = 0.5
        w_causal = 0.5
        p_max = 2
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
        # drate = 0.05
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
            SHoTClean_Soft = SHoTClean_C(dirty_series, s_max, s_min, t, is_soft=True, w_speed=w_speed, w_causal=w_causal, p_max=p_max)
            start_time = time.time()
            result_series_SHoTClean_Soft = SHoTClean_Soft.mainScreen()
            end_time = time.time()
            rms_SHoTClean_Soft = assist.calc_rms(result_series_SHoTClean_Soft)
            cost_SHoTClean_Soft = assist.calc_cost(result_series_SHoTClean_Soft)
            acc_SHoTClean_Soft = assist.calc_acc(result_series_SHoTClean_Soft)
            print(f"BASE Soft: {rms_SHoTClean_Soft}, cost: {cost_SHoTClean_Soft}, acc: {acc_SHoTClean_Soft}, time: {end_time - start_time}")

            # exit()
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
    # write_file_name = "results/Multi/SWaT/RMS_C.csv"
    # assist.write_csv(write_file_name, name, total_drate, total_rms)
    # write_file_name = "results/Multi/SWaT/COST_C.csv"
    # assist.write_csv(write_file_name, name, total_drate, total_cost)
    # write_file_name = "results/Multi/SWaT/NUM_C.csv"
    # assist.write_csv(write_file_name, name, total_drate, total_acc)
    # write_file_name = "results/Multi/SWaT/TIME_C.csv"
    # assist.write_csv(write_file_name, name, total_drate, total_time)