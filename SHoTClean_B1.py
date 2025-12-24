import time

import numpy as np

from tools.utils import Assist_Single
from tools.entity import TimePoint_Single, TimeSeries_Single
from typing import List, Tuple, Set
import warnings
warnings.filterwarnings("ignore")

"""Batch Setting Algorithm ONLY for Single-dimensional Data"""
class SHoTClean_B1:
    def __init__(self, timeseries, s_max, s_min, alpha=0.01, is_soft=True):
        self.timeseries = timeseries
        self.size = len(timeseries.get_timeseries())
        self.SMAX = s_max
        self.SMIN = s_min
        self.is_soft = is_soft
        self.alpha = alpha
        self._init_prior_distribution(self.timeseries.get_timeseries())

        series = self.timeseries.get_timeseries()
        self.time_array = np.array([p.timestamp for p in series])
        self.value_array = np.array([p.value for p in series])

    def _init_prior_distribution(self, timeseries=None):
        normal_points = [p.value for p in timeseries]
        self.prior_mu = np.mean(normal_points)
        self.prior_std = np.std(normal_points) + 1e-6

    def _temporal_decay(self, gap: int) -> float:
        if self.is_soft:
            return np.exp(-0.1 * gap)
        else:
            return 1.0

    def _compute_scores(self):
        if self.is_soft:
            deviation = np.abs(self.value_array - self.prior_mu) / self.prior_std
            return np.exp(-self.alpha * (deviation))
        else:
            scores = np.ones(self.size)
            for i in range(1, self.size):
                valid = any(self._is_valid_predecessor(j, i) for j in range(i))
                if not valid:
                    scores[i] = 0
            return scores

    def clean(self):
        outlier_index, scores = self.outlier_detection()
        clean_series = self.outlier_repair(outlier_index)
        return clean_series

    def outlier_detection(self):
        scores = self._compute_scores()

        dp = np.zeros(self.size)
        path = np.zeros(self.size, dtype=np.int32)
        max_score = 0.0
        end_idx = 0

        predecessors = self._precompute_predecessors()

        for i in range(self.size):
            dp[i] = scores[i]
            path[i] = i

            for j in predecessors[i]:
                candidate = dp[j] + scores[i] * self._temporal_decay(i-j)
                if candidate > dp[i]:
                    dp[i] = candidate
                    path[i] = j

            if dp[i] > max_score:
                max_score = dp[i]
                end_idx = i

        normal_indices = self._backtrack_path(path, end_idx)
        outlier_indices = [i for i in range(self.size) if i not in normal_indices]
        return outlier_indices, scores

    def _precompute_predecessors(self) -> List[List[int]]:
        predecessors = [[] for _ in range(self.size)]

        for i in range(1, self.size):
            valid_count = 0

            for j in range(i - 1, -1, -1):
                if self._is_valid_predecessor(j, i):
                    predecessors[i].append(j)
                    valid_count += 1
                    if valid_count >= 5:  # 限制前驱数量
                        break
        return predecessors

    def _is_valid_predecessor(self, j: int, i: int) -> bool:
        delta_t = self.time_array[i] - self.time_array[j]
        if delta_t <= 0:
            return False

        delta_v = self.value_array[i] - self.value_array[j]
        speed = delta_v / delta_t

        if self.SMIN <= speed <= self.SMAX:
            return True
        else:
            return False

    def _backtrack_path(self, path: np.ndarray, end_idx: int) -> Set[int]:
        normal_indices = set()
        current = end_idx
        while True:
            normal_indices.add(current)
            if path[current] == current:
                break
            current = path[current]
        return normal_indices

    def outlier_repair(self, outlier_index):
        label = np.ones(self.size, dtype=bool)
        label[outlier_index] = False

        repaired_values = self.value_array.copy()
        for i in outlier_index:
            prev_idx = self._find_nearest_normal(i, label, -1)
            next_idx = self._find_nearest_normal(i, label, 1)

            if prev_idx is not None and next_idx is not None:
                t_prev, v_prev = self.time_array[prev_idx], self.value_array[prev_idx]
                t_next, v_next = self.time_array[next_idx], self.value_array[next_idx]
                ratio = (self.time_array[i] - t_prev) / (t_next - t_prev)
                repaired_values[i] = v_prev + ratio * (v_next - v_prev)
            elif prev_idx is not None:
                repaired_values[i] = self.value_array[prev_idx]
            elif next_idx is not None:
                repaired_values[i] = self.value_array[next_idx]

        clean_series = TimeSeries_Single()
        for i in range(self.size):
            original_point = self.timeseries.get_timeseries()[i]
            tp = TimePoint_Single(
                self.time_array[i],
                repaired_values[i],
                original_point.truth,
                original_point.noise,
            )
            tp.label = label[i]
            clean_series.add_point(tp)

        return clean_series

    def _find_nearest_normal(self, idx: int, label: np.ndarray, direction: int) -> int:
        step = 1 if direction > 0 else -1
        current = idx + step
        while 0 <= current < self.size:
            if label[current]:
                return current
            current += step
        return None


if __name__ == "__main__":
    assist = Assist_Single()
    # input_file_name = "stock/stock12k.data"
    input_file_name = "CA/CA.csv"
    if 'stock' in input_file_name:
        s_max = 3
        s_min = -3
    elif 'CA' in input_file_name:
        s_max = 3000
        s_min = -3000
    else:
        raise ValueError("Unsupported input file name format")

    method_num = 2
    total_drate = np.zeros(10)
    total_rms = np.zeros((10, method_num))
    total_cost = np.zeros((10, method_num))
    total_acc = np.zeros((10, method_num))
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
            SHoTClean_Soft = SHoTClean_B1(dirty_series, s_max, s_min, is_soft=True)
            start_time = time.time()
            result_series_SHoTClean_Soft = SHoTClean_Soft.clean()
            end_time = time.time()
            rms_SHoTClean_Soft = assist.calc_rms(result_series_SHoTClean_Soft)
            cost_SHoTClean_Soft = assist.calc_cost(result_series_SHoTClean_Soft)
            acc_SHoTClean_Soft = assist.calc_acc(result_series_SHoTClean_Soft)
            print(f"RMS: {rms_SHoTClean_Soft}, MAE: {cost_SHoTClean_Soft}, Acc: {acc_SHoTClean_Soft}")

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
    # assist.write_csv(write_file_name, name, total_drate, total_num)
    # write_file_name = "results/One/test/TIME.csv"
    # assist.write_csv(write_file_name, name, total_drate, total_time)