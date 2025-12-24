import time

import numpy as np
from tools.utils import Assist
from tools.entity import TimePoint, TimeSeries
from typing import List, Tuple, Set, Union
import warnings
warnings.filterwarnings("ignore")


"""Batch Setting Algorithm for Multi(Single)-dimensional Data"""
class SHoTClean_B:
    def __init__(
        self,
        timeseries: TimeSeries,
        s_max: Union[float, List[float], np.ndarray],
        s_min: Union[float, List[float], np.ndarray],
        alpha: float = 0.01,
        is_soft: bool = True
    ):
        self.timeseries = timeseries
        self.size = len(timeseries.get_timeseries())

        self.SMAX = np.array(s_max, dtype=float) if not np.isscalar(s_max) else float(s_max)
        self.SMIN = np.array(s_min, dtype=float) if not np.isscalar(s_min) else float(s_min)

        self.is_soft = is_soft
        self.alpha = alpha

        series = self.timeseries.get_timeseries()
        self.time_array = np.array([p.timestamp for p in series], dtype=float)

        raw_values = np.array([p.value for p in series], dtype=float)
        if raw_values.ndim == 1:
            self.value_array = raw_values.reshape(-1, 1)
        else:
            self.value_array = raw_values
        self.N, self.D = self.value_array.shape

        self._init_prior_distribution()

    def _init_prior_distribution(self):
        self.prior_mu = np.mean(self.value_array, axis=0)   # (D,)
        self.prior_std = np.std(self.value_array, axis=0)   # (D,)
        self.prior_std[self.prior_std < 1e-6] = 1e-6

    def _temporal_decay(self, gap: int) -> float:
        if self.is_soft:
            return np.exp(-0.1 * gap)
        else:
            return 1.0

    def _compute_scores(self) -> np.ndarray:
        if self.is_soft:
            deviation = (self.value_array - self.prior_mu) / self.prior_std
            deviation_norm = np.linalg.norm(deviation, axis=1)
            scores = np.exp(-self.alpha * (deviation_norm))
            return scores
        else:
            return np.ones(self.N, dtype=float)

    def clean(self) -> TimeSeries:
        outlier_indices, scores = self.outlier_detection()
        clean_series = self.outlier_repair(outlier_indices, scores)
        return clean_series

    def outlier_detection(self) -> (List[int], np.ndarray):
        scores = self._compute_scores()
        dp = np.zeros(self.N, dtype=float)
        path = np.zeros(self.N, dtype=np.int32)
        max_score = -np.inf
        end_idx = 0

        predecessors = self._precompute_predecessors()

        for i in range(self.N):
            dp[i] = scores[i]
            path[i] = i
            for j in predecessors[i]:
                candidate = dp[j] + scores[i] * self._temporal_decay(i - j)
                if candidate > dp[i]:
                    dp[i] = candidate
                    path[i] = j
            if dp[i] > max_score:
                max_score = dp[i]
                end_idx = i

        normal_indices = self._backtrack_path(path, end_idx)
        outlier_indices = [idx for idx in range(self.N) if idx not in normal_indices]
        return outlier_indices, scores

    def _precompute_predecessors(self) -> List[List[int]]:
        predecessors: List[List[int]] = [[] for _ in range(self.N)]
        for i in range(1, self.N):
            valid_count = 0
            for j in range(i - 1, -1, -1):
                if self._is_valid_predecessor(j, i):
                    predecessors[i].append(j)
                    valid_count += 1
                    if valid_count >= 5:
                        break
        return predecessors

    def _is_valid_predecessor(self, j: int, i: int) -> bool:
        delta_t = self.time_array[i] - self.time_array[j]
        if delta_t <= 0:
            return False

        delta_v = self.value_array[i] - self.value_array[j]
        speed_vec = delta_v / delta_t

        if np.isscalar(self.SMIN) and np.isscalar(self.SMAX):
            speed_norm = np.linalg.norm(speed_vec)
            return (self.SMIN <= speed_norm <= self.SMAX)
        else:
            SMIN_vec = np.array(self.SMIN, dtype=float)
            SMAX_vec = np.array(self.SMAX, dtype=float)
            return np.all(speed_vec >= SMIN_vec) and np.all(speed_vec <= SMAX_vec)

    def _backtrack_path(self, path: np.ndarray, end_idx: int) -> Set[int]:
        normal_indices: Set[int] = set()
        cur = end_idx
        while True:
            normal_indices.add(cur)
            if path[cur] == cur:
                break
            cur = path[cur]
        return normal_indices

    def outlier_repair(self, outlier_indices: List[int], scores: np.ndarray) -> TimeSeries:
        label = np.ones(self.N, dtype=bool)
        label[outlier_indices] = False

        repaired_values = self.value_array.copy()

        for i in outlier_indices:
            prev_idx = self._find_nearest_normal(i, label, direction=-1)
            next_idx = self._find_nearest_normal(i, label, direction=1)

            if prev_idx is not None and next_idx is not None:
                t_prev = self.time_array[prev_idx]
                t_next = self.time_array[next_idx]
                v_prev = self.value_array[prev_idx]   # (D,)
                v_next = self.value_array[next_idx]   # (D,)
                ratio = (self.time_array[i] - t_prev) / (t_next - t_prev)
                repaired_values[i, :] = v_prev + ratio * (v_next - v_prev)
            elif prev_idx is not None:
                repaired_values[i, :] = self.value_array[prev_idx]
            elif next_idx is not None:
                repaired_values[i, :] = self.value_array[next_idx]
            else:
                repaired_values[i, :] = self.prior_mu

        clean_series = TimeSeries()
        for idx in range(self.N):
            orig_point = self.timeseries.get_timeseries()[idx]
            new_tp = TimePoint(
                self.time_array[idx],
                repaired_values[idx, :],
                orig_point.truth,
                orig_point.noise
            )
            new_tp.label = bool(label[idx])
            clean_series.add_point(new_tp)
        return clean_series

    def _find_nearest_normal(self, idx: int, label: np.ndarray, direction: int) -> Union[int, None]:
        step = 1 if direction > 0 else -1
        cur = idx + step
        while 0 <= cur < self.N:
            if label[cur]:
                return cur
            cur += step
        return None

if __name__ == "__main__":
    assist = Assist()
    # input_file_name = "exchange/exchange.data"
    input_file_name = "UCI/AEP.data"
    # input_file_name = "stock/stock12k.data"
    # input_file_name = "SWaT/SWaT.data"
    # input_file_name = "PSM/PSM.data"
    # input_file_name = "WADI/WADI.data"
    # input_file_name = "Porto/porto_10.data"
    if 'exchange' in input_file_name:
        s_max = 0.1
        s_min = -0.1
    elif 'UCI' in input_file_name:
        s_max = 10
        s_min = -10
    elif 'stock' in input_file_name:
        s_max = 3
        s_min = -3
    elif 'SWaT' in input_file_name:
        s_max = 13.0
        s_min = -13.0
    elif 'PSM' in input_file_name:
        s_max = 0.15
        s_min = -0.15
    elif 'porto' in input_file_name:
        s_max = 0.05
        s_min = -0.05
    elif 'WADI' in input_file_name:
        s_max = 3
        s_min = -3
    else:
        s_max = 0.1
        s_min = -0.1
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
            SHoTClean_Soft = SHoTClean_B(dirty_series, s_max, s_min, is_soft=True)
            start_time = time.time()
            result_series_SHoTClean_Soft = SHoTClean_Soft.clean()
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
    # write_file_name = "results/Multi/SWaT/RMS_B_size.csv"
    # assist.write_csv(write_file_name, name, total_drate, total_rms)
    # write_file_name = "results/Multi/SWaT/COST_B_size.csv"
    # assist.write_csv(write_file_name, name, total_drate, total_cost)
    # write_file_name = "results/Multi/SWaT/ACC_B_size.csv"
    # assist.write_csv(write_file_name, name, total_drate, total_acc)
    # write_file_name = "results/Multi/SWaT/TIME_B_size.csv"
    # assist.write_csv(write_file_name, name, total_drate, total_time)