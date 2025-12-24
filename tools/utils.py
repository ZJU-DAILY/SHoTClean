import os
import random
import csv
from typing import Union, Tuple, List

import numpy as np
from .entity import TimePoint, TimeSeries, TimeSeries_Single

class Assist_Single():
    if os.path.exists("../datasets/"):
        PATH = "../datasets/"
    else:
        PATH = "datasets/"

    def __init__(self):
        self.anomaly_indices = set()

    def save_data(self, time_series, filename, split_op=","):
        try:
            with open(os.path.join(self.PATH, filename), 'w', encoding="utf-8-sig") as file:
                timeseries = time_series.get_timeseries()
                for point in timeseries:
                    line = f"{point.timestamp}{split_op}{point.value}{split_op}{point.truth}\n"
                    file.write(line)
        except Exception as e:
            print(f"Error saving file: {e}")

    def read_data(self, filename, split_op):
        time_series = TimeSeries()
        try:
            with open(os.path.join(self.PATH, filename), 'r', encoding="utf-8-sig") as file:
                for line in file:
                    vals = line.strip().split(split_op)
                    timestamp = int(vals[0])
                    value = float(vals[1])
                    truth = float(vals[2])
                    time_series.add_point(TimePoint(timestamp, value, truth))
        except FileNotFoundError as e:
            print(f"Error reading file: {e}")
        return time_series

    def add_noise(self, time_series, drate, seed):
        random.seed(seed)
        np.random.seed(seed)
        noisy_series = TimeSeries()
        len_series = time_series.get_length()
        error_num = int(len_series * drate)
        min_error_index = 100
        min_val, max_val = self.get_min_max(time_series)
        noise_len = max_val - min_val
        avg = np.mean([p.get_value() for p in time_series.get_timeseries()])
        noise_std = avg * 0.3

        index_list = set()
        for _ in range(error_num):
            index = random.randint(min_error_index, len_series - 1)
            while index in index_list:
                index = random.randint(min_error_index, len_series - 1)
            index_list.add(index)

        self.anomaly_indices = index_list

        for i in range(len_series):
            if i not in index_list:
                timestamp = time_series.get_timeseries()[i].get_timestamp()
                truth = time_series.get_timeseries()[i].get_truth()
                noisy_series.add_point(TimePoint(timestamp, truth, truth))
            else:
                timestamp = time_series.get_timeseries()[i].get_timestamp()
                truth = time_series.get_timeseries()[i].get_truth()
                noise_guss = np.random.normal(0, noise_std)
                noisy_value = truth + noise_guss
                noise = max(min(noisy_value, max_val), min_val)
                noisy_series.add_point(TimePoint(timestamp, noise, truth))
        return noisy_series

    def add_noise_continuous(self, time_series, drate, seed, max_segment_ratio=0.05):
        random.seed(seed)
        np.random.seed(seed)

        noisy_series = TimeSeries()
        len_series = time_series.get_length()
        error_num = int(len_series * drate)
        min_error_index = 100
        min_val, max_val = self.get_min_max(time_series)
        avg = np.mean([p.get_value() for p in time_series.get_timeseries()])
        noise_std = avg * 0.3

        max_segment_len = max(1, int(len_series * max_segment_ratio * drate))

        segment_lengths = []
        remaining = error_num
        while remaining > 0:
            seg = random.randint(1, min(max_segment_len, remaining))
            segment_lengths.append(seg)
            remaining -= seg

        index_list = set()
        for seg in segment_lengths:
            while True:
                start_idx = random.randint(min_error_index, len_series - seg)
                candidate = set(range(start_idx, start_idx + seg))
                if not any(((idx - 1) in index_list or idx in index_list or (idx + 1) in index_list)
                           for idx in candidate):
                    index_list.update(candidate)
                    break

        self.anomaly_indices = index_list

        for i in range(len_series):
            timestamp = time_series.get_timeseries()[i].get_timestamp()
            truth = time_series.get_timeseries()[i].get_truth()
            if i in index_list:
                noise_guss = np.random.normal(0, noise_std)
                noisy_value = truth + noise_guss
                noise = max(min(noisy_value, max_val), min_val)
                noisy_series.add_point(TimePoint(timestamp, noise, truth))
            else:
                noisy_series.add_point(TimePoint(timestamp, truth, truth))

        return noisy_series

    def get_min_max(self, time_series):
        min_val = float('inf')
        max_val = -float('inf')
        ts = time_series.get_timeseries()
        for tp in ts:
            min_val = min(min_val, tp.get_truth())
            max_val = max(max_val, tp.get_truth())
        return min_val, max_val

    def calc_rms(self, time_series):
        cost = 0
        len_series = time_series.get_length()
        ts = time_series.get_timeseries()
        for p in ts:
            delta = p.get_value() - p.get_truth()
            cost += delta ** 2
        cost /= len_series
        return np.sqrt(cost)

    def calc_cost(self, time_series):
        cost = 0
        ts = time_series.get_timeseries()
        for tp in ts:
            cost += abs(tp.get_value() - tp.get_truth())
        cost /= len(ts)
        return cost

    def calc_acc(self, time_series):
        r1 = 0
        r2 = 0
        r3 = 0
        ts = time_series.get_timeseries()
        for tp in ts:
            r1 += abs(tp.get_value() - tp.get_truth())
            r2 += abs(tp.get_truth() - tp.get_noise())
            r3 += abs(tp.get_value() - tp.get_noise())
        return 1 - r1 / (r2 + r3)

    def point_num1(self, time_series):
        num = 0
        ts = time_series.get_timeseries()
        for tp in ts:
            if tp.get_value() != tp.get_truth():
                num += 1
        return num

    def calc_repaired_cost(self, time_series):
        cost = 0
        ts = time_series.get_timeseries()
        for tp in ts:
            cost += abs(tp.get_value() - tp.get_noise())
        cost /= len(ts)
        return cost

    def write_csv(self, write_filename, name, value, data):
        try:
            with open(write_filename, mode='w', newline='') as file:
                writer = csv.writer(file, delimiter='\t')
                writer.writerow([name[0]] + value.tolist())
                data_transposed = np.array(data).T
                for i, column in enumerate(data_transposed):
                    writer.writerow([name[i+1]] + [round(x, 4) for x in column])
        except IOError as e:
            print(f"Error writing file: {e}")

    def normalize(self, time_series):
        min_value, max_value = self.get_min_max(time_series)
        normalized_series = []
        for timestamp, value, truth in time_series:
            normalized_value = (value - min_value) / (max_value - min_value)
            normalized_truth = (truth - min_value) / (max_value - min_value)
            normalized_series.append((timestamp, normalized_value, normalized_truth))
        return normalized_series


class Assist():
    if os.path.exists("../datasets/"):
        PATH = "../datasets/"
    else:
        PATH = "datasets/"

    def save_data(self, time_series: TimeSeries, filename: str, split_op: str = ","):
        try:
            with open(os.path.join(self.PATH, filename), 'w', encoding="utf-8-sig") as file:
                for point in time_series.get_timeseries():
                    ts = int(point.get_timestamp())
                    val_arr = np.array(point.get_value(), dtype=float)
                    truth_arr = np.array(point.get_truth(), dtype=float)

                    if val_arr.ndim == 0:
                        val_arr = val_arr.reshape(1,)
                    if truth_arr.ndim == 0:
                        truth_arr = truth_arr.reshape(1,)

                    val_list = val_arr.tolist()
                    truth_list = truth_arr.tolist()

                    row = [str(ts)] + [str(v) for v in val_list] + [str(t) for t in truth_list]
                    line = split_op.join(row) + "\n"
                    file.write(line)
        except Exception as e:
            print(f"Error saving file: {e}")

    def read_data(self, filename: str, split_op: str = ",", size: float = 1.0) -> TimeSeries:
        time_series = TimeSeries()
        try:
            with open(os.path.join(self.PATH, filename), 'r', encoding="utf-8-sig") as file:
                lines = file.readlines()
                num = int(size * len(lines))
                for line in lines[:num]:
                    vals = line.strip().split(split_op)
                    if len(vals) < 3:
                        continue
                    timestamp = float(vals[0])
                    L = len(vals) - 1
                    if L % 2 != 0:
                        raise ValueError(f"Line has unexpected number of columns: {len(vals)}")
                    D = L // 2
                    value = [float(vals[i]) for i in range(1, 1 + D)]
                    truth = [float(vals[i]) for i in range(1 + D, 1 + 2 * D)]
                    time_series.add_point(TimePoint(timestamp, value, truth))

        except FileNotFoundError as e:
            print(f"Error reading file: {e}")
        except Exception as e:
            print(f"Error parsing file: {e}")
        return time_series

    def add_noise(self, time_series: TimeSeries, drate: float, seed: int) -> TimeSeries:
        random.seed(seed)
        np.random.seed(seed)
        noisy_series = TimeSeries()
        len_series = time_series.get_length()
        error_num = int(len_series * drate)

        min_val, max_val = self.get_min_max(time_series)

        truths = np.array([p.get_truth() for p in time_series.get_timeseries()], dtype=float)
        avg = np.mean(truths, axis=0)
        noise_std = avg * 0.3

        index_list = set()
        min_error_index = 100
        for _ in range(error_num):
            idx = random.randint(min_error_index, len_series - 1)
            while idx in index_list:
                idx = random.randint(min_error_index, len_series - 1)
            index_list.add(idx)

        for i in range(len_series):
            orig_tp = time_series.get_timeseries()[i]
            ts = orig_tp.get_timestamp()
            truth = np.array(orig_tp.get_truth(), dtype=float)
            if i not in index_list:
                noisy_series.add_point(TimePoint(ts, truth, truth))
            else:
                noise_guss = np.random.normal(loc=0.0, scale=abs(noise_std))
                noisy_value = truth + noise_guss
                noise_clipped = np.clip(noisy_value, min_val, max_val)
                noisy_series.add_point(TimePoint(ts, noise_clipped, truth))
        return noisy_series

    def add_noise_continuous(self, time_series: TimeSeries, drate: float, seed: int, max_segment_ratio: float = 0.1) -> TimeSeries:
        random.seed(seed)
        np.random.seed(seed)

        noisy_series = TimeSeries()
        len_series = time_series.get_length()
        total_noise_points = int(len_series * drate)
        max_segment_len = max(1, int(len_series * max_segment_ratio * drate))

        min_val, max_val = self.get_min_max(time_series)

        truths = np.array([p.get_truth() for p in time_series.get_timeseries()], dtype=float)
        avg = np.mean(truths, axis=0)
        noise_std = avg * 0.3

        noise_indices = set()
        current_noise_count = 0

        segment_lengths = []
        remaining = total_noise_points
        while remaining > 0:
            seg = random.randint(1, min(max_segment_len, remaining))
            segment_lengths.append(seg)
            remaining -= seg

        for seg in segment_lengths:
            start_idx = random.randint(100, len_series - seg)
            while any(idx in noise_indices for idx in range(start_idx, start_idx + seg)):
                start_idx = random.randint(100, len_series - seg)
            for i in range(start_idx, start_idx + seg):
                noise_indices.add(i)

        for i in range(len_series):
            orig_tp = time_series.get_timeseries()[i]
            ts = orig_tp.get_timestamp()
            truth = np.array(orig_tp.get_truth(), dtype=float)
            if i not in noise_indices:
                noisy_series.add_point(TimePoint(ts, truth, truth))
            else:
                noise_guss = np.random.normal(loc=0.0, scale=abs(noise_std))
                noisy_value = truth + noise_guss
                noise_clipped = np.clip(noisy_value, min_val, max_val)
                noisy_series.add_point(TimePoint(ts, noise_clipped, truth))

        return noisy_series

    def add_noise_random_dimension(self, time_series: TimeSeries, drate: float, seed: int, max_dim_rate: float = 1.0) -> TimeSeries:
        random.seed(seed)
        np.random.seed(seed)
        noisy_series = TimeSeries()
        len_series = time_series.get_length()
        error_num = int(len_series * drate)

        min_val, max_val = self.get_min_max(time_series)

        truths = np.array([p.get_truth() for p in time_series.get_timeseries()], dtype=float)
        avg = np.mean(truths, axis=0)
        noise_std = avg * 0.3

        index_list = set()
        min_error_index = 100
        for _ in range(error_num):
            idx = random.randint(min_error_index, len_series - 1)
            while idx in index_list:
                idx = random.randint(min_error_index, len_series - 1)
            index_list.add(idx)

        for i in range(len_series):
            orig_tp = time_series.get_timeseries()[i]
            ts = orig_tp.get_timestamp()
            truth = np.array(orig_tp.get_truth(), dtype=float)
            if i not in index_list:
                noisy_series.add_point(TimePoint(ts, truth, truth))
            else:
                noise_guss = np.random.normal(loc=0.0, scale=abs(noise_std))
                noisy_value = truth + noise_guss
                noise_clipped = np.clip(noisy_value, min_val, max_val)
                noise = truth.copy()

                dim = len(truth)
                num_replace = random.randint(1, dim*max_dim_rate)
                replace_dims = random.sample(range(dim), num_replace)

                for d in replace_dims:
                    noise[d] = noise_clipped[d]

                noisy_series.add_point(TimePoint(ts, noise, truth))
        return noisy_series

    def get_min_max(self, time_series: TimeSeries) -> (Union[float, np.ndarray], Union[float, np.ndarray]):
        ts_points = time_series.get_timeseries()
        if len(ts_points) == 0:
            return 0.0, 0.0
        truths = np.array([tp.get_truth() for tp in ts_points], dtype=float)
        if truths.ndim == 1:
            min_val = float(np.min(truths))
            max_val = float(np.max(truths))
        else:
            min_val = truths.min(axis=0)
            max_val = truths.max(axis=0)
        return min_val, max_val

    def calc_rms(self, time_series: TimeSeries) -> float:
        ts_points = time_series.get_timeseries()
        len_series = len(ts_points)
        if len_series == 0:
            return 0.0

        first_val = np.array(ts_points[0].get_value(), dtype=float)
        D = 1 if first_val.ndim == 0 else first_val.shape[0]

        total_sq = 0.0
        for tp in ts_points:
            diff = np.array(tp.get_value(), dtype=float) - np.array(tp.get_truth(), dtype=float)
            total_sq += np.sum(diff ** 2)

        mse = total_sq / (len_series * D)
        return float(np.sqrt(mse))

    def calc_cost(self, time_series: TimeSeries) -> float:
        ts_points = time_series.get_timeseries()
        len_series = len(ts_points)
        if len_series == 0:
            return 0.0

        first_val = np.array(ts_points[0].get_value(), dtype=float)
        D = 1 if first_val.ndim == 0 else first_val.shape[0]

        total_abs = 0.0
        for tp in ts_points:
            diff = np.abs(np.array(tp.get_value(), dtype=float) - np.array(tp.get_truth(), dtype=float))
            total_abs += np.sum(diff)

        mae = total_abs / (len_series * D)
        return float(mae)

    def calc_acc(self, time_series):
        ts_points = time_series.get_timeseries()
        first_val = np.array(ts_points[0].get_value(), dtype=float)
        D = 1 if first_val.ndim == 0 else first_val.shape[0]
        total_pred_err = 0.0
        total_base_err = 0.0
        for tp in ts_points:
            val = np.array(tp.get_value(), dtype=float)
            truth = np.array(tp.get_truth(), dtype=float)
            noise = np.array(tp.get_noise(), dtype=float)
            total_pred_err += np.sum(np.abs(val - truth))
            total_base_err += np.sum(np.abs(truth - noise) + np.abs(val - noise))
        return float(1 - total_pred_err / total_base_err)

    def point_num1(self, time_series: TimeSeries) -> int:
        count = 0
        for tp in time_series.get_timeseries():
            val = np.array(tp.get_value(), dtype=float)
            truth = np.array(tp.get_truth(), dtype=float)
            if not np.array_equal(val, truth):
                count += 1
        return count

    def write_csv(self, write_filename: str, name: list, value: np.ndarray, data: list):
        try:
            with open(write_filename, mode='w', newline='') as file:
                writer = csv.writer(file, delimiter='\t')
                writer.writerow([name[0]] + value.tolist())
                data_transposed = np.array(data).T
                for i, column in enumerate(data_transposed):
                    writer.writerow([name[i+1]] + [round(x, 4) for x in column])
        except IOError as e:
            print(f"Error writing file: {e}")

    def normalize(self, time_series: TimeSeries) -> TimeSeries:
        min_val, max_val = self.get_min_max(time_series)
        normalized_ts = TimeSeries()

        for tp in time_series.get_timeseries():
            ts = tp.get_timestamp()
            val = np.array(tp.get_value(), dtype=float)
            truth = np.array(tp.get_truth(), dtype=float)

            denom = max_val - min_val
            with np.errstate(divide='ignore', invalid='ignore'):
                norm_val = np.where(denom != 0, (val - min_val) / denom, 0.0)
                norm_truth = np.where(denom != 0, (truth - min_val) / denom, 0.0)

            normalized_ts.add_point(TimePoint(ts, norm_val, norm_truth))

        return normalized_ts

    @staticmethod
    def get_speed_limits(time_series, rate: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        pts = time_series.get_timeseries()
        if len(pts) < 2:
            raise ValueError("time-series must contain at least two points")

        n_dim = len(pts[0].get_truth())
        speeds: List[List[float]] = [[] for _ in range(n_dim)]

        for p1, p2 in zip(pts[:-1], pts[1:]):
            dt = p2.get_timestamp() - p1.get_timestamp()
            if dt == 0:
                continue
            v1 = p1.get_truth()
            v2 = p2.get_truth()
            for d in range(n_dim):
                speeds[d].append((v2[d] - v1[d]) / dt)

        s_max = np.zeros(n_dim)
        s_min = np.zeros(n_dim)

        for d in range(n_dim):
            arr = np.asarray(speeds[d])
            if arr.size == 0:
                s_max[d] = 0.0
                s_min[d] = 0.0
                continue

            if rate >= 1.0:
                s_max[d] = arr.max()
                s_min[d] = arr.min()
            else:
                s_max[d] = np.quantile(arr, rate)
                s_min[d] = np.quantile(arr, 1.0 - rate)

        return s_max, s_min

    def split_by_dimension(self, time_series: TimeSeries) -> List[TimeSeries]:
        pts = time_series.get_timeseries()
        if not pts:
            return []

        first_val = pts[0].get_value()
        if isinstance(first_val, (list, tuple, np.ndarray)):
            D = len(first_val)
        else:
            D = 1

        series_list: List[TimeSeries] = [TimeSeries() for _ in range(D)]

        for tp in pts:
            ts = tp.get_timestamp()
            val = tp.get_value()
            truth = tp.get_truth()

            if D == 1:
                series_list[0].add_point(TimePoint(ts, val, truth))
            else:
                for d in range(D):
                    val_d = val[d]
                    truth_d = truth[d]
                    series_list[d].add_point(TimePoint(ts, val_d, truth_d))

        return series_list
