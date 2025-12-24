import numpy as np
from typing import List, Set, Union

class TimePoint_Single:
    def __init__(self, timestamp, value, truth=None, noise=None):
        self.timestamp = timestamp
        self.value = value
        self.truth = truth if truth is not None else value
        self.noise = noise if noise is not None else value
        self.upperbound = float('inf')
        self.lowerbound = -self.upperbound
        self.status = 0
        self.label = False
        self.minVal = -float('inf')
        self.maxVal = float('inf')

    def get_value(self):
        return self.value

    def set_value(self, value):
        self.value = value

    def get_noise(self):
        return self.noise

    def get_truth(self):
        return self.truth

    def set_status(self, status):
        self.status = status

    def get_timestamp(self):
        return self.timestamp


class TimeSeries_Single:
    def __init__(self):
        self.timeseries = []
        self.class_id = 0
        self.num = 0
        self.num_list = []

    def add_point(self, tp):
        self.timeseries.append(tp)

    def get_length(self):
        return len(self.timeseries)

    def get_timeseries(self):
        return self.timeseries

    def get_sub_timeseries(self, begin, end):
        result = [tp for tp in self.timeseries if begin <= tp.timestamp <= end]
        return TimeSeries().from_list(result)

    def from_list(self, point_list):
        ts = TimeSeries()
        ts.timeseries = point_list
        return ts


class TimePoint:
    def __init__(self, timestamp: float, value: Union[float, List[float], np.ndarray], truth: Union[float, List[float], np.ndarray]=None,
                 noise: Union[float, List[float], np.ndarray]=None):
        self.timestamp = timestamp
        self.value = np.array(value, dtype=float)
        self.noise = np.array(noise, dtype=float) if noise is not None else self.value.copy()
        self.truth = np.array(truth, dtype=float) if truth is not None else self.value.copy()
        self.upperbound = np.full_like(self.value, np.inf)
        self.lowerbound = np.full_like(self.value, -np.inf)
        self.minVal = np.full_like(self.value, -np.inf)
        self.maxVal = np.full_like(self.value, np.inf)
        self.status = 0
        self.label = False

    def get_value(self) -> np.ndarray:
        return self.value

    def set_value(self, new_value: Union[float, List[float], np.ndarray]):
        self.value = np.array(new_value, dtype=float)

    def get_noise(self) -> np.ndarray:
        return self.noise

    def get_truth(self) -> np.ndarray:
        return self.truth

    def set_status(self, status: int):
        self.status = status

    def get_timestamp(self) -> float:
        return self.timestamp


class TimeSeries:
    def __init__(self):
        self.timeseries: List[TimePoint] = []
        self.class_id = 0
        self.num = 0
        self.num_list: List[int] = []

    def add_point(self, tp: TimePoint):
        self.timeseries.append(tp)

    def get_length(self) -> int:
        return len(self.timeseries)

    def get_timeseries(self) -> List[TimePoint]:
        return self.timeseries

    def get_sub_timeseries(self, begin: float, end: float) -> 'TimeSeries':
        result = [tp for tp in self.timeseries if begin <= tp.timestamp <= end]
        return TimeSeries().from_list(result)

    def from_list(self, point_list: List[TimePoint]) -> 'TimeSeries':
        ts = TimeSeries()
        ts.timeseries = point_list.copy()
        return ts