import time

import pulp
import numpy as np

from tools.entity import TimePoint, TimeSeries, TimeSeries_Single, TimePoint_Single
from tools.utils import Assist, Assist_Single


def Opt_Solver(timeseries, s_max, s_min, alpha=0.01, is_soft=True):
    series = timeseries.get_timeseries()
    size = len(series)
    time_arr = np.array([p.timestamp for p in series])
    val_arr = np.array([p.value for p in series])

    mu = np.mean(val_arr)
    sigma = np.std(val_arr) + 1e-6

    def decay(dt):
        return np.exp(-0.1 * dt) if is_soft else 1.0

    def valid(j, i):
        dt = time_arr[i] - time_arr[j]
        if dt <= 0: return False
        speed = (val_arr[i] - val_arr[j]) / dt
        return s_min <= speed <= s_max

    preds = [[] for _ in range(size)]
    succs = {i: [] for i in range(size)}
    for i in range(1, size):
        cnt = 0
        for j in range(i-1, -1, -1):
            if valid(j, i):
                preds[i].append(j)
                succs[j].append(i)
                cnt += 1
                if cnt >= 5: break

    scores = np.zeros(size)
    if is_soft:
        dev = np.abs(val_arr - mu) / sigma
        scores = np.exp(-alpha * dev)
    else:
        scores.fill(1.0)
        for i in range(1, size):
            if not any(valid(j, i) for j in range(i)):
                scores[i] = 0.0

    model = pulp.LpProblem('SHoTClean_B1_path', pulp.LpMaximize)
    x = [pulp.LpVariable(f'x_{i}', cat='Binary') for i in range(size)]
    r = [pulp.LpVariable(f'r_{i}', cat='Binary') for i in range(size)]
    e = {(j,i): pulp.LpVariable(f'e_{j}_{i}', cat='Binary')
         for i in range(size) for j in preds[i]}

    obj = pulp.lpSum(scores[i] * r[i] for i in range(size))
    for (j,i), v in e.items():
        obj += scores[i] * decay(i-j) * v
    model += obj

    for i in range(size):
        inc = pulp.lpSum(e[(j,i)] for j in preds[i]) if preds[i] else 0
        model += inc + r[i] == x[i]
    model += pulp.lpSum(r) == 1
    for (j,i), v in e.items():
        model += v <= x[j]
        model += v <= x[i]
    for i in range(size):
        out = pulp.lpSum(e[(i,k)] for k in succs[i]) if succs[i] else 0
        model += out <= 1

    model.solve(pulp.PULP_CBC_CMD(msg=False))

    normal = {i for i in range(size) if pulp.value(x[i]) > 0.5}
    outliers = [i for i in range(size) if i not in normal]

    label = np.ones(size, bool)
    label[outliers] = False
    repaired = val_arr.copy()
    def find_near(idx, direction):
        step = 1 if direction > 0 else -1
        cur = idx + step
        while 0 <= cur < size:
            if label[cur]:
                return cur
            cur += step
        return None

    for i in outliers:
        p0 = find_near(i, -1)
        p1 = find_near(i, 1)
        if p0 is not None and p1 is not None:
            t0, v0 = time_arr[p0], val_arr[p0]
            t1, v1 = time_arr[p1], val_arr[p1]
            ratio = (time_arr[i] - t0) / (t1 - t0)
            repaired[i] = v0 + ratio * (v1 - v0)
        elif p0 is not None:
            repaired[i] = val_arr[p0]
        elif p1 is not None:
            repaired[i] = val_arr[p1]

    clean = TimeSeries_Single()
    for idx, pt in enumerate(series):
        tp = TimePoint_Single(pt.timestamp, repaired[idx], pt.truth)
        tp.label = label[idx]
        clean.add_point(tp)
    return clean


if __name__ == "__main__":
    assist = Assist_Single()
    # input_file_name = "stock/stock12k.data"
    input_file_name = "FRED/TOTALSA.data"
    # input_file_name = "exchange/exchange.data"
    if 'stock' in input_file_name:
        s_max = 3
        s_min = -3
    elif 'FRED' in input_file_name:
        s_max = 2
        s_min = -2
    else:
        s_max = 0.1
        s_min = -0.1
    rmse = 0
    cost = 0
    num = 0
    cost_time = 0
    for i in range(1):
        dirty_series = assist.read_data(input_file_name, ",")
        dirty_series = assist.add_noise(dirty_series, 0.05, i+1)
        start_time = time.time()
        repaired_series = Opt_Solver(dirty_series, s_max, s_min)
        end_time = time.time()
        rms_BASE = assist.calc_rms(repaired_series)
        cost_BASE = assist.calc_cost(repaired_series)
        num_BASE = assist.point_num1(repaired_series)
        cost_time += end_time - start_time
        rmse += rms_BASE
        cost += cost_BASE
        num += num_BASE
    print(f"RMSE: {rmse:.4f}, Cost: {cost:.4f}, Num: {num:.4f}, Time: {cost_time:.4f} seconds")