import numpy as np

def _is_valid_predecessor(values, j, i, smin, smax):
    delta_t = i - j
    if delta_t <= 0:
        return False

    delta_v = values[i] - values[j]
    speed = delta_v / delta_t

    if smin <= speed <= smax:
        return True
    else:
        return False

def _precompute_predecessors(values, smin=-1, smax=1):
    predecessors = [[] for _ in range(len(values))]

    for i in range(1, len(values)):
        for j in range(i - 1, -1, -1):
            if _is_valid_predecessor(values, j, i, smin, smax):
                predecessors[i].append(j)
    return predecessors

def _backtrack_path(paths, end_idx):
    normal_indices = set()
    current = end_idx
    while True:
        normal_indices.add(current)
        if paths[current] == current:
            break
        current = paths[current]
    return normal_indices

is_soft = True
values = [0.35, 0.40, 0.45, 0.50, 0.90, 1.15, 0.65, 0.60, 0.55]
smin   = -0.50001
smax   = 0.30001

avg = sum(values) / len(values)
std = (sum((x - avg) ** 2 for x in values) / len(values)) ** 0.5

print(f"Average: {avg}")
print(f"Standard Deviation: {std}")

if is_soft:
    scores = np.exp(-(np.array(values) - avg) / std)
else:
    scores = np.ones(len(values))

print(f"Scores: {scores}")
path = np.zeros(len(values), dtype=int)
dp = np.zeros(len(values))
max_score = 0.0
end_idx = 0
predecessors = _precompute_predecessors(values, smin, smax)
print(predecessors)

for i in range(len(values)):
    dp[i] = scores[i]
    path[i] = i
    for j in predecessors[i]:
        candidate = dp[j] + scores[i] * np.exp(-0.1 * (i-j))
        if candidate > dp[i]:
            dp[i] = candidate
            path[i] = j

    if dp[i] > max_score:
        max_score = dp[i]
        end_idx = i

print(dp)
print(path)
print(f"Max Score: {max_score} at Index: {end_idx}")
normal_indices = _backtrack_path(path, end_idx)
outlier_indices = [i for i in range(len(values)) if i not in normal_indices]
print(f"Normal Indices: {normal_indices}")
print(f"Outlier Indices: {outlier_indices}")
for i in outlier_indices:
    print(f"Outlier Value: {values[i]} at Index: {i}")