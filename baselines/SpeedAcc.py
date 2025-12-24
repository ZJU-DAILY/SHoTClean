
class SpeedAcc:
    def __init__(self, time_series, T, S_MAX, S_MIN, A_MAX, A_MIN):
        self.time_series = time_series
        self.T = T
        self.S_MAX = S_MAX
        self.S_MIN = S_MIN
        self.A_MAX = A_MAX
        self.A_MIN = A_MIN

    def main_sli_up(self):
        total_list = self.time_series.get_timeseries()
        total = len(total_list)
        if total < 2:
            return self.time_series

        kT = [total - 1] * total
        kn, ki = 0, 1
        while kn < total and ki < total:
            if total_list[ki].timestamp - total_list[kn].timestamp > self.T:
                kT[kn] = ki
                kn += 1
                ki = kn + 1
            else:
                ki += 1

        tpk = total_list.copy()
        xkmin = [float('-inf')] * total
        xkmax = [float('inf')] * total
        xkminS = [0.0] * total
        xkmaxS = [0.0] * total

        for k in range(2):
            xK = []
            if k != 0:
                delta_t = tpk[k].timestamp - tpk[k - 1].timestamp
                xkmin[k] = tpk[k - 1].value + self.S_MIN * delta_t
                xkmax[k] = tpk[k - 1].value + self.S_MAX * delta_t

            for i in range(k + 1, kT[k]):
                if total_list[i].timestamp - total_list[k].timestamp <= self.T:
                    delta = tpk[k].timestamp - tpk[i].timestamp
                    val_speed_min = tpk[i].value + self.S_MIN * delta
                    val_speed_max = tpk[i].value + self.S_MAX * delta
                    xK.extend([val_speed_min, val_speed_max])

            xK.append(tpk[k].value)
            xK.sort()
            xK_mid = xK[len(xK) // 2]

            if xkmax[k] < xK_mid:
                tpk[k].value = xkmax[k]
            elif xkmin[k] > xK_mid:
                tpk[k].value = xkmin[k]
            else:
                tpk[k].value = xK_mid

        for k in range(2, total):
            xK = []
            prev_delta_t = tpk[k - 1].timestamp - tpk[k - 2].timestamp
            current_delta_t = tpk[k].timestamp - tpk[k - 1].timestamp

            if prev_delta_t == 0:
                prev_speed = 0
            else:
                prev_speed = (tpk[k - 1].value - tpk[k - 2].value) / prev_delta_t

            xkmin[k] = (self.A_MIN * current_delta_t + prev_speed) * current_delta_t + tpk[k - 1].value
            xkmax[k] = (self.A_MAX * current_delta_t + prev_speed) * current_delta_t + tpk[k - 1].value

            xkminS[k] = tpk[k - 1].value + self.S_MIN * current_delta_t
            xkmaxS[k] = tpk[k - 1].value + self.S_MAX * current_delta_t

            xkmin[k] = max(xkmin[k], xkminS[k])
            xkmax[k] = min(xkmax[k], xkmaxS[k])

            for j in range(k + 1, kT[k]):
                for i in range(j + 1, kT[k]):
                    if total_list[i].timestamp - total_list[k].timestamp <= self.T:
                        delta_t_ij = tpk[i].timestamp - tpk[j].timestamp
                        if delta_t_ij == 0:
                            continue

                        speed_ij = (tpk[i].value - tpk[j].value) / delta_t_ij
                        delta_t_jk = tpk[j].timestamp - tpk[k].timestamp

                        val_acc_min = (self.A_MIN - speed_ij) * delta_t_jk + tpk[j].value
                        val_acc_max = (self.A_MAX - speed_ij) * delta_t_jk + tpk[j].value
                        xK.extend([val_acc_min, val_acc_max])

                        val_speed_min = tpk[j].value + self.S_MIN * delta_t_jk
                        val_speed_max = tpk[j].value + self.S_MAX * delta_t_jk
                        xK.extend([val_speed_min, val_speed_max])

            xK.append(tpk[k].value)
            xK.sort()
            xK_mid = xK[len(xK) // 2]

            if xkmax[k] < xK_mid:
                tpk[k].value = xkmax[k]
            elif xkmin[k] > xK_mid:
                tpk[k].value = xkmin[k]
            else:
                tpk[k].value = xK_mid

        return self.time_series