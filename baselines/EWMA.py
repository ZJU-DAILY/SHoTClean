
class EWMA:
    def __init__(self, time_series, alpha):
        self.time_series = time_series
        self.alpha = alpha
        self.temp_list = self.time_series.get_timeseries()

    def set_alpha(self, alpha):
        self.alpha = alpha

    def main_exp(self):
        if not self.temp_list:
            return self.time_series

        first_point = self.temp_list[0]
        s = first_point.get_truth()
        x = s
        old_stamp = first_point.get_timestamp()

        for tp in self.temp_list:
            cur_stamp = tp.get_timestamp()
            while cur_stamp > old_stamp:
                s = self.alpha * x + (1 - self.alpha) * s
                old_stamp += 1
            x = tp.get_value()
            tp.set_value(s)
        return self.time_series