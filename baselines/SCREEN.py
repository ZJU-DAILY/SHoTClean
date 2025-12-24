from tools.entity import TimeSeries

class SCREEN:
    def __init__(self, timeseries, sMax, sMin, t):
        self.timeseries = timeseries
        self.T = t
        self.SMAX = sMax
        self.SMIN = sMin
        self.kp = None

    def main_screen(self):
        total_list = self.timeseries.get_timeseries()
        size = len(total_list)

        pre_end = -1
        pre_point = None
        temp_series = TimeSeries()

        read_index = 1  # the point should be read in

        # initial
        tp = total_list[0]
        temp_series.add_point(tp)
        w_start_time = tp.timestamp
        w_end_time = w_start_time
        w_goal_time = w_start_time + self.T

        while read_index < size:
            tp = total_list[read_index]
            cur_time = tp.timestamp

            if cur_time > w_goal_time:
                while True:
                    temp_list = temp_series.get_timeseries()
                    if not temp_list:
                        # if all the points in tempList has been handled
                        temp_series.add_point(tp)  # the current point should be a new start
                        w_goal_time = cur_time + self.T
                        w_end_time = cur_time
                        break

                    self.kp = temp_list[0]
                    w_start_time = self.kp.timestamp
                    w_goal_time = w_start_time + self.T

                    if cur_time <= w_goal_time:
                        temp_series.add_point(tp)
                        w_end_time = cur_time
                        break

                    cur_end = w_end_time
                    if pre_end == -1:
                        pre_point = self.kp

                    self.local(temp_series, pre_point)

                    pre_point = self.kp
                    pre_point.set_status(1)
                    pre_end = cur_end

                    # remove the keyPoint
                    temp_series.timeseries.pop(0)
            else:
                if cur_time > w_end_time:
                    temp_series.add_point(tp)
                    w_end_time = cur_time

            read_index += 1

        # Handle the last window
        while temp_series.get_length() > 0:
            temp_list = temp_series.get_timeseries()
            self.kp = temp_list[0]
            if pre_point is None:
                pre_point = self.kp

            self.local(temp_series, pre_point)
            pre_point = self.kp
            temp_list.pop(0)

        return self.timeseries

    def local(self, time_series, pre_point):
        temp_list = time_series.get_timeseries()
        pre_time = pre_point.timestamp
        pre_val = pre_point.value
        kp_time = self.kp.timestamp

        lower_bound = pre_val + self.SMIN * (kp_time - pre_time)
        upper_bound = pre_val + self.SMAX * (kp_time - pre_time)

        # form candidates
        xk_list = [self.kp.value]

        for i in range(1, len(temp_list)):
            tp = temp_list[i]
            val = tp.value
            d_time = kp_time - tp.timestamp
            xk_list.append(val + self.SMIN * d_time)
            xk_list.append(val + self.SMAX * d_time)

        xk_list.sort()
        # x_mid = xk_list[-1]
        x_mid = xk_list[len(xk_list) // 2]
        value = x_mid
        if upper_bound < x_mid:
            value = upper_bound
        elif lower_bound > x_mid:
            value = lower_bound

        self.kp.set_value(value)

