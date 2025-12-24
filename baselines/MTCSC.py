from tools.entity import TimeSeries


class MTCSC:
    def __init__(self, timeseries, sMax, sMin, t):
        self.timeseries = timeseries
        self.kp = None
        self.T = t 
        self.SMAX = sMax 
        self.SMIN = sMin 

    def mainScreen(self):
        totalList = self.timeseries.get_timeseries()
        size = len(totalList)

        preEnd = -1
        prePoint = None 

        tempSeries = TimeSeries()
        tempList = []

        readIndex = 1
        tp = totalList[0]
        tempSeries.add_point(tp)
        wStartTime = tp.get_timestamp()
        wEndTime = wStartTime
        wGoalTime = wStartTime + self.T

        while readIndex < size:
            tp = totalList[readIndex]
            curTime = tp.get_timestamp()

            if curTime > wGoalTime:
                while True:
                    tempList = tempSeries.get_timeseries()
                    if len(tempList) == 0:
                        tempSeries.add_point(tp)
                        wGoalTime = curTime + self.T
                        wEndTime = curTime
                        break

                    self.kp = tempList[0]
                    wStartTime = self.kp.get_timestamp()
                    wGoalTime = wStartTime + self.T

                    if curTime <= wGoalTime:
                        tempSeries.add_point(tp)
                        wEndTime = curTime
                        break

                    curEnd = wEndTime

                    if preEnd == -1:
                        prePoint = self.kp

                    self.local(tempSeries, prePoint)

                    prePoint = self.kp
                    preEnd = curEnd

                    tempSeries.get_timeseries().pop(0)
            else:
                if curTime > wEndTime:
                    tempSeries.add_point(tp)
                    wEndTime = curTime
            readIndex += 1

        while len(tempSeries.get_timeseries()) > 0:
            tempList = tempSeries.get_timeseries()
            self.kp = tempList[0]
            if prePoint is None:
                prePoint = self.kp
            self.local(tempSeries, prePoint)
            prePoint = self.kp
            tempList.pop(0)

        return self.timeseries

    def distance(self, prePoint, kp):
        preVal = prePoint.get_value()
        kpVal = kp.get_value()
        distance = kpVal - preVal
        return distance

    def local(self, timeSeries, prePoint):
        tempList = timeSeries.get_timeseries()

        preTime = prePoint.get_timestamp()
        preVal = prePoint.get_value()
        kpTime = self.kp.get_timestamp()
        kpVal = self.kp.get_value()
        lowerBound = preVal + self.SMIN * (kpTime - preTime)
        upperBound = preVal + self.SMAX * (kpTime - preTime)

        length = len(tempList)
        top = [0] * (length + 1)
        len_array = [0] * (length + 1)

        if length == 1:
            if lowerBound > kpVal or upperBound < kpVal:
                modify = preVal
                self.kp.set_value(modify)
            return

        topIndex = 0
        for i in range(1, length):
            tp1 = tempList[i]
            t1 = tp1.get_timestamp()
            if ((t1 - preTime) * self.SMAX) >= self.distance(prePoint, tp1) >= ((t1 - preTime) * self.SMIN):
                top[i] = -1
                len_array[i] = 1
                topIndex = i
                break

        for i in range(topIndex + 1, length):
            tp1 = tempList[i]
            tp2 = tempList[i - 1]
            t1 = tp1.get_timestamp()
            t2 = tp2.get_timestamp()

            if ((t1 - t2) * self.SMAX) >= self.distance(tp2, tp1) >= ((t1 - t2) * self.SMIN):
                if top[i - 1] == -1:
                    top[i] = i - 1
                    len_array[i - 1] += 1
                elif top[i - 1] > 0:
                    top[i] = top[i - 1]
                    len_array[top[i - 1]] += 1
            else:
                for j in range(i - 1, topIndex - 1, -1):
                    tpVal = tp1.get_value()
                    tp2 = tempList[j]
                    t2 = tp2.get_timestamp()
                    if (self.distance(tp2, tp1) > (t1 - t2) * self.SMAX or self.distance(tp2, tp1) < (
                            t1 - t2) * self.SMIN) and top[j] > 0 or j == topIndex:
                        if ((t1 - preTime) * self.SMIN) <= self.distance(prePoint, tp1) <= ((t1 - preTime) * self.SMAX):
                            top[i] = -1
                            len_array[i] = 1
                        break
                    elif (self.distance(tp2, tp1) > (t1 - t2) * self.SMAX or self.distance(tp2, tp1) < (
                            t1 - t2) * self.SMIN) and (top[j] == 0 or top[j] == -1):
                        continue
                    elif (t1 - t2) * self.SMAX >= self.distance(tp2, tp1) >= (t1 - t2) * self.SMIN:
                        if top[j] == -1:
                            top[i] = j
                            len_array[j] += 1
                        elif top[j] > 0:
                            top[i] = top[j]
                            len_array[top[j]] += 1
                        break
                    else:
                        print("error")
                        continue

        maxIndex = topIndex
        for i in range(maxIndex, length):
            if len_array[i] > len_array[maxIndex]:
                maxIndex = i
        maxPoint = tempList[maxIndex]
        maxTime = maxPoint.get_timestamp()
        maxVal = maxPoint.get_value()
        lowerBound_max = maxVal + self.SMAX * (kpTime - maxTime)
        upperBound_max = maxVal + self.SMIN * (kpTime - maxTime)

        lowerBound = max(lowerBound, lowerBound_max)
        upperBound = min(upperBound, upperBound_max)

        if upperBound < kpVal or lowerBound > kpVal:
            pre_dis = kpTime - preTime
            pre_next_dis = maxTime - preTime
            rate = pre_dis / pre_next_dis
            modify = (maxVal - preVal) * rate + preVal
            self.kp.set_value(modify)
