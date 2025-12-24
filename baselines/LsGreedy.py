import heapq

from tqdm import tqdm


class LsGreedy:
    def __init__(self, timeseries):
        self.eps = 1e-12
        self.timeseries = timeseries
        self.AllList = timeseries.get_timeseries()
        self.Size = len(self.AllList)
        self.time = [tp.get_timestamp() for tp in self.AllList]
        self.original = [tp.get_value() for tp in self.AllList]
        self.center = 0.0
        self.sigma = 0.0
        self.setParameters()

    @staticmethod
    def speed(origin, time):
        n = len(origin)
        speed = []
        for i in range(n - 1):
            speed_val = (origin[i + 1] - origin[i]) / (time[i + 1] - time[i])
            speed.append(speed_val)
        return speed

    @staticmethod
    def variation(origin):
        n = len(origin)
        var = []
        for i in range(n - 1):
            var.append(origin[i + 1] - origin[i])
        return var

    @staticmethod
    def median(lst):
        sorted_lst = sorted(lst)
        n = len(sorted_lst)
        if n % 2 == 0:
            return (sorted_lst[n // 2 - 1] + sorted_lst[n // 2]) / 2
        else:
            return sorted_lst[(n - 1) // 2]

    @staticmethod
    def mad(value):
        mid = LsGreedy.median(value)
        d = [abs(v - mid) for v in value]
        return 1.4826 * LsGreedy.median(d)

    def setParameters(self):
        speed = self.speed(self.original, self.time)
        speedchange = self.variation(speed)
        self.sigma = self.mad(speedchange)

    def repair(self):
        class RepairNode:
            def __init__(self, index, parent):
                self.parent = parent
                self.index = index
                prePoint = parent.AllList[index - 1]
                iPoint = parent.AllList[index]
                nextPoint = parent.AllList[index + 1]

                v1 = (nextPoint.value - iPoint.value) / (parent.time[index + 1] - parent.time[index])
                v2 = (iPoint.value - prePoint.value) / (parent.time[index] - parent.time[index - 1])
                self.u = v1 - v2
                self.priority = abs(self.u - parent.center)

            def modify(self):
                parent = self.parent
                temp = 0.0
                if parent.sigma < parent.eps:
                    temp = abs(self.u - parent.center)
                else:
                    temp = max(parent.sigma, abs(self.u - parent.center) / 3)

                delta = (parent.time[self.index + 1] - parent.time[self.index]) * \
                        (parent.time[self.index] - parent.time[self.index - 1]) / \
                        (parent.time[self.index + 1] - parent.time[self.index - 1])
                delta *= temp

                iPoint = parent.AllList[self.index]
                if self.u > parent.center:
                    iPoint.value += delta
                else:
                    iPoint.value -= delta

            def __lt__(self, other):
                return self.priority > other.priority

        heap = []
        table = [None] * self.Size

        for i in range(1, self.Size - 1):
            node = RepairNode(i, self)
            table[i] = node
            if abs(node.u - self.center) > 3 * self.sigma:
                heapq.heappush(heap, node)

        max_iterations = 5000
        iteration = 0
        for _ in tqdm(range(max_iterations), desc="Repairing", leave=False):
            if not heap:
                break

            top = heap[0]
            if abs(top.u - self.center) < max(self.eps, 3 * self.sigma):
                break

            top = heapq.heappop(heap)
            top.modify()

            start = max(1, top.index - 1)
            end = min(self.Size - 2, top.index + 1)

            for i in range(start, end + 1):
                if table[i] is not None and table[i] in heap:
                    heap.remove(table[i])
                    heapq.heapify(heap)

                new_node = RepairNode(i, self)
                table[i] = new_node
                if abs(new_node.u - self.center) > 3 * self.sigma:
                    heapq.heappush(heap, new_node)

            iteration += 1

        return self.timeseries