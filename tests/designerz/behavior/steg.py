""" 策略模式 """

class Strategy():
    """

    策略模式定义一系列算法，并将每个算法封装起来，使它们可以相互替换。策略模式使得算法可以在不影响客户端的情况下发生变化。

    应用场景：排序算法、加密算法、日志策略等。
    策略模式做的情况是,可以在不需要关闭服务的情况下,动态的变换策略
    和工厂模式有点像
    """
    def __init__(self):
        self.p = """

from abc import ABC, abstractmethod
from typing import List

# 策略接口
class SortStrategy(ABC):
    @abstractmethod
    def sort(self, data: List[int]) -> List[int]:
        pass

# 具体策略：快速排序
class QuickSortStrategy(SortStrategy):
    def sort(self, data: List[int]) -> List[int]:
        print("Using QuickSort")
        return sorted(data)  # 这里使用Python内置排序作为简化的快速排序实现

# 具体策略：插入排序
class InsertionSortStrategy(SortStrategy):
    def sort(self, data: List[int]) -> List[int]:
        print("Using InsertionSort")
        for i in range(1, len(data)):
            key = data[i]
            j = i - 1
            while j >= 0 and key < data[j]:
                data[j + 1] = data[j]
                j -= 1
            data[j + 1] = key
        return data

# 上下文
class SortContext:
    def __init__(self, strategy: SortStrategy):
        self._strategy = strategy

    def set_strategy(self, strategy: SortStrategy):
        self._strategy = strategy

    def sort(self, data: List[int]) -> List[int]:
        return self._strategy.sort(data)

# 客户端代码
if __name__ == "__main__":
    data = [5, 3, 8, 4, 2]

    context = SortContext(QuickSortStrategy())
    print(context.sort(data))

    context.set_strategy(InsertionSortStrategy())
    print(context.sort(data))


"""
    def __repr__(self):
        return self.p