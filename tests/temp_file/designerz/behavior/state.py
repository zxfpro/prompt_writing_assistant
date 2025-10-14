""" 状态模式 """


class State():
    """
    状态模式
    """
    def __init__(self):
        self.p = """
from abc import ABC, abstractmethod

# 抽象状态
class State(ABC):
    @abstractmethod
    def handle(self, context: 'Context'):
        pass

# 具体状态：灯打开
class OnState(State):
    def handle(self, context: 'Context'):
        print("Light is already on. Turning it off.")
        context.set_state(OffState())

# 具体状态：灯关闭
class OffState(State):
    def handle(self, context: 'Context'):
        print("Light is off. Turning it on.")
        context.set_state(OnState())

# 上下文
class Context:
    def __init__(self, state: State):
        self._state = state

    def request(self):
        self._state.handle(self)

    def set_state(self, state: State):
        self._state = state

# 客户端代码
if __name__ == "__main__":
    # 创建上下文和初始状态
    context = Context(OffState())

    # 改变状态
    context.request()
    context.request()
    context.request()

"""
    def __repr__(self):
        return self.p



