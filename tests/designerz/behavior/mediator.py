""" 中介者模式（Mediator） """



class Mediator():

    """
## 中介者模式（Mediator）

简单来说就是群聊, 广播
也可以做小群 也可以做1对1 1对多 多对1


    """
    def __init__(self):
        self.p = """
from abc import ABC, abstractmethod

# 中介者接口
class Mediator(ABC):
    @abstractmethod
    def send(self, message: str, colleague: 'Colleague'):
        pass

# 具体中介者
class ConcreteMediator(Mediator):
    def __init__(self):
        self._colleagues = []

    def add_colleague(self, colleague: 'Colleague'):
        self._colleagues.append(colleague)
        colleague.set_mediator(self)

    def send(self, message: str, colleague: 'Colleague'):
        for c in self._colleagues:
            if c != colleague:
                c.receive(message)

# 同事对象接口
class Colleague(ABC):
    def __init__(self):
        self._mediator = None

    def set_mediator(self, mediator: Mediator):
        self._mediator = mediator

    @abstractmethod
    def send(self, message: str):
        pass

    @abstractmethod
    def receive(self, message: str):
        pass

# 具体同事对象
class ConcreteColleague(Colleague):
    def __init__(self, name):
        super().__init__()
        self._name = name

    def send(self, message: str):
        print(f"{self._name} sends message: {message}")
        self._mediator.send(message, self)

    def receive(self, message: str):
        print(f"{self._name} receives message: {message}")

# 客户端代码
if __name__ == "__main__":
    # 创建中介者
    mediator = ConcreteMediator()

    # 创建同事对象
    colleague1 = ConcreteColleague("User1")
    colleague2 = ConcreteColleague("User2")
    colleague3 = ConcreteColleague("User3")

    # 将同事对象添加到中介者
    mediator.add_colleague(colleague1)
    mediator.add_colleague(colleague2)
    mediator.add_colleague(colleague3)

    # 发送消息
    colleague1.send("Hello everyone!")

"""
    def __repr__(self):
        return self.p
    