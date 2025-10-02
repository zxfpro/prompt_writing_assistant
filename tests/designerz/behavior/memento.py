""" 备忘录模式 """


# @title 备忘录模式
#@markdown Please enter your details below:
# 主要用于实现状态回滚功能,回滚操作
class Memento:
    def __init__(self, state: str):
        self._state = state

    def get_state(self) -> str:
        return self._state

    def set_state(self, state: str):
        self._state = state

class Originator:
    def __init__(self):
        self._state = ""

    def set_state(self, state: str):
        self._state = state

    def get_state(self) -> str:
        return self._state

    def create_memento(self) -> Memento:
        return Memento(self._state)

    def set_memento(self, memento: Memento):
        self._state = memento.get_state()

class Caretaker:
    def __init__(self):
        self._memento = None

    def save(self, memento: Memento):
        self._memento = memento

    def retrieve(self) -> Memento:
        return self._memento

# 客户端代码
if __name__ == "__main__":
    originator = Originator()
    caretaker = Caretaker()

    # 设置和保存状态
    originator.set_state("State 1")
    print(f"Current State: {originator.get_state()}")
    caretaker.save(originator.create_memento())

    # 修改状态
    originator.set_state("State 2")
    print(f"Current State: {originator.get_state()}")

    # 恢复到先前的状态
    originator.set_memento(caretaker.retrieve())
    print(f"Restored State: {originator.get_state()}")
