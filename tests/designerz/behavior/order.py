""" 命令模式  """
class Order():
    """
## 命令模式

命令模式（Command Pattern）是一种行为型设计模式，它将请求封装为对象，使得可以使用不同的请求、队列或日志来参数化其他对象。命令模式还支持撤销操作。
解耦请求发送者和接收者：请求发送者只需知道命令对象，而不需要知道接收者是谁或如何处理请求。
支持撤销和恢复操作：通过存储命令对象，可以轻松实现撤销和恢复功能。
支持命令队列：命令对象可以排队执行，使得系统可以支持请求的日志记录和事务管理。
增加新的命令容易：添加新的命令只需实现命令接口，不需要修改现有的代码。

    """
    def __init__(self):
        self.p = """

from abc import ABC, abstractmethod

# 命令接口
class Command(ABC):
    @abstractmethod
    def execute(self):
        pass

# 接收者
class Light:
    def on(self):
        print("Light is ON")
    
    def off(self):
        print("Light is OFF")

# 具体命令 - 打开灯
class LightOnCommand(Command):
    def __init__(self, light: Light):
        self.light = light
    
    def execute(self):
        self.light.on()

# 具体命令 - 关闭灯
class LightOffCommand(Command):
    def __init__(self, light: Light):
        self.light = light
    
    def execute(self):
        self.light.off()

# 调用者
class RemoteControl:
    def __init__(self):
        self.command = None
    
    def set_command(self, command: Command):
        self.command = command
    
    def press_button(self):
        if self.command:
            self.command.execute()

# 客户端代码
if __name__ == "__main__":
    # 创建接收者
    light = Light()

    # 创建具体命令
    light_on = LightOnCommand(light)
    light_off = LightOffCommand(light)

    # 创建调用者
    remote = RemoteControl()

    # 设置命令并执行
    remote.set_command(light_on)
    remote.press_button()  # 输出：Light is ON

    remote.set_command(light_off)
    remote.press_button()  # 输出：Light is OFF

"""
    def __repr__(self):
        return self.p
    



