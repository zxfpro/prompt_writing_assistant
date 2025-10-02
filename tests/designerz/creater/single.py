""" 单例模式 """

class Singleton():
    # 全局只初始化一次 一个类只能存在一个实例 底层内存共享
    def __init__(self,class_name:str = "Init"):
        self.p = f"""
class {class_name}:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self,value = None):
        # 只有第一次初始化时设置值，后续的初始化调用不会更改实例的值
        if not hasattr(self, 'initialized'):
            self.initialized = True
            self.value = value

setting = {class_name}()
del {class_name}

"""
    def __repr__(self):
        return self.p
    
    def get_UML(self):
        return """
```mermaid
classDiagram
    direction LR

    class Singleton {
        - $instance: Singleton
        - Singleton()
        + $getInstance(): Singleton
    }

    class Client {
        +doSomething()
    }

    %% 关系
    %% 客户端依赖于单例类来获取实例
    Client ..> Singleton : requests instance via getInstance()

    %% 单例类自我依赖，因为 getInstance 方法会创建并返回自身的实例
    Singleton "1" --o "1" Singleton : holds
    note for Singleton "The getInstance() method creates and/or returns the single static instance."

```
"""