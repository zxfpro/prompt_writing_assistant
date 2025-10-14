""" 适配器模式 """
class Adapter():
    """
使用效果就是直接调用老接口的对象的话对方咩有这个方法,
那么就将老接口传入一个适配器,这样,老接口就有了新方法 有点像电源适配器,或者插头转换器
## 适配器模式

    """
    def __init__(self):
        self.p = """
class NewPrinter(ABC):
    def print_content(self,content):
        raise NotImplementedError
    
class Adapter(NewPrinter):
    def __init__(self, old_function):
        self.old_function = old_function
        
    def print_content(self, content):
        self.old_function.print(content)
"""
    def __repr__(self):
        return self.p