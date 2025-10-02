""" 代理模式
代理模式的分类
远程代理（Remote Proxy）：控制对远程对象的访问。
虚拟代理（Virtual Proxy）：控制对资源消耗大的对象的访问，可以延迟对象的创建。
保护代理（Protection Proxy）：控制对原始对象的访问权限。
智能代理（Smart Proxy）：在访问对象时执行一些附加操作，例如记录访问日志、引用计数等。

简单来说就是包一层, 这一层中来控制 目标对象的使用
"""

class Agent():
    """
    """
    def __init__(self):
        self.p = """
# 抽象主题
class Image(ABC):
    @abstractmethod
    def display(self):
        pass

# 真实主题
class RealImage(Image):
    def __init__(self, filename):
        self.filename = filename
        self.load_from_disk()

    def load_from_disk(self):
        print(f"Loading {self.filename}")

    def display(self):
        print(f"Displaying {self.filename}")

# 代理
class ProxyImage(Image):
    def __init__(self, filename):
        self.filename = filename
        self.real_image = None

    def display(self):
        if self.real_image is None:
            self.real_image = RealImage(self.filename)
        self.real_image.display()




# 使用代理模式
proxy_image = ProxyImage("test_image.jpg")


# 其实就是延迟加载的逻辑
proxy_image.display()


"""
    def __repr__(self):
        return self.p

