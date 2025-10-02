""" 桥接模式 """

class Connect():
    """
    将抽象与实现解耦，使两者可以独立地变化
    简单来说，当你的系统中有两个维度（或者说两种变化）需要独立地扩展时，桥接模式能够很好地处理这种复杂性，避免类爆炸，并保持代码的灵活性和可维护性。

    解决的问题
        类爆炸问题： 当一个类有两个或多个独立变化的维度时，如果使用继承，会导致子类的数量呈几何级数增长（例如，颜色和形状的组合）。
        紧耦合问题： 抽象和实现紧密耦合，导致修改一方会影响另一方。
        灵活性不足： 难以在运行时改变对象的实现。

    桥接模式可以防止维度爆炸

    若没有桥接模式 （传统继承方式）
    这样，每增加一个形状或一个操作系统，都会导致类数量的爆炸式增长（M * N 个类）。
    Shape (抽象类)
    - Circle (圆形)
        - WindowsCircle
        - LinuxCircle
        - MacOSCircle
    - Rectangle (矩形)
        - WindowsRectangle
        - LinuxRectangle
        - MacOSRectangle


    使用桥接模式：
    我们可以将“形状”作为抽象层次，将“绘制API（操作系统）”作为实现层次。
        抽象层次 (Shape):
            Shape (抽象类)
                Circle (具体形状)
                Rectangle (具体形状)
        实现层次 (DrawingAPI):
            DrawingAPI (接口/抽象类)
                WindowsDrawingAPI (具体实现)
                LinuxDrawingAPI (具体实现)
                MacOSDrawingAPI (具体实现)
    """
    def __init__(self):
        self.p = '''
import abc

# 1. 实现类接口 (Implementor)
class DrawingAPI(abc.ABC):
    """
    定义绘图功能的接口，例如在不同操作系统上的具体绘图操作。
    """
    @abc.abstractmethod
    def draw_circle(self, x: int, y: int, radius: int):
        pass

    @abc.abstractmethod
    def draw_rectangle(self, x1: int, y1: int, x2: int, y2: int):
        pass

# 2. 具体实现类 (ConcreteImplementor)
class WindowsDrawingAPI(DrawingAPI):
    def draw_circle(self, x: int, y: int, radius: int):
        print(f"在 Windows 系统上绘制圆形: 中心({x},{y}), 半径{radius}")

    def draw_rectangle(self, x1: int, y1: int, x2: int, y2: int):
        print(f"在 Windows 系统上绘制矩形: 左上({x1},{y1}), 右下({x2},{y2})")

class LinuxDrawingAPI(DrawingAPI):
    def draw_circle(self, x: int, y: int, radius: int):
        print(f"在 Linux 系统上绘制圆形: 中心({x},{y}), 半径{radius}")

    def draw_rectangle(self, x1: int, y1: int, x2: int, y2: int):
        print(f"在 Linux 系统上绘制矩形: 左上({x1},{y1}), 右下({x2},{y2})")

class MacOSDrawingAPI(DrawingAPI):
    def draw_circle(self, x: int, y: int, radius: int):
        print(f"在 MacOS 系统上绘制圆形: 中心({x},{y}), 半径{radius}")

    def draw_rectangle(self, x1: int, y1: int, x2: int, y2: int):
        print(f"在 MacOS 系统上绘制矩形: 左上({x1},{y1}), 右下({x2},{y2})")

# 3. 抽象类 (Abstraction)
class Shape(abc.ABC):
    """
    定义形状的抽象类，持有绘图API的引用。
    """
    def __init__(self, drawing_api: DrawingAPI):
        self._drawing_api = drawing_api

    @abc.abstractmethod
    def draw(self):
        """抽象的绘制方法"""
        pass

    @abc.abstractmethod
    def resize(self, factor: float):
        """抽象的调整大小方法"""
        pass

# 4. 修正抽象类 (RefinedAbstraction)
class Circle(Shape):
    def __init__(self, x: int, y: int, radius: int, drawing_api: DrawingAPI):
        super().__init__(drawing_api)
        self._x = x
        self._y = y
        self._radius = radius

    def draw(self):
        self._drawing_api.draw_circle(self._x, self._y, self._radius)

    def resize(self, factor: float):
        self._radius *= factor
        print(f"圆形半径调整为: {self._radius}")

class Rectangle(Shape):
    def __init__(self, x1: int, y1: int, x2: int, y2: int, drawing_api: DrawingAPI):
        super().__init__(drawing_api)
        self._x1 = x1
        self._y1 = y1
        self._x2 = x2
        self._y2 = y2

    def draw(self):
        self._drawing_api.draw_rectangle(self._x1, self._y1, self._x2, self._y2)

    def resize(self, factor: float):
        # 简单示例，实际可能更复杂
        width = (self._x2 - self._x1) * factor
        height = (self._y2 - self._y1) * factor
        self._x2 = self._x1 + width
        self._y2 = self._y1 + height
        print(f"矩形调整为: ({self._x1},{self._y1}) 到 ({self._x2},{self._y2})")

# 客户端代码
if __name__ == "__main__":
    # 在 Windows 上绘制
    windows_api = WindowsDrawingAPI()
    circle_on_windows = Circle(10, 20, 5, windows_api)
    rect_on_windows = Rectangle(5, 5, 15, 10, windows_api)

    print("--- 在 Windows 上绘制 ---")
    circle_on_windows.draw()
    rect_on_windows.draw()
    circle_on_windows.resize(1.5)
    circle_on_windows.draw()

    print("\n--- 在 Linux 上绘制 ---")
    # 在 Linux 上绘制
    linux_api = LinuxDrawingAPI()
    circle_on_linux = Circle(30, 40, 8, linux_api)
    rect_on_linux = Rectangle(20, 20, 30, 25, linux_api)

    circle_on_linux.draw()
    rect_on_linux.draw()

    print("\n--- 在 MacOS 上绘制 ---")
    # 在 MacOS 上绘制
    macos_api = MacOSDrawingAPI()
    circle_on_macos = Circle(50, 60, 10, macos_api)
    circle_on_macos.draw()
'''
    def __repr__(self):
        return self.p
