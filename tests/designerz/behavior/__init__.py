# 　　其中有分为：
# 　　算法封装：模板方法、策略、命令模式
# 　　对象去耦：中介、观察者模式
# 　　抽象集合：迭代器模式
# 　　行为扩展：访问者、责任链模式
# 　　对象状态：状态模式
# 　　解释器模式



# 　　行为扩展：责任链模式

from .watcher import Watcher
from .steg import Strategy
from .state import State
from .order import Order
from .mediator import Mediator
from .fwz import Visitor
from .explan import Explain




'''
=

### 文件内容

1. **`adapter.py`**:
```python
# 做适配器的文件

# test_111/adapter.py
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, TextIO
from src.test_111.funs import RealDataReaderImpl, RealDataAnalyzerImpl, RealReportGeneratorImpl
from src.test_111.mock import MockDataReader, MockDataAnalyzer, MockReportGenerator

# --- 1. 定义接口 (Interfaces / Ports) ---
# ▼▼▼ MODIFICATION / ADDITION: New interfaces for CSV analysis capabilities ▼▼▼
class IDataReader(ABC):
    """
    数据读取器接口：定义从源读取数据的能力。
    """
    @abstractmethod
    def read_csv(self, source: str) -> List[Dict]:
        """从指定源读取CSV数据并返回字典列表。"""
        pass

class IDataAnalyzer(ABC):
    """
    数据分析器接口：定义对数据进行统计分析的能力。
    """
    @abstractmethod
    def analyze_column_stats(self, data: List[Dict], column_name: str) -> Any: # 返回IAnalysisStrategy
        """分析指定列的统计数据（例如：总和、平均值）。"""
        pass

class IReportGenerator(ABC):
    """
    报告生成器接口：定义根据分析结果生成报告的能力。
    """
    @abstractmethod
    def generate_report(self, analysis_results: Dict[str, Any]) -> str:
        """根据分析结果生成可读的报告字符串。"""
        pass

# --- 2. AdapterType 枚举 (用于工厂模式) ---
class AdapterType(Enum):
    MOCK = 'MOCK'
    REAL = 'REAL'

# --- 3. 适配器工厂 (AdapterFactory) ---
# ▼▼▼ YOUR FACTORY TEMPLATE IS USED HERE (Adapted to our needs) ▼▼▼
class AdapterFactory:
    @classmethod
    def new_data_reader(cls, type: AdapterType, **kwargs) -> IDataReader:
        if type == AdapterType.MOCK:
            return MockDataReader()
        elif type == AdapterType.REAL:
            return RealDataReader()
        else:
            raise ValueError(f"Unknown AdapterType for IDataReader: {type}")

    @classmethod
    def new_data_analyzer(cls, type: AdapterType, **kwargs) -> IDataAnalyzer:
        if type == AdapterType.MOCK:
            return MockDataAnalyzer()
        elif type == AdapterType.REAL:
            return RealDataAnalyzer()
        else:
            raise ValueError(f"Unknown AdapterType for IDataAnalyzer: {type}")

    @classmethod
    def new_report_generator(cls, type: AdapterType, **kwargs) -> IReportGenerator:
        if type == AdapterType.MOCK:
            return MockReportGenerator()
        elif type == AdapterType.REAL:
            return RealReportGenerator()
        else:
            raise ValueError(f"Unknown AdapterType for IReportGenerator: {type}")

# --- 4. 适配器具体实现 (Mock & Real) ---
# Real 适配器 (继承自funs.py中的实现)
class RealDataReader(IDataReader, RealDataReaderImpl):
    def __init__(self):
        super().__init__()

from src.test_111.core import IAnalysisStrategy # 导入IAnalysisStrategy

class RealDataAnalyzer(IDataAnalyzer):
    def __init__(self):
        # RealDataAnalyzerImpl 实际上就是我们需要的分析策略
        self._strategy = RealDataAnalyzerImpl()

    def analyze_column_stats(self, data: List[Dict], column_name: str) -> IAnalysisStrategy:
        # 这里不再执行分析，而是返回一个IAnalysisStrategy的实例
        return self._strategy

class MockDataAnalyzer(IDataAnalyzer):
    def __init__(self):
        # MockDataAnalyzerImpl 实际上就是我们需要的分析策略
        self._strategy = MockDataAnalyzerImpl()

    def analyze_column_stats(self, data: List[Dict], column_name: str) -> IAnalysisStrategy:
        # 这里不再执行分析，而是返回一个IAnalysisStrategy的实例
        return self._strategy

class RealReportGenerator(IReportGenerator, RealReportGeneratorImpl):
    def __init__(self):
        super().__init__()
```

2. **`core.py`**:
```python
# 编写核心使用代码的部分

# test_111/funs.py
from typing import Any, Dict, List
from abc import ABC, abstractmethod
from src.test_111.adapter import IDataReader, IDataAnalyzer, IReportGenerator, AdapterFactory, AdapterType

# 定义分析策略接口
class IAnalysisStrategy(ABC):
    """
    分析策略接口：定义不同的数据分析算法。
    """
    @abstractmethod
    def analyze(self, data: List[Dict], column_name: str) -> Dict[str, Any]:
        """执行具体的分析算法。"""
        pass

# 这里应该只包含设计模式相关的内容，而不包含具体的逻辑实现
# 具体的逻辑实现都应该从adapter导入
# 例如，你可以定义一个使用这些接口的Facade或者Service类
class CSVProcessor:
    def __init__(self, data_reader: IDataReader, data_analyzer: IDataAnalyzer, report_generator: IReportGenerator):
        self.data_reader = data_reader
        self.data_analyzer = data_analyzer
        self.report_generator = report_generator

    def process_csv_data(self, source: str, column_name: str) -> str:
        data = self.data_reader.read_csv(source)
        # 从data_analyzer获取IAnalysisStrategy的实例
        analysis_strategy: IAnalysisStrategy = self.data_analyzer.analyze_column_stats(data, column_name)
        analysis_results = analysis_strategy.analyze(data, column_name)
        report = self.report_generator.generate_report(analysis_results)
        return report

# 示例如何使用AdapterFactory来获取具体的实现
def get_real_csv_processor():
    reader = AdapterFactory.new_data_reader(AdapterType.REAL)
    analyzer = AdapterFactory.new_data_analyzer(AdapterType.REAL)
    generator = AdapterFactory.new_report_generator(AdapterType.REAL)
    return CSVProcessor(reader, analyzer, generator)

def get_mock_csv_processor():
    reader = AdapterFactory.new_data_reader(AdapterType.MOCK)
    analyzer = AdapterFactory.new_data_analyzer(AdapterType.MOCK)
    generator = AdapterFactory.new_report_generator(AdapterType.MOCK)
    return CSVProcessor(reader, analyzer, generator)
```

3. **`funs.py`**:
```python
# test_111/funs.py
from typing import Any, Dict, List
import csv # 用于真实读取CSV
import io # 用于真实读取CSV

# ▼▼▼ MODIFICATION / ADDITION: Real implementations of capabilities ▼▼▼
class RealDataReaderImpl:
    def __init__(self):
        print("RealDataReaderImpl: Initialized (reads actual CSV data).")

    def read_csv(self, source: str) -> List[Dict]:
        """从指定路径读取CSV数据并返回字典列表。"""
        print(f"RealDataReaderImpl: Reading actual CSV from {source}.")
        # 实际逻辑：从文件中读取
        # 为了演示，我们假设source是一个包含CSV数据的字符串或者一个文件路径
        # 这里为了不依赖实际文件，假设source就是CSV内容
        if source.endswith(".csv"):
             with open(source, 'r', encoding='utf-8') as f:
                 reader = csv.DictReader(f)
                 return list(reader)
        else: # 假设source直接是CSV内容的字符串
            f = io.StringIO(source)
            reader = csv.DictReader(f)
            return list(reader)

class RealDataAnalyzerImpl:
    def __init__(self):
        print("RealDataAnalyzerImpl: Initialized (performs actual analysis).")

    def analyze_column_stats(self, data: List[Dict], column_name: str) -> Dict[str, Any]:
        """对指定列进行实际的统计分析。"""
        print(f"RealDataAnalyzerImpl: Analyzing actual column '{column_name}'.")
        values = []
        for row in data:
            if column_name in row:
                try:
                    values.append(float(row[column_name])) # 尝试转换为数字进行计算
                except (ValueError, TypeError):
                    continue # 忽略非数字值

        if not values:
            return {"sum": 0.0, "avg": 0.0, "count": 0}

        total_sum = sum(values)
        return {"sum": total_sum, "avg": total_sum / len(values), "count": len(values)}

class RealReportGeneratorImpl:
    def __init__(self):
        print("RealReportGeneratorImpl: Initialized (generates actual report).")

    def generate_report(self, analysis_results: Dict[str, Any]) -> str:
        """根据分析结果生成可读的报告字符串。"""
        print(f"RealReportGeneratorImpl: Generating actual report for {analysis_results}.")
        report_str = f"--- CSV Analysis Report ---\n"
        for key, value in analysis_results.items():
            report_str += f"{key}: {value}\n"
        report_str += "---------------------------"
        return report_str
```





你是一个专业的python架构师, 请使用以下设计模式给定模版, 并针对具体问题灵活使用


# 模式模版

## 适配器模式
class NewPrinter(ABC):
    def print_content(self,content):
        raise NotImplementedError
    
class Adapter(NewPrinter):
    def __init__(self, old_function):
        self.old_function = old_function
        
    def print_content(self, content):
        self.old_function.print(content)

## 代理模式
class Image(ABC):
    @abstractmethod
    def display(self):
        pass

class RealImage(Image):
    def __init__(self, filename):
        self.filename = filename
        self.load_from_disk()

    def load_from_disk(self):
        print(f"Loading {self.filename}")

    def display(self):
        print(f"Displaying {self.filename}")

class ProxyImage(Image):
    def __init__(self, filename):
        self.filename = filename
        self.real_image = None

    def display(self):
        if self.real_image is None:
            self.real_image = RealImage(self.filename)
        self.real_image.display()

proxy_image = ProxyImage("test_image.jpg")

proxy_image.display()


## 外观模式
class TV:
    def on(self):
        print("TV is on")

    def off(self):
        print("TV is off")

class SoundSystem:
    def on(self):
        print("Sound system is on")

    def off(self):
        print("Sound system is off")

    def set_volume(self, volume):
        print(f"Sound system volume set to {volume}")

class DVDPlayer:
    def on(self):
        print("DVD player is on")

    def off(self):
        print("DVD player is off")

    def play(self, movie):
        print(f"Playing movie: {movie}")


class HomeTheaterFacade:
    def __init__(self, tv: TV, sound_system: SoundSystem, dvd_player: DVDPlayer):
        self._tv = tv
        self._sound_system = sound_system
        self._dvd_player = dvd_player

    def watch_movie(self, movie):
        print("Get ready to watch a movie...")
        self._tv.on()
        self._sound_system.on()
        self._sound_system.set_volume(20)
        self._dvd_player.on()
        self._dvd_player.play(movie)

    def end_movie(self):
        print("Shutting down the home theater...")
        self._tv.off()
        self._sound_system.off()
        self._dvd_player.off()

tv = TV()
sound_system = SoundSystem()
dvd_player = DVDPlayer()

home_theater = HomeTheaterFacade(tv, sound_system, dvd_player)
home_theater.watch_movie("Inception")
home_theater.end_movie()

## 组合模式
from abc import ABC, abstractmethod

class FileSystemComponent(ABC):
    @abstractmethod
    def operation(self):
        pass

class File(FileSystemComponent):
    def __init__(self, name):
        self.name = name

    def operation(self):
        return f"File: {self.name}"

class Directory(FileSystemComponent):
    def __init__(self, name):
        self.name = name
        self.children = []

    def add(self, component: FileSystemComponent):
        self.children.append(component)

    def remove(self, component: FileSystemComponent):
        self.children.remove(component)

    def operation(self):
        results = [f"Directory: {self.name}"]
        for child in self.children:
            results.append(child.operation())
        return "\n".join(results)

## 工厂模式
from enum import Enum
from typing import List, Any

class {EnumClassName}(Enum):
    {Option1} = '{Option1}'
    {Option2} = '{Option2}'
    # 添加更多选项

class {FactoryClassName}:
    def __new__(cls, type: {EnumClassName}) -> Any:
        assert type.value in [i.value for i in {EnumClassName}]
        instance = None

        if type.value == '{Option1}':

            # instance = SomeClass(param1=value1, param2=value2)
            pass

        elif type.value == '{Option2}':

            # instance = AnotherClass(param1=value1, param2=value2)
            pass


        else:
            raise Exception('Unknown type')

        return instance

        
## 桥接模式
import abc

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
        width = (self._x2 - self._x1) * factor
        height = (self._y2 - self._y1) * factor
        self._x2 = self._x1 + width
        self._y2 = self._y1 + height
        print(f"矩形调整为: ({self._x1},{self._y1}) 到 ({self._x2},{self._y2})")

if __name__ == "__main__":
    windows_api = WindowsDrawingAPI()
    circle_on_windows = Circle(10, 20, 5, windows_api)
    rect_on_windows = Rectangle(5, 5, 15, 10, windows_api)

    print("--- 在 Windows 上绘制 ---")
    circle_on_windows.draw()
    rect_on_windows.draw()
    circle_on_windows.resize(1.5)
    circle_on_windows.draw()

    print("\n--- 在 Linux 上绘制 ---")
    linux_api = LinuxDrawingAPI()
    circle_on_linux = Circle(30, 40, 8, linux_api)
    rect_on_linux = Rectangle(20, 20, 30, 25, linux_api)

    circle_on_linux.draw()
    rect_on_linux.draw()

    print("\n--- 在 MacOS 上绘制 ---")
    macos_api = MacOSDrawingAPI()
    circle_on_macos = Circle(50, 60, 10, macos_api)
    circle_on_macos.draw()

    

## 建造者模式

class ABCHouse():
    pass


class House(ABCHouse):
    def __init__(self):
        self.foundation = None
        self.structure = None
        self.roof = None
        self.interior = None

    def __str__(self):
        return f"House with self.foundation, self.structure, self.roof, and self.interior"

# 生成器接口
class HouseBuilder:
    def build_foundation(self):
        pass

    def build_structure(self):
        pass

    def build_roof(self):
        pass

    def build_interior(self):
        pass

    def get_house(self):
        pass

# 具体生成器
class ConcreteHouseBuilder(HouseBuilder):
    def __init__(self):
        self.house = House() # ABCHouse

    def build_foundation(self):
        self.house.foundation = "Concrete foundation"

    def build_structure(self):
        self.house.structure = "Wood and brick structure"

    def build_roof(self):
        self.house.roof = "Shingle roof"

    def build_interior(self):
        self.house.interior = "Modern interior"

    def get_house(self):
        return self.house

# 指挥者
class Director:
    def __init__(self, builder: HouseBuilder):
        self.builder = builder

    def construct_house(self):
        self.builder.build_foundation()
        self.builder.build_structure()
        self.builder.build_roof()
        self.builder.build_interior()
        return self.builder.get_house()

{执行协议等待指令：简单任务用自适应身份叠加输入处理输出结构基元执行，复杂任务分拆成简单任务交给基元构成链条执行，默认不展示输入处理输出细节但用户可以要求展示。}
'''
