""" 组合模式 """

class Combo():
    """
# 理解了,是一种高级复杂的模式,类似于文件系统 文件是叶子 文件夹是节点,节点可以包括节点 节点也可以包括叶子,也会产生多级嵌套 
# 可以使用这种
# 使用组合模式
file1 = File("file1.txt")
file2 = File("file2.txt")

directory1 = Directory("dir1")
directory1.add(file1)

directory2 = Directory("dir2")
directory2.add(file2)
directory2.add(directory1)

print(directory2.operation())

    """
    def __init__(self):
        self.p = """
from abc import ABC, abstractmethod

# Component 接口
class FileSystemComponent(ABC):
    @abstractmethod
    def operation(self):
        pass

# Leaf 类
class File(FileSystemComponent):
    def __init__(self, name):
        self.name = name

    def operation(self):
        return f"File: {self.name}"

# Composite 类
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

"""    
    def __repr__(self):
        return self.p