""" origin 原型/克隆模式 """

class Origin():
    # 具体原型 当创建对象的成本较高时, 初始化复杂
    def __init__(self,class_name:str = "Origin"):
        self.p = f"""
import copy
class {class_name}():
    def __init__(self, id, radius):
        self.id = id
        self.radius = radius
        #todo init

    def __str__(self):
        return f"Circle [ID=, Radius=]"
    
    def clone(self):
        # 原型模式
        return copy.deepcopy(self)
"""
    def __repr__(self):
        return self.p
