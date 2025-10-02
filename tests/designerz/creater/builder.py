""" build 建造者模式"""

class Builder():
    """ 目的是将构建过程与表示分开 """
    def __init__(self):
        """
        特别适合流程, 工作流
        """
        self.p = f"""
# 产品

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

"""

    def __repr__(self):
        return self.p