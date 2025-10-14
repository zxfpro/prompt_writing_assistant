""" 享元模式 """

class ShareYuan():
    """
    """
    def __init__(self,decorator):
        self.p = """
import abc

# 1. 抽象享元角色 (Flyweight)
class Chessman(abc.ABC):
    @abc.abstractmethod
    def display(self, x: int, y: int):
        "显示棋子，x, y 是外部状态"
        pass

# 2. 具体享元角色 (ConcreteFlyweight)
class BlackChessman(Chessman):
    def __init__(self):
        self._color = "黑色" # 内部状态
        print(f"创建了一个 {self._color} 棋子实例。")

    def display(self, x: int, y: int):
        print(f"在 ({x}, {y}) 位置显示 {self._color} 棋子。")

class WhiteChessman(Chessman):
    def __init__(self):
        self._color = "白色" # 内部状态
        print(f"创建了一个 {self._color} 棋子实例。")

    def display(self, x: int, y: int):
        print(f"在 ({x}, {y}) 位置显示 {self._color} 棋子。")

# 3. 享元工厂角色 (FlyweightFactory)
class ChessmanFactory:
    _chessmen = {} # 享元池

    def get_chessman(self, color: str) -> Chessman:
        if color not in self._chessmen:
            if color == "黑色":
                self._chessmen[color] = BlackChessman()
            elif color == "白色":
                self._chessmen[color] = WhiteChessman()
            else:
                raise ValueError("不支持的棋子颜色！")
        return self._chessmen[color]

    def get_chessman_count(self):
        return len(self._chessmen)

# 4. 客户端角色 (Client)
if __name__ == "__main__":
    factory = ChessmanFactory()

    # 创建并显示多个黑棋和白棋，但实际只创建了两个享元实例
    print("--- 放置棋子 ---")
    black1 = factory.get_chessman("黑色")
    black1.display(1, 1)

    white1 = factory.get_chessman("白色")
    white1.display(2, 2)

    black2 = factory.get_chessman("黑色") # 从享元池获取已存在的实例
    black2.display(3, 3)

    white2 = factory.get_chessman("白色") # 从享元池获取已存在的实例
    white2.display(4, 4)

    black3 = factory.get_chessman("黑色")
    black3.display(5, 5)

    print("\n--- 检查享元实例数量 ---")
    print(f"实际创建的享元实例数量: {factory.get_chessman_count()}") # 应该只有2个

    # 尝试访问同一个享元对象
    print("\n--- 验证享元对象是否相同 ---")
    print(f"black1 和 black2 是同一个对象: {black1 is black2}")
    print(f"white1 和 white2 是同一个对象: {white1 is white2}")

    # 假设游戏中有100个黑棋和100个白棋
    # 如果没有享元模式，需要创建200个对象
    # 有了享元模式，只需要创建2个对象，然后重复使用它们。
"""
    def __repr__(self):
        return self.p