""" 访问者模式 """

class Visitor():
    def __init__(self):
        self.p = """
import math
# 我们来写一个简单的访问者模式 demo。这个 demo 将模拟一个图形结构，包含圆形和方形，并且我们将使用访问者模式来计算它们的面积和周长。
# 场景： 有一个图形的集合，包含圆形和方形。我们需要计算这个集合中所有图形的总面积和总周长。
# 使用访问者模式的优势： 如果将来需要添加新的图形类型（例如，三角形）或者新的计算方式（例如，计算重心），我们可以很容易地扩展，而无需修改原有的图形类。
# 1. Element 接口
class Element:
    def accept(self, visitor):
        pass

# 2. 具体 Element 类
class Circle(Element):
    def __init__(self, radius):
        self.radius = radius

    def accept(self, visitor):
        visitor.visit_circle(self)

    def get_radius(self):
        return self.radius

class Square(Element):
    def __init__(self, side):
        self.side = side

    def accept(self, visitor):
        visitor.visit_square(self)

    def get_side(self):
        return self.side

# 3. Visitor 接口
class Visitor:
    def visit_circle(self, circle):
        pass

    def visit_square(self, square):
        pass

# 4. 具体 Visitor 类
class AreaCalculatorVisitor(Visitor):
    def __init__(self):
        self._total_area = 0

    def visit_circle(self, circle):
        area = math.pi * circle.get_radius() ** 2
        print(f"Calculating area of Circle (radius={circle.get_radius()}): {area:.2f}")
        self._total_area += area

    def visit_square(self, square):
        area = square.get_side() ** 2
        print(f"Calculating area of Square (side={square.get_side()}): {area:.2f}")
        self._total_area += area

    def get_total_area(self):
        return self._total_area

class PerimeterCalculatorVisitor(Visitor):
    def __init__(self):
        self._total_perimeter = 0

    def visit_circle(self, circle):
        perimeter = 2 * math.pi * circle.get_radius()
        print(f"Calculating perimeter of Circle (radius={circle.get_radius()}): {perimeter:.2f}")
        self._total_perimeter += perimeter

    def visit_square(self, square):
        perimeter = 4 * square.get_side()
        print(f"Calculating perimeter of Square (side={square.get_side()}): {perimeter:.2f}")
        self._total_perimeter += perimeter

    def get_total_perimeter(self):
        return self._total_perimeter

# 5. 客户端代码
if __name__ == "__main__":
    # 创建图形对象
    shapes = [
        Circle(5),
        Square(4),
        Circle(3),
        Square(6)
    ]

    # 创建面积计算访问者
    area_calculator = AreaCalculatorVisitor()

    # 遍历图形，接受面积计算访问者
    print("--- Calculating Areas ---")
    for shape in shapes:
        shape.accept(area_calculator)

    print(f"\nTotal Area: {area_calculator.get_total_area():.2f}")

    # 创建周长计算访问者
    perimeter_calculator = PerimeterCalculatorVisitor()

    # 遍历图形，接受周长计算访问者
    print("\n--- Calculating Perimeters ---")
    for shape in shapes:
        shape.accept(perimeter_calculator)

    print(f"\nTotal Perimeter: {perimeter_calculator.get_total_perimeter():.2f}")

    # 假设将来需要添加新的操作，例如计算对角线长度 (对于方形)
    class DiagonalCalculatorVisitor(Visitor):
        def __init__(self):
            self._total_diagonal = 0

        def visit_circle(self, circle):
            # 圆形没有对角线，可以忽略或者打印提示
            print(f"Circle (radius={circle.get_radius()}) has no diagonal.")
            pass

        def visit_square(self, square):
            diagonal = math.sqrt(2) * square.get_side()
            print(f"Calculating diagonal of Square (side={square.get_side()}): {diagonal:.2f}")
            self._total_diagonal += diagonal

        def get_total_diagonal(self):
            return self._total_diagonal

    # 创建对角线计算访问者
    diagonal_calculator = DiagonalCalculatorVisitor()

    # 遍历图形，接受对角线计算访问者
    print("\n--- Calculating Diagonals ---")
    for shape in shapes:
        shape.accept(diagonal_calculator)

    print(f"\nTotal Diagonal (of squares): {diagonal_calculator.get_total_diagonal():.2f}")

"""

    def __repr__(self):
        return self.p
