''' '''
class Explain():
    # 订阅者 
    """
    解释器模式的应用场景
编译器：编译器中的语法解析器使用解释器模式来解析和解释代码。
脚本语言：解释脚本语言，如正则表达式解析器、SQL解析器等。
配置文件解析：解析和评估简单的配置文件或命令行参数。
机器人指令解释：机器人或自动化系统中的命令解释和执行。

没太懂
    """
    def __init__(self):
        self.p = """
from abc import ABC, abstractmethod

# 上下文类，存储变量的值
class Context:
    def __init__(self):
        self.data = {}
    
    def set(self, variable, value):
        self.data[variable] = value
    
    def get(self, variable):
        return self.data.get(variable)

# 抽象表达式类
class Expression(ABC):
    @abstractmethod
    def interpret(self, context: Context):
        pass

# 终结符表达式：表示变量
class VariableExpression(Expression):
    def __init__(self, name):
        self.name = name
    
    def interpret(self, context: Context):
        return context.get(self.name)

# 终结符表达式：表示数字
class NumberExpression(Expression):
    def __init__(self, number):
        self.number = number
    
    def interpret(self, context: Context):
        return self.number

# 非终结符表达式：加法表达式
class AddExpression(Expression):
    def __init__(self, left: Expression, right: Expression):
        self.left = left
        self.right = right
    
    def interpret(self, context: Context):
        return self.left.interpret(context) + self.right.interpret(context)

# 非终结符表达式：减法表达式
class SubtractExpression(Expression):
    def __init__(self, left: Expression, right: Expression):
        self.left = left
        self.right = right
    
    def interpret(self, context: Context):
        return self.left.interpret(context) - self.right.interpret(context)

# 客户端代码
if __name__ == "__main__":
    # 创建上下文并设置变量的值
    context = Context()
    context.set("x", 10)
    context.set("y", 20)

    # 创建表达式
    expression = AddExpression(
        SubtractExpression(
            NumberExpression(5),
            VariableExpression("x")
        ),
        VariableExpression("y")
    )

    # 解释并计算表达式的值
    result = expression.interpret(context)
    print(f"Result of the expression: {result}")  # 输出：Result of the expression: 15


"""

    def __repr__(self):
        return self.p
