""" 工厂模式 """
class Factory():
    """ 也可以直接在类里写静态方法 """
    def __init__(self,FactoryClassName,EnumClassName,Option1,Option2):
        """
        """
        self.p = f"""
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
"""
    def __repr__(self):
        return self.p
