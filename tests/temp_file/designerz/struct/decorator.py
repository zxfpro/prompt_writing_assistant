""" 装饰器模式 """


class Decorator():
    """
使用效果就是直接调用老接口的对象的话对方咩有这个方法,
那么就将老接口传入一个适配器,这样,老接口就有了新方法 有点像电源适配器,或者插头转换器

    """
    def __init__(self,decorator):
        self.p = f"""
import functools
def {decorator}(a = None):
    def outer_packing(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            print(a,'a')
            print(func.__name__)  # 函数名
            print(args)  # (1, 2)
            print(kwargs)  # 'c': 3
            return result
        return wrapper
    return outer_packing
"""
    def __repr__(self):
        return self.p