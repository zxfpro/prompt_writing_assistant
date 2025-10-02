""" 责任链模式  """
class Order():
    """
## 责任链 有点像勾爪, 和传送带的关系
    """
    def __init__(self):
        self.p = """

## 责任链 有点像勾爪, 和传送带的关系

from abc import ABC, abstractmethod

class Handler(ABC):
    def __init__(self):
        self._next_handler = None

    def set_next(self, handler):
        self._next_handler = handler
        return handler

    @abstractmethod
    def handle(self, request):
        if self._next_handler:
            return self._next_handler.handle(request)
        return None

class InfoHandler(Handler):
    def handle(self, request):
        if request == "INFO":
            print("InfoHandler: Handling INFO level request")
        else:
            super().handle(request)

class DebugHandler(Handler):
    def handle(self, request):
        if request == "DEBUG":
            print("DebugHandler: Handling DEBUG level request")
        else:
            super().handle(request)

class ErrorHandler(Handler):
    def handle(self, request):
        if request == "ERROR":
            print("ErrorHandler: Handling ERROR level request")
        else:
            super().handle(request)

# 客户端代码
if __name__ == "__main__":
    # 创建具体处理者
    info_handler = InfoHandler()
    debug_handler = DebugHandler()
    error_handler = ErrorHandler()

    # 设置处理链
    info_handler.set_next(debug_handler).set_next(error_handler)

    # 提交请求
    requests = ["INFO", "DEBUG", "ERROR", "UNKNOWN"]
    for req in requests:
        print(f"Client: Submitting {req} request")
        info_handler.handle(req)
        print()

"""
    def __repr__(self):
        return self.p
    



