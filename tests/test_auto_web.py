from prompt_writing_assistant.utils import super_print
from prompt_writing_assistant.web import edit_server
from prompt_writing_assistant.web import gener_post_server,gener_post_server_multi,gener_get_server

import pytest


# 格式输出不稳定

def test_edit_server():
    """
    1 uv run python -m temp_web # 搭建起虚拟服务
    2 运行, 写入服务更新, 然后获得curl 命令, 验证是否通行

    优势是: 结合了本地化, 可以同步的测试服务是否可用, 同时生成curl 命令
    缺点是: 冗长, 繁琐, 单一操作
    """
    # uv run python -m temp_web
    ask = '''
# 记忆合并
class Memory_card_list_Request(BaseModel):
    memory_cards: list[str] = Field(..., description="要评分的记忆卡片内容")

@app.post("/memory_card/merge")
async def memory_card_merge_server(request: Memory_card_list_Request):
    """
    记忆卡片质量评分
    接收一个记忆卡片内容字符串，并返回其质量评分。
    """
    result = memory_card_merge(memory_cards=request.memory_cards)
    return {
        "message": "memory card merge successfully",
        "result": result
    }

帮我修改为post方法
'''
    
    out_curl = edit_server(ask,output_py = 'temp_web.py')
    super_print(out_curl,"test_edit_server_curl")




def test_get_server():
    """
    自动生成get 方法

    优点: 可控性强, 靠模板制定函数
    缺点: 人类介入程度高, 单一操作
    """
    code = '''
    def brief(self,memory_cards:list[str])->str:
        """
        数字分身介绍
        """
        return {"title":"我的数字分身标题","content":"我的数字分身简介"}
    '''
    result = gener_get_server(prompt=code)
    super_print(result,'gener_get_server')



def test_post_server():
    """
    自动生成get 方法

    优点: 可控性强, 靠模板制定函数
    缺点: 人类介入程度高, 单一操作
    """
    code = '''
    def brief(self,memory_cards:list[str])->str:
        """
        数字分身介绍
        """
        return {"title":"我的数字分身标题","content":"我的数字分身简介"}
    '''
    result = gener_post_server(prompt=code)
    super_print(result,'post_server')



def test_post_server_multi():
    """
    自动生成get 方法

    优点: 可控性强, 靠模板制定函数, 多数操作, 有格式监控
    缺点: 人类介入程度高
    """
    code = '''
class Memo():
    def __init__(self):
        """
        初始化
        """
        
    def build_from_datas(datas:dict)->None:
        """
        构建数据
        """
        
    def get_info_from_tree(self, prompt:str)-> str:
        """
        获取信息
        """
        
    def eval_(self)-> str:
        """
        这是一个长时间运行的同步操作
        """
        
    async def aeval_(self)-> str:
        """
        这是一个长时间运行的异步操作, 作用同eval_
        """
        
    def get_eval(self)-> str:
        """
        获取eval 的执行结果
        """
    '''
    result = gener_post_server_multi(prompt=code)
    super_print(result,'post_server')
