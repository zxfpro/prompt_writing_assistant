# 测试 
import pytest
from prompt_writing_assistant.prompt_helper import intellect,IntellectType,aintellect
from prompt_writing_assistant.prompt_helper import prompt_finetune_to_sql, get_prompts_from_sql, save_prompt_by_sql
from prompt_writing_assistant.utils import super_print
from prompt_writing_assistant.utils import extract_article,extract_json
from prompt_writing_assistant.prompt_helper import Base_Evals
from dotenv import load_dotenv
load_dotenv()

def test_get_prompts_from_sql():
    # get_prompts_from_sql
    result = get_prompts_from_sql(prompt_id = "123134123")
    print(result)



def test_save_prompt_by_sql():
    new_prompt = """
一个测试的提示词
"""
    save_prompt_by_sql(prompt_id = "123134123",
                       new_prompt = new_prompt)


def test_prompt_finetune_to_sql():
    # prompt_finetune_to_sql
    opinion = "将这个fastapi的接口改为post方式,"
    prompt_finetune_to_sql(prompt_id = "123134123",
                           opinion=opinion)


# 半自动编写/优化提示词
def test_intellect():
    @intellect(IntellectType.train,"1231231","改为使用$符号")
    def prompts(input_):
        # 可以直接输出, 也可以编写后处理的逻辑 extract_json 等
        return input_

    result = prompts("你好, 我的电话号码是12343213123, 身份证是2454532345")
    print(result,'result')


# 半自动编写/优化提示词
async def test_aintellect():
    @aintellect(IntellectType.train,"1231231","改为使用$符号")
    def prompts(input_):
        # 可以直接输出, 也可以编写后处理的逻辑 extract_json 等
        return input_

    result = await prompts("你好, 我的电话号码是12343213123, 身份证是2454532345")
    print(result,'result')

# Evals

class AEvals(Base_Evals):
    def __init__(self):
        self.MIN_SUCCESS_RATE = 00.0 # 这里定义通过阈值, 高于该比例则通过

    def _assert_eval_function(self,params):
        print(params,'params')


def test_evals():
    aeval = AEvals()
    test_cases = [
            (1, 2, 3),      # Pass
            (-1, 1, 0),     # Pass
            (0, 0, 0),      # Pass
            (100, 200, None), # Fail
            (-5, -3, -8)    # Pass
        ]
    aeval.get_success_rate(test_cases)



#########--时空光年-数字人生--#############
"""
0098 数字分身简介
0099 数字分身性格提取
0100 # 数字分身信息脱敏
"""


import os

def test_work():
    host = os.getenv("MySQL_DB_HOST")
    user = os.getenv("MySQL_DB_USER")
    passward = os.getenv("MySQL_DB_PASSWORD")
    name = os.getenv("MySQL_DB_NAME")
    table = os.getenv("MySQL_DB_Table_Name")
    print(host)
    print()
    print(user)
    print()
    print(passward)
    print()
    print(name)
    print()
    print(table)
    print()

from datetime import datetime
from db_help.mysql import MySQLManager
def test_db_help():
    
    db_manager = MySQLManager(host=os.getenv("MySQL_DB_HOST"),
                 user=os.getenv("MySQL_DB_USER"),
                 password=os.getenv("MySQL_DB_PASSWORD"),
                 database=os.getenv("MySQL_DB_NAME")
                 )
    
    table_name = "llm_prompt"
    user1_id = db_manager.insert(table_name, {'prompt_id': '1234134', 'version': '1.0', 'timestamp': datetime.now(),"prompt":"你好"})
    user2_id = db_manager.insert(table_name, {'prompt_id': '2345234234', 'version': '1.1', 'timestamp': datetime.now(),"prompt":"你好2223"})
    db_manager.close()

