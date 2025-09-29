# 测试 
import pytest
from prompt_writing_assistant.prompt_helper import IntellectType
from prompt_writing_assistant.prompt_helper import Base_Evals
from dotenv import load_dotenv
load_dotenv("local.env", override=True)

from prompt_writing_assistant.prompt_helper import Intel


def test_Intel():
    intels = Intel()

def test_get_prompts():
    intels = Intel()
    result = intels.get_prompts_from_sql(
        table_name = "prompts_data",
        prompt_id = "1231231")
    print(result)

def test_get_prompts_version():
    intels = Intel()
    result = intels.get_prompts_from_sql(
        table_name = "prompts_data",
        prompt_id = "1231231",
        version="1.1"
        )
    print(result)

def test_intellect():
    intels = Intel()
    @intels.intellect(IntellectType.train,
                      "数字人生王者001",
                      table_name ="prompts_data",
                      demand = "改为使用$符号")
    def prompts(input_):
        # 后处理, 也可以编写后处理的逻辑 extract_json 等
        return input_

    result = prompts("你好, 我的电话号码是12343213123, 身份证是2454532345")
    print(result,'result')

def test_prompt_finetune_to_sql():
    # prompt_finetune_to_sql
    intels = Intel()
    demand = "将这个fastapi的接口改为post方式,"
    result = intels.prompt_finetune_to_sql(
        prompt_id = "数字人生王者001",
        table_name ="prompts_data",
        demand=demand)
    
    print(result)


from prompt_writing_assistant.prompt_helper import Base_Evals

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




