# 测试 
import pytest
from prompt_writing_assistant.prompt_helper import intellect,IntellectType,aintellect
from prompt_writing_assistant.prompt_helper import prompt_finetune_to_sql, get_prompts_from_sql, save_prompt_by_sql
from prompt_writing_assistant.utils import super_print
from prompt_writing_assistant.prompt_helper import Base_Evals


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

