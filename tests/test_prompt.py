# 测试 
from prompt_writing_assistant.core import prompt_finetune
from prompt_writing_assistant.utils import super_print
import pytest

from prompt_writing_assistant.core import intellect,IntellectType

# 半自动编写/优化提示词
def test_work():
    @intellect(IntellectType.train,"1231231","改为使用$符号")
    def prompts(input_):
        # 可以直接输出, 也可以编写后处理的逻辑 extract_json 等
        return input_

    prompts("你好, 我的电话号码是12343213123, 身份证是2454532345")



def test_prompt_finetune():
    """
    自动化编写prompt
    """
    opinion = "将这个fastapi的接口改为post方式,"
    prompt = """

"""

    output = prompt_finetune(prompt,
              opinion = opinion)

    super_print(output,"test_prompt_writer")




# 测试提示词的稳定性
def test_add_basic_with_success_rate():
    """
    测试加法函数，并计算内部测试用例的通过比例。
    注意：这会使 pytest 将整个函数视为一个测试，
    而不是为每个数据组生成独立的测试报告。
    """
    # 这里定义通过阈值, 高于该比例则通过
    MIN_SUCCESS_RATE = 00.0 
    
    # 这里定义数据
    test_cases = [
        (1, 2, 3),      # Pass
        (-1, 1, 0),     # Pass
        (0, 0, 0),      # Pass
        (100, 200, None), # Fail
        (-5, -3, -8)    # Pass
    ]

    # 这里定义断言
    def add(input_a,input_b,expected_sum):
        assert expected_sum == input_a + input_b + 1


    successful_assertions = 0
    total_assertions = len(test_cases)
    failed_cases = []

    for i, (input_a, input_b, expected_sum) in enumerate(test_cases):
        try:
            # 这里将参数传入
            add(input_a,input_b,expected_sum)
            successful_assertions += 1
        except AssertionError as e:
            failed_cases.append(f"Case {i+1} ({input_a}+{input_b}): FAILED. Expected {expected_sum},. Error: {e}")
        except Exception as e: # 捕获其他可能的错误
            failed_cases.append(f"Case {i+1} ({input_a}+{input_b}): ERROR. Input {input_a},{input_b}. Error: {e}")
            print(f"Case {i+1} ({input_a}+{input_b}): ERROR. Input {input_a},{input_b}. Error: {e}")

    success_rate = (successful_assertions / total_assertions) * 100
    print(f"\n--- Aggregated Results ---")
    print(f"Total test cases: {total_assertions}")
    print(f"Successful cases: {successful_assertions}")
    print(f"Failed cases count: {len(failed_cases)}")
    print(f"Success Rate: {success_rate:.2f}%")

    assert success_rate >= MIN_SUCCESS_RATE, \
        f"Test failed: Success rate {success_rate:.2f}% is below required {MIN_SUCCESS_RATE:.2f}%." + \
        f"\nFailed cases details:\n" + "\n".join(failed_cases)









