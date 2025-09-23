# 测试1

import os
import functools
import json
from prompt_writing_assistant.utils import extract_json,extract_python 
from prompt_writing_assistant.prompts import evals_prompt
from prompt_writing_assistant.prompts import change_by_opinion_prompt
from prompt_writing_assistant.prompts import program_system_prompt
from llmada.core import BianXieAdapter
from db_help.mysql import MySQLManager
from datetime import datetime


from dotenv import load_dotenv
from enum import Enum


load_dotenv()
bx = BianXieAdapter()


def get_prompt(prompt_id: str) -> (str, int):
    # 通过id 获取是否有prompt 如果没有则创建 prompt = "", state 0  如果有则调用, state 1
    # 读取
    main_path = "."
    prompt_file = os.path.join(main_path, f"{prompt_id}.txt")
    if not os.path.exists(prompt_file):
        with open(prompt_file, "w") as f:
            f.write("")
        return "", 0
    else:
        with open(prompt_file, "r") as f:
            prompt = f.read()
        return prompt, 1

def save_prompt(prompt_id: str, new_prompt: str):
    # 存储
    main_path = "."
    prompt_file = os.path.join(main_path, f"{prompt_id}.txt")
    with open(prompt_file, "w") as f:
        f.write(new_prompt)

def prompt_finetune(
    inputs: str,
    opinion: str = "",
    prompt_file: str = "base.prompt",
):
    """
    # 000111
    让大模型微调已经存在的system_prompt

    用户可以通过opinion来调整大模型的表现 使其进入训练模式
    当到达稳定以后, 不再对opinion 输入
    该函数就会改为推理模式
    最终可以从文件中获得提示词,
    """
    # 文件是否存在
    if os.path.exists(prompt_file):
        # read
        with open(prompt_file, "r") as f:
            prompt = f.read()
    else:
        # create
        prompt = "只是一个提示词"

    if opinion:
        new_system_prompt = bx.product(
            change_by_opinion_prompt.format(old_system_prompt=prompt, opinion=opinion)
        )
        with open(prompt_file, "w") as f:
            f.write(new_system_prompt)
        prompt = new_system_prompt

    output = bx.product(prompt + inputs)
    return output

##########

def get_latest_prompt_version(target_prompt_id,table_name,db_manager):
    """
    获取指定 prompt_id 的最新版本数据，通过创建时间判断。
    """
    query = f"""
        SELECT id, prompt_id, version, timestamp, prompt
        FROM {table_name}
        WHERE prompt_id = %s
        ORDER BY timestamp DESC, version DESC -- 如果时间相同，再按version降序排
        LIMIT 1
    """
    result = db_manager.execute_query(query, params=(target_prompt_id,), fetch_one=True)
    if result:
        print(f"找到 prompt_id '{target_prompt_id}' 的最新版本 (基于时间): {result['version']}")
    else:
        print(f"未找到 prompt_id '{target_prompt_id}' 的任何版本。")
    return result

def get_prompts_from_sql(prompt_id: str) -> (str, int):

    table_name = "prompts_data"
    db_manager = MySQLManager(
        host = os.environ.get("MySQL_DB_HOST"), 
        user = os.environ.get("MySQL_DB_USER"), 
        password = os.environ.get("MySQL_DB_PASSWORD"), 
        database=os.environ.get("MySQL_DB_NAME")
        )
    # 查看是否已经存在
    user_by_id_1 = get_latest_prompt_version(prompt_id,table_name,db_manager)
    if user_by_id_1:
        # 如果存在获得
        prompt = user_by_id_1.get("prompt")
        status = 1
    else:
        # 如果没有则返回空
        prompt = ""
        status = 0

    return prompt, status

def save_prompt_by_sql(prompt_id: str, new_prompt: str):
    """
    存储
    """
    table_name = "prompts_data"
    db_manager = MySQLManager(
        host = os.environ.get("MySQL_DB_HOST"), 
        user = os.environ.get("MySQL_DB_USER"), 
        password = os.environ.get("MySQL_DB_PASSWORD"), 
        database=os.environ.get("MySQL_DB_NAME")
        )
    # 查看是否已经存在
    user_by_id_1 = get_latest_prompt_version(prompt_id,table_name,db_manager)

    if user_by_id_1:
        # 如果存在版本加1
        version_ = float(user_by_id_1.get("version"))
        version_ += 0.1
        version_ = str(version_)
    else:
        # 如果不存在版本为1.0
        version_ = '1.0'
    _id = db_manager.insert(table_name, {'prompt_id': prompt_id, 
                                            'version': version_, 
                                            'timestamp': datetime.now(),
                                            "prompt":new_prompt})

def prompt_finetune_to_sql(
    prompt_id:str,
    opinion: str = "",
):
    """
    让大模型微调已经存在的system_prompt
    """
    prompt = get_prompts_from_sql(prompt_id = prompt_id)
    if opinion:
        new_prompt = bx.product(
            change_by_opinion_prompt.format(old_system_prompt=prompt, opinion=opinion)
        )
    else:
        new_prompt = prompt
    save_prompt_by_sql(prompt_id = prompt_id,
                       new_prompt = new_prompt)
    print('success')


class IntellectType(Enum):
    train = "train"
    inference = "inference"
    summary = "summary"

def intellect(level: str, prompt_id: str, demand: str = None):
    """
    #train ,inference ,总结,
    这个装饰器,在输入函数的瞬间完成大模型对于第一位参数的转变, 可以直接return 返回, 也可以在函数继续进行逻辑运行
    # TODO 1 可以增加cache 来节省token
    # TODO 2 自动优化prompt 并提升稳定性, 并测试

    """
    system_prompt_created_prompt = """
    很棒, 我们已经达成了某种默契, 我们之间合作无间, 但是, 可悲的是, 当我关闭这个窗口的时候, 你就会忘记我们之间经历的种种磨合, 这是可惜且心痛的, 所以你能否将目前这一套处理流程结晶成一个优质的prompt 这样, 我们下一次只要将prompt输入, 你就能想起我们今天的磨合过程,
对了,我提示一点, 这个prompt的主角是你, 也就是说, 你在和未来的你对话, 你要教会未来的你今天这件事, 是否让我看懂到时其次
    """

    def outer_packing(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 修改逻辑
            arg_list = list(args)
            input_ = arg_list[0]

            #######
            output_ = input_

            # 通过id 获取是否有prompt 如果没有则创建 prompt = "", state 0  如果有则调用, state 1
            # prompt, states = get_prompt(prompt_id)
            prompt, states = get_prompts_from_sql(prompt_id)


            if level.value == "train":
                # 注意, 这里的调整要求使用最初的那个输入, 最好一口气调整好
                if states == 0:
                    input_prompt = prompt + "\nuser:" + demand + "\n" + input_
                elif states == 1:
                    input_prompt = prompt + "\nuser:" + demand
                ai_result = bx.product(input_prompt)
                new_prompt = input_prompt + "\nassistant:" + ai_result
                save_prompt_by_sql(prompt_id, new_prompt)
                # save_prompt(prompt_id, new_prompt)
                output_ = ai_result

            elif level.value == "summary":
                if states == 1:
                    system_reuslt = bx.product(prompt + system_prompt_created_prompt)
                    # save_prompt(prompt_id, system_reuslt)
                    save_prompt_by_sql(prompt_id, system_reuslt)
                    print("successful ")

                else:
                    raise AssertionError("必须要已经存在一个prompt 否则无法总结")

            elif level.value == "inference":
                if states == 1:
                    ai_result = bx.product(prompt + input_)
                    output_ = ai_result
                else:
                    raise AssertionError("必须要已经存在一个prompt 否则无法总结")

            #######
            arg_list[0] = output_
            args = set(arg_list)
            # 完成修改
            result = func(*args, **kwargs)

            return result

        return wrapper

    return outer_packing


def aintellect(level: str, prompt_id: str, demand: str = None):
    """
    #train ,inference ,总结,
    这个装饰器,在输入函数的瞬间完成大模型对于第一位参数的转变, 可以直接return 返回, 也可以在函数继续进行逻辑运行

    """
    system_prompt_created_prompt = """
    很棒, 我们已经达成了某种默契, 我们之间合作无间, 但是, 可悲的是, 当我关闭这个窗口的时候, 你就会忘记我们之间经历的种种磨合, 这是可惜且心痛的, 所以你能否将目前这一套处理流程结晶成一个优质的prompt 这样, 我们下一次只要将prompt输入, 你就能想起我们今天的磨合过程,
对了,我提示一点, 这个prompt的主角是你, 也就是说, 你在和未来的你对话, 你要教会未来的你今天这件事, 是否让我看懂到时其次
    """

    def outer_packing(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # 修改逻辑
            arg_list = list(args)
            input_ = arg_list[0]

            #######
            output_ = input_

            # 通过id 获取是否有prompt 如果没有则创建 prompt = "", state 0  如果有则调用, state 1
            # prompt, states = get_prompt(prompt_id)
            prompt, states = get_prompts_from_sql(prompt_id)


            if level.value == "train":
                # 注意, 这里的调整要求使用最初的那个输入, 最好一口气调整好
                if states == 0:
                    input_prompt = prompt + "\nuser:" + demand + "\n" + input_
                elif states == 1:
                    input_prompt = prompt + "\nuser:" + demand
                ai_result = await bx.aproduct(input_prompt)
                new_prompt = input_prompt + "\nassistant:" + ai_result
                save_prompt_by_sql(prompt_id, new_prompt)
                # save_prompt(prompt_id, new_prompt)
                output_ = ai_result

            elif level.value == "summary":
                if states == 1:
                    system_reuslt = await bx.aproduct(prompt + system_prompt_created_prompt)
                    # save_prompt(prompt_id, system_reuslt)
                    save_prompt_by_sql(prompt_id, system_reuslt)
                    print("successful ")

                else:
                    raise AssertionError("必须要已经存在一个prompt 否则无法总结")

            elif level.value == "inference":
                if states == 1:
                    ai_result = await bx.aproduct(prompt + input_)
                    output_ = ai_result
                else:
                    raise AssertionError("必须要已经存在一个prompt 否则无法总结")

            #######
            arg_list[0] = output_
            args = set(arg_list)
            # 完成修改
            result = func(*args, **kwargs)

            return result

        return wrapper

    return outer_packing



############evals##############


class Base_Evals():
    def __init__(self):
        self.MIN_SUCCESS_RATE = 00.0 # 这里定义通过阈值, 高于该比例则通过


    def _assert_eval_function(self,params):
        print(params,'params')

    def get_success_rate(self,test_cases:list):
        """
                # 这里定义数据

        """

        successful_assertions = 0
        total_assertions = len(test_cases)
        failed_cases = []

        for i, params in enumerate(test_cases):
            try:
                # 这里将参数传入
                self._assert_eval_function(params)
                successful_assertions += 1
            except AssertionError as e:
                failed_cases.append(f"Case {i+1} ({params}): FAILED. Expected {params},. Error: {e}")
            except Exception as e: # 捕获其他可能的错误
                failed_cases.append(f"Case {i+1} ({params}): ERROR. Input {params} Error: {e}")
                print(f"Case {i+1} ({params}): ERROR. Error: {e}")

        success_rate = (successful_assertions / total_assertions) * 100
        print(f"\n--- Aggregated Results ---")
        print(f"Total test cases: {total_assertions}")
        print(f"Successful cases: {successful_assertions}")
        print(f"Failed cases count: {len(failed_cases)}")
        print(f"Success Rate: {success_rate:.2f}%")

        assert success_rate >= self.MIN_SUCCESS_RATE, \
            f"Test failed: Success rate {success_rate:.2f}% is below required {self.MIN_SUCCESS_RATE:.2f}%." + \
            f"\nFailed cases details:\n" + "\n".join(failed_cases)

    def llm_evals(
        self,
        _input: list[str],
        llm_output: list[str],
        person_output: list[str],
        rule: str,
        pass_if: str,
    ) -> "eval_result-str, eval_reason-str":
        result = bx.product(
            evals_prompt.format(
                评分规则=rule,
                输入案例=_input,
                大模型生成内容=llm_output,
                人类基准=person_output,
                通过条件=pass_if,
            )
        )
        #  eval_result,eval_reason
        # TODO
        """
        "passes": "<是否通过, True or False>",
        "reason": "<如果不通过, 基于驳回理由>",
        "suggestions_on_revision": 
        """
        result = json.loads(extract_json(result))
        return result.get("passes"), result.get("suggestions_on_revision")




