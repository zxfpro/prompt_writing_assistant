# 测试1

import os
import functools
import json
from prompt_writing_assistant.utils import extract_json,extract_python, extract_prompt
from prompt_writing_assistant.prompts import evals_prompt
from prompt_writing_assistant.prompts import change_by_opinion_prompt
from prompt_writing_assistant.prompts import program_system_prompt
from llmada.core import BianXieAdapter
from db_help.mysql import MySQLManager
from datetime import datetime


from dotenv import load_dotenv
from enum import Enum
from prompt_writing_assistant.log import Log
logger = Log.logger

load_dotenv()
bx = BianXieAdapter()

def editing_log(content):
    logger.debug(content)

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
        SELECT id, prompt_id, version, timestamp, prompt, use_case
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

def get_specific_prompt_version(target_prompt_id, target_version, table_name, db_manager):
    """
    获取指定 prompt_id 和特定版本的数据。

    Args:
        target_prompt_id (str): 目标提示词的唯一标识符。
        target_version (int): 目标提示词的版本号。
        table_name (str): 存储提示词数据的数据库表名。
        db_manager (DBManager): 数据库管理器的实例，用于执行查询。

    Returns:
        dict or None: 如果找到，返回包含 id, prompt_id, version, timestamp, prompt 字段的字典；
                      否则返回 None。
    """
    query = f"""
        SELECT id, prompt_id, version, timestamp, prompt, use_case
        FROM {table_name}
        WHERE prompt_id = %s AND version = %s
        LIMIT 1
    """
    
    # 组合参数，顺序与 query 中的 %s 对应
    params = (target_prompt_id, target_version)
    result = db_manager.execute_query(query, params=params, fetch_one=True)
    if result:
        print(f"找到 prompt_id '{target_prompt_id}', 版本 '{target_version}' 的提示词数据。")
    else:
        print(f"未找到 prompt_id '{target_prompt_id}', 版本 '{target_version}' 的提示词数据。")
    return result


def get_prompts_from_sql(prompt_id: str,version = None,table_name = "",return_use_case = False) -> tuple[str, int]:
    """
    从sql获取提示词
    """
    db_manager = MySQLManager(
        host = os.environ.get("MySQL_DB_HOST"), 
        user = os.environ.get("MySQL_DB_USER"), 
        password = os.environ.get("MySQL_DB_PASSWORD"), 
        database=os.environ.get("MySQL_DB_NAME")
        )
    table_name = table_name or os.environ.get("MySQL_DB_Table_Name")

    # 查看是否已经存在
    if version:
        user_by_id_1 = get_specific_prompt_version(prompt_id,version,table_name,db_manager)
        if user_by_id_1:
            # 如果存在获得
            prompt = user_by_id_1.get("prompt")
            status = 1
        else:
            # 否则提示warning 然后调用最新的
            user_by_id_1 = get_latest_prompt_version(prompt_id,table_name,db_manager)
            if user_by_id_1:
                # 打印正在使用什么版本
                prompt = user_by_id_1.get("prompt")
                status = 1
            else:
                # 打印, 没有找到 warning 
                # 如果没有则返回空
                prompt = ""
                status = 0
            status = 1

    else:
        user_by_id_1 = get_latest_prompt_version(prompt_id,table_name,db_manager)
        if user_by_id_1:
            # 如果存在获得
            prompt = user_by_id_1.get("prompt")
            status = 1
        else:
            # 如果没有则返回空
            prompt = ""
            status = 0
    if not return_use_case:
        return prompt, status
    else:
        if user_by_id_1:
            editing_log(user_by_id_1.keys())
            return prompt, status, user_by_id_1.get('use_case',' 空 ')
        else:
            return prompt, status, ' 空 '

def save_prompt_by_sql(prompt_id: str,
                       new_prompt: str,
                       table_name = "prompts_data",
                       input_data:str = ""):
    """
    从sql保存提示词
    """
    db_manager = MySQLManager(
        host = os.environ.get("MySQL_DB_HOST"), 
        user = os.environ.get("MySQL_DB_USER"), 
        password = os.environ.get("MySQL_DB_PASSWORD"), 
        database=os.environ.get("MySQL_DB_NAME")
        )
    table_name = table_name or os.environ.get("MySQL_DB_Table_Name")
    # 查看是否已经存在
    user_by_id_1 = get_latest_prompt_version(prompt_id,table_name,db_manager)
    
    if user_by_id_1:
        # 如果存在版本加1
        version_ori = user_by_id_1.get("version")
        _, version = version_ori.split(".")
        version = int(version)
        version += 1
        version_ = f"1.{version}"

    else:
        # 如果不存在版本为1.0
        version_ = '1.0'
    _id = db_manager.insert(table_name, {'prompt_id': prompt_id, 
                                            'version': version_, 
                                            'timestamp': datetime.now(),
                                            "prompt":new_prompt,
                                            "use_case":input_data})

def prompt_finetune_to_sql(
    prompt_id:str,
    version = None,
    opinion: str = "",
    table_name = "prompts_data"
):
    """
    让大模型微调已经存在的system_prompt
    """
    table_name = table_name or os.environ.get("MySQL_DB_Table_Name")
    prompt, _ = get_prompts_from_sql(prompt_id = prompt_id,version = version,table_name = table_name)
    if opinion:
        new_prompt = bx.product(
            change_by_opinion_prompt.format(old_system_prompt=prompt, opinion=opinion)
        )
    else:
        new_prompt = prompt
    save_prompt_by_sql(prompt_id = prompt_id,
                       new_prompt = new_prompt,
                       table_name = table_name,
                       input_data = " ")
    print('success')





def save_prompt_by_sql_2(prompt_id: str,
                       table_name = "prompts_data",
                       use_case:str = "",
                       solution: str = ""):
    """
    从sql保存提示词
    """
    db_manager = MySQLManager(
        host = os.environ.get("MySQL_DB_HOST"), 
        user = os.environ.get("MySQL_DB_USER"), 
        password = os.environ.get("MySQL_DB_PASSWORD"), 
        database=os.environ.get("MySQL_DB_NAME")
        )
    table_name = table_name or os.environ.get("MySQL_DB_Table_Name")

    _id = db_manager.insert(table_name, {'prompt_id': prompt_id, 
                                          "use_case":use_case,
                                          "solution":solution})


class IntellectType(Enum):
    train = "train"
    inference = "inference"
    summary = "summary"


def intellect(type: str, prompt_id: str,version: str = None, demand: str = None,table_name = ""):
    """
    1 标定入参必须是第一个位置
    2 train ,inference ,summery,

    这个装饰器,在输入函数的瞬间完成大模型对于第一位参数的转变, 可以直接return 返回, 也可以在函数继续进行逻辑运行
    函数中是对大模型输出的后处理
    """
    table_name = table_name or os.environ.get("MySQL_DB_Table_Name")
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
            prompt, states, before_input = get_prompts_from_sql(prompt_id,version,table_name = table_name,
                                                  return_use_case = True)
            if input_ == "":
                input_ = "无"

            if type.value == "train":
                # 注意, 这里的调整要求使用最初的那个输入, 最好一口气调整好
                if states == 0:
                    input_prompt = "user:\n" + demand + "\n----input----\n" + input_
                elif states == 1:
                    chat_history = prompt
                    
                    if input_ == before_input: # 输入没变, 说明还是针对同一个输入进行讨论
                        if not demand:
                            # warning 这个分支不应该出现, 这里加入warning 
                            input_prompt = chat_history + "请再试一次"
                        else:

                            input_prompt = chat_history + "\nuser:" + demand
                
                    else:
                        if not demand:
                            input_prompt = chat_history + "\n----input-----\n" + input_
                        else:
                            input_prompt = chat_history + "\nuser:" + demand + "\n-----input----\n" + input_
                
                ai_result = bx.product(input_prompt)
                chat_history = input_prompt + "\nassistant:\n" + ai_result # 用聊天记录作为完整提示词
                save_prompt_by_sql(prompt_id, chat_history,table_name = table_name,
                                   input_data = input_)
                output_ = ai_result

            elif type.value == "inference":
                if states == 1:
                    ai_result = bx.product(prompt + "\n-----input----\n" +  input_)
                    save_prompt_by_sql_2(prompt_id,
                                         table_name = "use_case",
                                         use_case = input_,
                                         solution = ""
                                         )
                    output_ = ai_result
                else:
                    raise AssertionError("必须要已经存在一个prompt 否则无法总结")

            elif type.value == "summary":
                if states == 1:
                    system_prompt_created_prompt = """
    很棒, 我们已经达成了某种默契, 我们之间合作无间, 但是, 可悲的是, 当我关闭这个窗口的时候, 你就会忘记我们之间经历的种种磨合, 这是可惜且心痛的, 所以你能否将目前这一套处理流程结晶成一个优质的prompt 这样, 我们下一次只要将prompt输入, 你就能想起我们今天的磨合过程,
对了,我提示一点, 这个prompt的主角是你, 也就是说, 你在和未来的你对话, 你要教会未来的你今天这件事, 是否让我看懂到时其次

只要输出提示词内容即可, 不需要任何的说明和解释
    """
                    system_reuslt = bx.product(prompt + system_prompt_created_prompt)
                    s_prompt = extract_prompt(system_reuslt)
                    if s_prompt:
                        save_prompt_by_sql(prompt_id, s_prompt,table_name = table_name,
                                           input_data = " summary ")
                    else:
                        save_prompt_by_sql(prompt_id, system_reuslt,table_name = table_name,
                                           input_data = " summary ")


                else:
                    raise AssertionError("必须要已经存在一个prompt 否则无法总结")

            #######
            arg_list[0] = output_
            args = arg_list
            # 完成修改
            result = func(*args, **kwargs)

            return result

        return wrapper

    return outer_packing


############evals##############


class Base_Evals():
    def __init__(self):
        """
        # TODO 2 自动优化prompt 并提升稳定性, 并测试
        """
        self.MIN_SUCCESS_RATE = 00.0 # 这里定义通过阈值, 高于该比例则通过


    def _assert_eval_function(self,params):
        #这里定义函数的评价体系
        print(params,'params')

    def get_success_rate(self,test_cases:list[tuple]):
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




