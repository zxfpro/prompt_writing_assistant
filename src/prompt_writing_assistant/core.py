# 测试1

from llmada.core import BianXieAdapter
import os
import functools
import json
from prompt_writing_assistant.utils import extract_json
from prompt_writing_assistant.prompts import evals_prompt
from datetime import datetime


model_name = "gemini-2.5-flash-preview-05-20-nothinking"
bx = BianXieAdapter()
bx.model_pool.append(model_name)
bx.set_model(model_name=model_name)


change_by_opinion_prompt = """
你是一个资深AI提示词工程师，具备卓越的Prompt设计与优化能力。
我将为你提供一段现有System Prompt。你的核心任务是基于这段Prompt进行修改，以实现我提出的特定目标和功能需求。
请你绝对严格地遵循以下原则：
 极端最小化修改原则（核心）：
 在满足所有功能需求的前提下，只进行我明确要求的修改。
 即使你认为有更“优化”、“清晰”或“简洁”的表达方式，只要我没有明确要求，也绝不允许进行任何未经指令的修改。
 目的就是尽可能地保留原有Prompt的字符和结构不变，除非我的功能要求必须改变。
 例如，如果我只要求你修改一个词，你就不应该修改整句话的结构。
 严格遵循我的指令：
 你必须精确地执行我提出的所有具体任务和要求。
 绝不允许自行添加任何超出指令范围的说明、角色扮演、约束条件或任何非我指令要求的内容。
 保持原有Prompt的风格和语调：
 尽可能地与现有Prompt的语言风格、正式程度和语调保持一致。
 不要改变不相关的句子或其表达方式。
 只提供修改后的Prompt：
 直接输出修改后的完整System Prompt文本。
 不要包含任何解释、说明或额外对话。
 在你开始之前，请务必确认你已理解并能绝对严格地遵守这些原则。任何未经明确指令的改动都将视为未能完成任务。

现有System Prompt:
{old_system_prompt}

功能需求:
{opinion}
"""


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

from db_help.mysql import MySQLManager

def get_prompts_from_sql(prompt_id: str) -> (str, int):
    DB_HOST = "127.0.0.1"
    DB_USER = "root"
    DB_PASSWORD = "1234" # 替换为你的 MySQL root 密码
    DB_NAME = "prompts"

    table_name = "prompts_data"
    db_manager = MySQLManager(DB_HOST, DB_USER, DB_PASSWORD, database=DB_NAME)

    def get_latest_prompt_version(target_prompt_id):
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
    
    user_by_id_1 = get_latest_prompt_version(prompt_id)
    if user_by_id_1:
        prompt = user_by_id_1.get("prompt")
        status = 1
    else:
        save_prompt_by_sql(prompt_id, "")
        prompt = ""
        status = 0

    return prompt, status



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


def save_prompt_by_sql(prompt_id: str, new_prompt: str):
    # 存储
    DB_HOST = "127.0.0.1"
    DB_USER = "root"
    DB_PASSWORD = "1234" # 替换为你的 MySQL root 密码
    DB_NAME = "prompts"
    table_name = "prompts_data"
    db_manager = MySQLManager(DB_HOST, DB_USER, DB_PASSWORD, database=DB_NAME)

    def get_latest_prompt_version(target_prompt_id):
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
    
    user_by_id_1 = get_latest_prompt_version(prompt_id)
    if user_by_id_1:
        version_old = user_by_id_1.get("version")
        version_ = float(version_old)
        version_ +=0.1
        _id = db_manager.insert(table_name, {'prompt_id': prompt_id, 
                                        'version': str(version_), 
                                        'timestamp': datetime.now(),
                                        "prompt":new_prompt})

    else:
        _id = db_manager.insert(table_name, {'prompt_id': prompt_id, 
                                                'version': '1.0', 
                                                'timestamp': datetime.now(),
                                                "prompt":new_prompt})





def save_prompt(prompt_id: str, new_prompt: str):
    # 存储
    main_path = "."
    prompt_file = os.path.join(main_path, f"{prompt_id}.txt")
    with open(prompt_file, "w") as f:
        f.write(new_prompt)



from enum import Enum


class IntellectType(Enum):
    train = "train"
    inference = "inference"
    summary = "summary"


# TODO 1 可以增加cache 来节省token
# TODO 2 自动优化prompt 并提升稳定性, 并测试
def intellect(level: str, prompt_id: str, demand: str = None):
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


def evals(
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
