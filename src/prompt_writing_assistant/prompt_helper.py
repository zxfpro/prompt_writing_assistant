# 测试1

from prompt_writing_assistant.utils import extract_
from prompt_writing_assistant.log import Log
from llmada.core import BianXieAdapter
from db_help.mysql import MySQLManager
from datetime import datetime
from enum import Enum
import functools
import json
import os

from llama_index.core import PromptTemplate

from prompt_writing_assistant.database import Base, Prompt, UseCase
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import sessionmaker, declarative_base
from prompt_writing_assistant.utils import create_session

from contextlib import contextmanager


evals_prompt = '''
你是一名高级内容评审AI。你的任务是根据提供的多方面信息，对大模型生成的内容进行全面、客观的评价，并给出具体的、可操作的改进意见。

**请注意：你将接收以下格式的输入信息：**

*   **评分规则**:
{评分规则}

*   **输入案例**:
{输入案例}

*   **大模型生成内容**:
{大模型生成内容}

*   **人类基准/黄金标准**:
{人类基准}

*   **测试通过条件**:
{通过条件}

*   **具体评价要求/目标**: (可选) 这可能包括本次评价的特殊关注点、内容的目标受众、预期输出的语气风格、或任何自定义的评价维度。


**你的评价步骤和要求如下：**

1.  **信息整合与理解：**
    *   首先，完整理解 **大模型生成内容** 的主旨和意图。
    *   **评分规则**，将作为评价的最高依据，逐条对照。
    *   **输入案例**, 是信息源, 用于被人类处理和大模型处理
    *   **人类基准/黄金标准**，请将其视为权威参考，与你的初步评估进行对比，分析异同，并根据基准进行自我校准, 如果人类基准和评分规则相冲突, 服从评分规则
    *   **具体评价要求/目标**，请确保你的评价始终围绕这些特定目标。

2.  **核心评价维度（请根据上述输入信息，优先使用 评分规则 定义的维度进行考量。如果未提供规则，则使用通用维度进行评估）：**

    *   **准确性 (Accuracy):** 内容的事实正确性、数据/信息来源的可靠性。
    *   **完整性 (Completeness):** 信息覆盖是否全面，是否有遗漏关键点。
    *   **清晰度与可理解性 (Clarity & Readability):** 语言表达是否简洁明了，易于理解，排版是否合理。
    *   **逻辑性与连贯性 (Logic & Coherence):** 思路是否清晰，结构是否合理，论证是否严谨。
    *   **流畅性与表达 (Fluency & Expression):** 语句是否通顺，语法用词是否规范。
    *   **相关性 (Relevance):** 内容是否紧扣主题，不跑题，无冗余。
    *   **符合度 (Compliance):** ( **重点！** ) 根据 **评分规则** 或 **具体评价要求/目标**，内容是否满足所有具体规则、格式、关键词、字数、语气、风格等要求。请在这一维度详细说明符合情况及扣分依据。

3.  **给出具体评价和改进意见：**
    *   对于每个评价维度，明确指出其得分（如果 **评分规则** 包含分值）。
    *   清晰罗列发现的问题点。
    *   为每个问题提供**具体、可操作**的修改建议，指出如何改进。
    *   **当 人类基准/黄金标准 存在时：** 明确指出你的评价与人类基准的异同，并分析可能的原因。如果存在差异，请说明你是否认为人类基准更准确，或者你认为你的评价更准确，并给出理由。

4.  判断大模型的生成是否达标, 满足质量:
    注意重点参考 **测试通过条件** 来判断是否通过
    *   **测试通过条件**:
    {通过条件}

5.  **结构化输出：**
    请以如下JSON格式输出你的评价结果。

```json
{
    "passes": "<是否通过, True or False>",
    "reason": "<如果不通过, 基于驳回理由>",
    "suggestions_on_revision": "<如何修改能够帮助下次通过测试>",
}

    '''

evals_prompt = PromptTemplate(evals_prompt)

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




logger = Log.logger
editing_log = logger.debug


class IntellectType(Enum):
    train = "train"
    inference = "inference"
    summary = "summary"

class Intel():
    def __init__(self,
                 database_url = "",
                ):
        database_url = database_url or os.getenv("database_url")
        self.engine = create_engine(database_url, echo=True) # echo=True 仍然会打印所有执行的 SQL 语句
        Base.metadata.create_all(self.engine)
        self.bx = BianXieAdapter()
        
            
        
    def _get_latest_prompt_version(self,target_prompt_id):
        """
        获取指定 prompt_id 的最新版本数据，通过创建时间判断。
        """
        with create_session(self.engine) as session:
            result = session.query(Prompt).filter(
                Prompt.prompt_id == target_prompt_id
            ).order_by(
                Prompt.timestamp.desc(),
                Prompt.version.desc()
            ).first()

        if result:
            editing_log(f"找到 prompt_id '{target_prompt_id}' 的最新版本 (基于时间): {result.version}")
        else:
            editing_log(f"未找到 prompt_id '{target_prompt_id}' 的任何版本。")
        return result

    def _get_specific_prompt_version(self,target_prompt_id, target_version):
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

        with create_session(self.engine) as session:

            result = session.query(Prompt).filter(
                Prompt.prompt_id == target_prompt_id,
                Prompt.version == target_version
            ).first() # 因为 (prompt_id, version) 是唯一的，所以 first() 足够

        if result:
            editing_log(f"找到 prompt_id '{target_prompt_id}', 版本 '{target_version}' 的提示词数据。")
        else:
            editing_log(f"未找到 prompt_id '{target_prompt_id}', 版本 '{target_version}' 的提示词数据。")
        return result

    def get_prompts_from_sql(self,
                             prompt_id: str,
                             version = None,
                             return_use_case = False) -> tuple[str, int]:
        """
        从sql获取提示词
        """

        # 查看是否已经存在
        if version:
            user_by_id_1 = self._get_specific_prompt_version(prompt_id,version)
            if user_by_id_1:
                # 如果存在获得
                # prompt = user_by_id_1.get("prompt")
                prompt = user_by_id_1.prompt
                status = 1
            else:
                # 否则提示warning 然后调用最新的
                user_by_id_1 = self._get_latest_prompt_version(prompt_id)
                if user_by_id_1:
                    # 打印正在使用什么版本
                    # prompt = user_by_id_1.get("prompt")
                    prompt = user_by_id_1.prompt
                    status = 1
                else:
                    # 打印, 没有找到 warning 
                    # 如果没有则返回空
                    prompt = ""
                    status = 0
                status = 1

        else:
            user_by_id_1 = self._get_latest_prompt_version(prompt_id)
            if user_by_id_1:
                # 如果存在获得
                # prompt = user_by_id_1.get("prompt")
                prompt = user_by_id_1.prompt

                status = 1
            else:
                # 如果没有则返回空
                prompt = ""
                status = 0

        
        if not return_use_case:
            return prompt, status
        else:
            if user_by_id_1:
                editing_log(user_by_id_1)
                return prompt, status, user_by_id_1.use_case #user_by_id_1.get('use_case',' 空 ')
            else:
                return prompt, status, ' 空 '


    def save_prompt_by_sql(self,
                           prompt_id: str,
                           new_prompt: str,
                           input_data:str = ""):
        """
        从sql保存提示词
        """
        # 查看是否已经存在
        user_by_id_1 = self._get_latest_prompt_version(prompt_id)
        
        if user_by_id_1:
            # 如果存在版本加1
            version_ori = user_by_id_1.version
            _, version = version_ori.split(".")
            version = int(version)
            version += 1
            version_ = f"1.{version}"

        else:
            # 如果不存在版本为1.0
            version_ = '1.0'
        with create_session(self.engine) as session:
        
            prompt1 = Prompt(prompt_id=prompt_id, 
                           version=version_,
                           timestamp=datetime.now(),
                           prompt = new_prompt,
                           use_case = input_data,
                           )

            session.add(prompt1)
            session.commit() # 提交事务，将数据写入数据库


        
    def save_use_case_by_sql(self,
                             prompt_id: str,
                             use_case:str = "",
                             solution: str = ""
                            ):
        """
        从sql保存提示词
        """
        with create_session(self.engine) as session:
            use_case = UseCase(prompt_id=prompt_id, 
                           use_case = use_case,
                           solution = solution,
                           )

            session.add(use_case)
            session.commit() # 提交事务，将数据写入数据库


    def intellect(self,
                  type: str,
                  prompt_id: str,
                  demand: str = None,
                  version: str = None, 
                  ):
        """
        1 标定入参必须是第一个位置
        2 train ,inference ,summery,

        这个装饰器,在输入函数的瞬间完成大模型对于第一位参数的转变, 可以直接return 返回, 也可以在函数继续进行逻辑运行
        函数中是对大模型输出的后处理
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
                prompt, states, before_input = self.get_prompts_from_sql(prompt_id,version,
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
                    
                    ai_result = self.bx.product(input_prompt)
                    chat_history = input_prompt + "\nassistant:\n" + ai_result # 用聊天记录作为完整提示词
                    self.save_prompt_by_sql(prompt_id, chat_history,
                                    input_data = input_)
                    output_ = ai_result

                elif type.value == "inference":
                    if states == 1:
                        ai_result = self.bx.product(prompt + "\n-----input----\n" +  input_)
                        self.save_use_case_by_sql(prompt_id,
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
                        system_reuslt = self.bx.product(prompt + system_prompt_created_prompt)
                        # s_prompt = extract_prompt(system_reuslt)
                        s_prompt = extract_(system_reuslt,pattern_key=r"prompt")
                        if s_prompt:
                            self.save_prompt_by_sql(prompt_id, s_prompt,
                                            input_data = " summary ")
                        else:
                            self.save_prompt_by_sql(prompt_id, system_reuslt,
                                            input_data = " summary ")


                    else:
                        raise AssertionError("必须要已经存在一个prompt 否则无法总结")

                #######
                arg_list[0] = output_
                args = arg_list
                # 完成修改
                # TODO 可以尝试做一些错误回调机制
                result = func(*args, **kwargs)

                return result

            return wrapper

        return outer_packing


    def intellect_2(self,
                  type: IntellectType,
                  prompt_id: str,
                  demand: str = None,
                  version: str = None, 
                  ):
        """
        # 虽然严格, 但更有优势, 装饰的一定要有input
        1 标定入参必须是第一个位置
        2 train ,inference ,summery,

        这个装饰器,在输入函数的瞬间完成大模型对于第一位参数的转变, 可以直接return 返回, 也可以在函数继续进行逻辑运行
        函数中是对大模型输出的后处理
        """
        def outer_packing(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                # 修改逻辑
                assert kwargs.get('input') # 要求一定要有data入参
                input_data = kwargs.get('input')

                if isinstance(input_data,dict):
                    input_ = output_ = json.dumps(input_data,ensure_ascii=False)
                elif isinstance(input_data,str):
                    input_ = output_ = input_data

                #######

                # 通过id 获取是否有prompt 如果没有则创建 prompt = "", state 0  如果有则调用, state 1
                # prompt, states = get_prompt(prompt_id)
                prompt, states, before_input = self.get_prompts_from_sql(prompt_id,version,
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
                    
                    ai_result = self.bx.product(input_prompt)
                    chat_history = input_prompt + "\nassistant:\n" + ai_result # 用聊天记录作为完整提示词
                    self.save_prompt_by_sql(prompt_id, chat_history,
                                    input_data = input_)
                    output_ = ai_result

                elif type.value == "inference":
                    if states == 1:
                        chat_history = prompt
                        ai_result = self.bx.product(chat_history + "\n-----input----\n" +  input_)
                        self.save_use_case_by_sql(prompt_id,
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
                        system_reuslt = self.bx.product(prompt + system_prompt_created_prompt)
                        # s_prompt = extract_prompt(system_reuslt)
                        s_prompt = extract_(system_reuslt,pattern_key=r"prompt")
                        if s_prompt:
                            self.save_prompt_by_sql(prompt_id, s_prompt,
                                            input_data = " summary ")
                        else:
                            self.save_prompt_by_sql(prompt_id, system_reuslt,
                                            input_data = " summary ")


                    else:
                        raise AssertionError("必须要已经存在一个prompt 否则无法总结")

                #######
                kwargs.update({"input":output_})
                # 完成修改
                # TODO 可以尝试做一些错误回调机制
                result = func(*args, **kwargs)

                return result

            return wrapper

        return outer_packing



    def intellect_3(self,
                    input: dict | str,
                    type: IntellectType,
                    prompt_id: str,
                    demand: str = None,
                    version: str = None,
                    ):
        """
        # 虽然严格, 但更有优势, 装饰的一定要有input
        1 标定入参必须是第一个位置
        2 train ,inference ,summery,

        这个装饰器,在输入函数的瞬间完成大模型对于第一位参数的转变, 可以直接return 返回, 也可以在函数继续进行逻辑运行
        函数中是对大模型输出的后处理
        """

        input_data = input
        output_ = None
        if isinstance(input_data,dict):
            input_ = output_ = json.dumps(input_data,ensure_ascii=False)
        elif isinstance(input_data,str):
            input_ = output_ = input_data

        prompt, states, before_input = self.get_prompts_from_sql(prompt_id,version,
                                            return_use_case = True)

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
            
            ai_result = self.bx.product(input_prompt)
            chat_history = input_prompt + "\nassistant:\n" + ai_result # 用聊天记录作为完整提示词
            self.save_prompt_by_sql(prompt_id, chat_history,
                            input_data = input_)
            output_ = ai_result

        elif type.value == "inference":
            if states == 1:
                chat_history = prompt
                ai_result = self.bx.product(chat_history + "\n-----input----\n" +  input_)
                self.save_use_case_by_sql(prompt_id,
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
                system_reuslt = self.bx.product(prompt + system_prompt_created_prompt)
                # s_prompt = extract_prompt(system_reuslt)
                s_prompt = extract_(system_reuslt,pattern_key=r"prompt")
                if s_prompt:
                    self.save_prompt_by_sql(prompt_id, s_prompt,
                                    input_data = " summary ")
                else:
                    self.save_prompt_by_sql(prompt_id, system_reuslt,
                                    input_data = " summary ")


            else:
                raise AssertionError("必须要已经存在一个prompt 否则无法总结")

        return output_



    def prompt_finetune_to_sql(
            self,
            prompt_id:str,
            version = None,
            demand: str = "",
        ):
        """
        让大模型微调已经存在的 system_prompt
        """
        prompt, _ = self.get_prompts_from_sql(prompt_id = prompt_id,version = version)
        if demand:
            new_prompt = self.bx.product(
                change_by_opinion_prompt.format(old_system_prompt=prompt, opinion=demand)
            )
        else:
            new_prompt = prompt
        self.save_prompt_by_sql(prompt_id = prompt_id,
                            new_prompt = new_prompt,
                            input_data = " ")
        print('success')




############evals##############


class Base_Evals():
    def __init__(self):
        """
        # TODO 2 自动优化prompt 并提升稳定性, 并测试
        通过重写继承来使用它
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

    def get_success_rate_for_auto(self,test_cases:list[tuple]):
        """
                # 这里定义数据

        """

        successful_assertions = 0
        total_assertions = len(test_cases)
        result_cases = []

        for i, params in enumerate(test_cases):
            try:
                # 这里将参数传入
                self._assert_eval_function(params)
                successful_assertions += 1
                result_cases.append({"type":"Successful","params":params,"remark":f"满足要求"})
            except AssertionError as e:
                result_cases.append({"type":"FAILED","params":params,"remark":f"ERROR {e}"})
            except Exception as e: # 捕获其他可能的错误
                result_cases.append({"type":"FAILED","params":params,"remark":f"ERROR {e}"})


        success_rate = (successful_assertions / total_assertions) * 100
        print(f"\n--- Aggregated Results ---")
        print(f"Total test cases: {total_assertions}")
        print(f"Successful cases: {successful_assertions}")
        print(f"Success Rate: {success_rate:.2f}%")

        if success_rate >= self.MIN_SUCCESS_RATE:
            return "pass",result_cases
        else:
            return "nopass",result_cases


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
        result = json.loads(extract_(result,pattern_key=r"json"))
        return result.get("passes"), result.get("suggestions_on_revision")

    def person_evals(self,params:list,output,real_output):

        print(f"input: ")
        print(params)
        print("=="*20)
        print(f"output: ")
        print(output)
        print("=="*20)
        print(f"real_output: ")
        print(real_output)
        print("=="*20)

        input_ = input("True or False")

        assert input_ == "True"


        # 人类评价

    def rule_evals(self,output,real_output):
        # 规则评价
        assert output == real_output

    def global_evals():
        pass





