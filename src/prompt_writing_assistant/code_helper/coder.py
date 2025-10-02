from typing import List, Dict, Any
from prompt_writing_assistant.utils import extract_python, extract_json
from llmada.core import BianXieAdapter
from prompt_writing_assistant.prompt_helper import Intel,IntellectType
from prompt_writing_assistant.log import Log
import json
import re
from prompt_writing_assistant.prompt_helper import Intel, IntellectType
from prompt_writing_assistant.utils import extract_json,extract_
import json

import importlib
import yaml
import qdrant_client
from qdrant_client import QdrantClient, models
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import VectorStoreIndex
from llama_index.core import Document
from enum import Enum
from uuid import uuid4

logger = Log.logger



intel = Intel()


@intel.intellect_2(IntellectType.inference,
                 prompt_id ="代码修改-最小化改动001",
                 table_name ="prompts_data",
                 demand = "微调代码")
def edit(data:dict):
    # 可以做pydantic 的校验
    data = extract_python(data)
    return data

def run(code, function_requirement):
    data = edit(data = {"源码":code,
            "功能需求":function_requirement
            })
    print(data)
    return data


def highly_flexible_function(x="帮我将 日期 2025/12/03 向前回退12天", **kwargs):
    try:

        params = locals()

        # 优化提示信息，只在kwargs不为空时添加入参信息
        prompt_user_part = f"{x}"
        if kwargs:
            prompt_user_part += f' 入参 {params["kwargs"]}'

        bx = BianXieAdapter()
        result = bx.product(
            prompt=f"""
# 用户输入案例：
# [用户指令和输入参数的组合，例如：帮我将 日期 2025/12/03 向前回退12天 入参 data_str = "2025/12/03"]
#
# 根据用户输入案例，自动识别用户指令，提取所有“入参”信息及其名称和值。
# 编写一个 Python 函数，函数名称固定为 `Function`。
# 函数的输入参数应严格按照识别出的“入参”名称和类型来定义。
# 函数只包含一个定义，不依赖外部库或复杂结构。
# 函数应根据用户指令实现对应功能。
# 代码应简洁且易于执行。
# 输出格式应为完整的 Python 函数定义，包括文档字符串（docstring）。
{prompt_user_part}
"""
        )
        result_ = extract_python(result)

        if not result_:
            logger.error("提取到的Python代码为空，无法执行。")
            return None  # 返回None或抛出异常

        runs = result_ + "\n" + f'result = Function(**{params["kwargs"]})'
        logger.info(f"即将执行的代码：\n{runs}")

        rut = {"result": ""}
        # 使用exec执行代码，并捕获可能的错误
        try:
            exec(
                runs, globals(), rut
            )  # 将globals()作为全局作用域，避免依赖外部locals()
        except Exception as e:
            logger.error(f"执行动态生成的代码时发生错误: {e}")
            return None  # 返回None或抛出异常

        return rut.get("result")

    except ImportError:
        logger.error("无法导入 llmada.core，请确保已安装相关库。")
        return None
    except Exception as e:
        logger.error(f"在 work 函数中发生未知错误: {e}")
        return None





#########


class Paper():
    def __init__(self,content):
        self.content = content

    @intel.intellect_2(IntellectType.train,
                 prompt_id ="白板编辑-修改画板001",
                 table_name ="prompts_data",
                 demand = "微调代码")
    def system_prompt(self,data):
        return data
        
    def talk(self,prompt:str):
        data = {"data":self.content+ prompt}
        result = self.system_prompt(data = data)
        # result = bx.product(system_prompt+ self.content+ prompt)
        print(result,'result')
        result_json = json.loads(extract_json(result))
        print(result_json,'result_json')
        for ops in result_json:
            self.deal(ops.get('type'), ops.get('operations'))
            
    
    def deal(self, type_, operations:str):
        if type_ == "add":
            self.add(operations)
        elif type_ == "delete":
            self.delete(operations)
        else:
            print('error')

    def add(self, operations:str):
        print('add running')
        match_tips = re.search(r'<mark>(.*?)</mark>', operations, re.DOTALL)
        positon_ = operations.replace(f"<mark>{match_tips.group(1)}</mark>","")
        # 锁定原文
        positon_frist = operations.replace('<mark>',"").replace('</mark>',"")
        print(positon_frist,'positon_frist')
        print('==========')
        print(positon_,'positon__')
        self.content = self.content.replace(positon_,positon_frist)

    def delete(self, operations:str):   
        # 制定替换内容
        print('delete running')
        match_tips = re.search(r'<mark>(.*?)</mark>', operations, re.DOTALL)
        positon_ = operations.replace(f"<mark>{match_tips.group(1)}</mark>","")
        # 锁定原文
        positon_frist = operations.replace('<mark>',"").replace('</mark>',"")
        print(positon_frist,'positon_frist')
        assert positon_frist in self.content
        print('==========')
        print(positon_,'positon__')
        
        self.content = self.content.replace(positon_frist,positon_)



'''
# ## 有一个原始的程序框图, -> 可以通过需求来调整程序框图 -> 结合数据库来调度程序框图
# 一个新的任务, => 基本需求, -> 根据需求调取之前的程序框图, -> 融合程序框图 -> 调整程序框图到完成满意,-> 由程序框图实现代码, 并拉取出待实现函数
# -> 用知识库中获取代码与对应的测试, 整合到待实现函数中, -> 剩余的使用封装好的包进行补充, -> 创新的补充, -> ...


inputs = """
帮我实现一个文件读取的逻辑
"""
program_chart = init_program_chart_mermaid(inputs) # TODO 类图另做吧
# 一直循环, 直到满意
program_chart = finetune_program_chart(program_chart + "如果文件不存在, 就创建")
codes = fill_code_frame_program_chart(program_chart) #TODO 可以尝试配置对应的pytest

'''

intel = Intel()

# TODO 数据用例等, 只是保存, 真正做统计的时候可以导出选择合适的数据
@intel.intellect(IntellectType.train,
                table_name="prompts_data",
               prompt_id="程序框图-根据需求创建",
               demand = """
注意输出格式:
```mermaid
程序框图内容
```
               """)
def init_program_chart_mermaid(input_):
    result = extract_(input_,"mermaid")
    input_ = f"""
```mermaid
{result}
```
    """
    with open("/Users/zhaoxuefeng/GitHub/obsidian/工作/TODO/1根据需求创建.md",'w') as f:
        f.write(input_)
    return input_

@intel.intellect(IntellectType.inference,
                table_name="prompts_data",
               prompt_id="程序框图-白板微调",
               demand = """
根据输入的程序框图, 和提出的需求, 生成新的程序框图
               """)
def finetune_program_chart(input_):
    result = extract_(input_,"mermaid")
    input_ = f"""
```mermaid
{result}
```
    """
    with open("/Users/zhaoxuefeng/GitHub/obsidian/工作/TODO/1根据需求创建.md",'w') as f:
        f.write(input_)
    return input_

@intel.intellect(IntellectType.train,
                table_name="prompts_data",
               prompt_id="程序框图-框架实现",
               demand = """
使用中文注释
               """)
def fill_code_frame_program_chart(program_chart):
    result = extract_(program_chart,"python")
    code = f"""
```python
{result}
```
    """
    
    with open("/Users/zhaoxuefeng/GitHub/obsidian/工作/TODO/3框架实现.md",'w') as f:
        f.write(code)
    return code






## 分析编码习惯


class MeetingMessageHistory:
    def __init__(self):
        self._messages: List[Dict[str, Any]] = []

    def add_message(self, role: str, content: str, speaker_name: str = None):
        """添加一条消息到历史记录。"""
        message = {"role": role, "content": content}
        if speaker_name:
            message["speaker"] = speaker_name # 添加发言者元信息
        self._messages.append(message)

    def get_messages(self) -> List[Dict[str, Any]]:
        """获取当前完整的消息历史。"""
        return self._messages

    def clear(self):
        """清空消息历史。"""
        self._messages = []

    def __str__(self) -> str:
         return "\n".join([f"[{msg.get('speaker', msg['role'])}] {msg['content']}" for msg in self._messages])
    
    
# 模拟一个外部 LLM 调用函数，以便在框架中演示
def simulate_external_llm_call(messages: List[Dict[str, Any]], model_name: str = "default") -> str:
     """模拟调用外部 LLM 函数."""
     print(messages[0].get('speaker'),'messages')
     print(model_name,'model_name')

     bx = BianXieAdapter()
     bx.set_model(model_name)
     result = bx.chat(messages)
     simulated_response = f"[{model_name}] Responding to '{result}"
     return simulated_response

class MeetingOrganizer:
    def __init__(self):
        # 存储参会者信息：名称和使用的模型
        self._participants: List[Dict[str, str]] = []
        self._history = MeetingMessageHistory()
        self._topic: str = ""
        self._background: str = ""
        # TODO: 在实际使用时，这里应该引用您真实的 LLM 调用函数
        self._llm_caller = simulate_external_llm_call # 指向您外部的 LLM 调用函数

    def set_llm_caller(self, caller_func):
         """设置外部的 LLM 调用函数."""
         self._llm_caller = caller_func
         print("External LLM caller function set.")


    def add_participant(self, name: str, model_name: str = "default"):
        """添加一个参会者 (LLM) 到会议中。"""
        participant_info = {"name": name, "model": model_name}
        self._participants.append(participant_info)
        print(f"Added participant: {name} (using model: {model_name})")

    def set_topic(self, topic: str, background: str = ""):
        """设置会议主题和背景。"""
        self._topic = topic
        self._background = background
        initial_message = f"Meeting Topic: {topic}\nBackground: {background}"
        # 可以将主题和背景作为用户输入的第一条消息，或者 system 消息
        self._history.add_message("user", initial_message, speaker_name="Meeting Host")
        print(f"Meeting topic set: {topic}")

    def run_simple_round(self):
        """执行一轮简单的会议：每个参会 LLM 基于当前历史回复一次。"""
        if not self._participants:
            print("No participants in the meeting.")
            return

        print("\n--- Running a Simple Meeting Round ---")
        current_history = self._history.get_messages()

        for participant in self._participants:
            participant_name = participant["name"]
            model_to_use = participant["model"]
            try:
                # 调用外部 LLM 函数
                print(current_history,'current_history')
                response_content = self._llm_caller(current_history, model_name=model_to_use)
                # 将回复添加到历史中，并标记发言者
                self._history.add_message("assistant", response_content, speaker_name=participant_name)
                print(f"'{participant_name}' responded.")
            except Exception as e:
                print(f"Error during '{participant_name}' participation: {e}")
                # 在框架阶段，简单的错误打印即可

    def get_discussion_history(self) -> List[Dict[str, Any]]:
        """获取完整的讨论消息历史。"""
        return self._history.get_messages()

    def get_simple_summary(self) -> str:
        """获取简单的讨论摘要（第一阶段：拼接所有 LLM 发言）。"""
        print("\n--- Generating Simple Summary ---")
        summary_parts = []
        for message in self._history.get_messages():
            # 提取 assistant 角色的发言作为摘要内容
            if message.get("role") == "assistant":
                 speaker = message.get("speaker", "Unknown Assistant")
                 summary_parts.append(f"[{speaker}]: {message['content']}")

        return "\n\n".join(summary_parts)

    def display_history(self):
         """打印格式化的讨论历史。"""
         print("\n--- Full Discussion History ---")
         print(self._history)

