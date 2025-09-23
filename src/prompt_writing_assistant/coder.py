from typing import List, Dict, Any
from typing import List, Dict, Any

import logging
from prompt_writing_assistant.utils import extract_python

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
    
    
# 假设外部有一个 LLM 调用模块
# import your_llm_module # 您需要根据您的实际情况导入
from llmada import BianXieAdapter


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





#########


# 配置日志记录
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def work(x="帮我将 日期 2025/12/03 向前回退12天", **kwargs):
    try:
        from llmada.core import BianXieAdapter

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
        xx = extract_python(result)

        if not xx:
            logging.error("提取到的Python代码为空，无法执行。")
            return None  # 返回None或抛出异常

        runs = xx + "\n" + f'result = Function(**{params["kwargs"]})'
        logging.info(f"即将执行的代码：\n{runs}")

        rut = {"result": ""}
        # 使用exec执行代码，并捕获可能的错误
        try:
            exec(
                runs, globals(), rut
            )  # 将globals()作为全局作用域，避免依赖外部locals()
        except Exception as e:
            logging.error(f"执行动态生成的代码时发生错误: {e}")
            return None  # 返回None或抛出异常

        return rut.get("result")

    except ImportError:
        logging.error("无法导入 llmada.core，请确保已安装相关库。")
        return None
    except Exception as e:
        logging.error(f"在 work 函数中发生未知错误: {e}")
        return None


# 示例调用
if __name__ == "__main__":
    pass
    # 示例1：带参数
    # result1 = work(x="帮我将 日期 2025/12/03 向前回退12天", data_str="2025/12/03", days=12)
    # print(f"示例1 执行结果: {result1}")

    # 示例2：不带参数
    # result2 = work(x="帮我计算 1加1")
    # print(f"示例2 执行结果: {result2}")



# 绘图板子
import json
from prompt_writing_assistant.utils import extract_json
import re


from llmada.core import BianXieAdapter
bx = BianXieAdapter()
bx.model_name = "gemini-2.5-flash-preview-05-20-nothinking"

# 再优化
system_prompt = """
你是一名专业的文本编辑助理，负责根据用户的指令对文本进行修改。
你的输出必须严格遵循以下 JSON 格式：

```json
[
    {
        "type": "add" | "delete", // 具体的操作类型，只能是 "add" 或 "delete"
        "operations": "前置上下文<mark>操作内容</mark>后置上下文" // 具体的操作内容及其上下文。
                       // - 对于 "add" 类型：<mark>...</mark> 内部是新增的内容。
                       // - 对于 "delete" 类型：<mark>...</mark> 内部是被删除的内容。
                       // - 上下文应简洁明了，仅包含足够定位操作的文字，不宜过长。不可修改原文本
    },
    // 可以有多个操作对象
]
"""

class Paper():
    def __init__(self,content):
        self.content = content
        
    def talk(self,prompt:str):
        result = bx.product(system_prompt+ self.content+ prompt)
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




from typing import Any, List
from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.embeddings import BaseEmbedding
from volcenginesdkarkruntime import Ark
from prompt_writing_assistant.utils import super_print

class VolcanoEmbedding(BaseEmbedding):
    _model = PrivateAttr()
    _ark_client = PrivateAttr()
    _encoding_format = PrivateAttr()

    def __init__(
        self,
        model_name: str = "doubao-embedding-text-240715",
        api_key: str = "",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._ark_client = Ark(api_key=api_key)
        self._model = model_name
        self._encoding_format = "float"
    @classmethod
    def class_name(cls) -> str:
        return "ark"
    
    def _get_query_embedding(self, query: str) -> List[float]:
        """
        获取查询字符串的 embedding。
        通常查询和文档使用相同的 embedding 模型。
        """
        
        resp = self._ark_client.embeddings.create(
            model=self._model,
            input=[query],
            encoding_format=self._encoding_format,
        )
        return resp.data[0].embedding

    def _get_text_embedding(self, text: str) -> List[float]:
        """
        获取单个文档字符串的 embedding。
        """
        resp = self._ark_client.embeddings.create(
            model=self._model,
            input=[text],
            encoding_format=self._encoding_format,
        )
        return resp.data[0].embedding

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        批量获取文档字符串的 embedding。
        如果你的火山模型支持批量推理，强烈建议实现此方法以提高效率。
        否则，它可以简单地循环调用 _get_text_embedding。
        """
        resp = self._ark_client.embeddings.create(
            model=self._model,
            input=texts,
            encoding_format=self._encoding_format,
        )
        return [i.embedding for i in resp.data]

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)




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
class TextType(Enum):
    Code = 0
    Prompt = 1

class Code_Manager():
    def __init__(self):
        self.postprocess = SimilarityPostprocessor(similarity_cutoff=0.5)
        client = qdrant_client.QdrantClient(
            host="localhost",
            port=6333,
        )
        vector_store = QdrantVectorStore(client=client, collection_name="code")
        self.embed_model = VolcanoEmbedding(model_name = "doubao-embedding-text-240715",
                                            api_key ="ac89ee8d-ba2a-4e31-bad7-021cf411c673",)
        self.index = VectorStoreIndex.from_vector_store(vector_store,embed_model=self.embed_model)
        self.retriver = self.index.as_retriever(similarity_top_k=10)

    def create_collection(self,collection_name:str = "diglife",vector_dimension:str = 2560):
        distance_metric = models.Distance.COSINE # 使用余弦相似度

        # 2. 定义 Collection 参数
        config = load_config()
        
        client = qdrant_client.QdrantClient(
            host=config.get("host","localhost"),
            port=config.get("port",6333),
        )

        # 3. 创建 Collection (推荐使用 recreate_collection)
        try:
            client.recreate_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(size=vector_dimension, distance=distance_metric),
            )
        except Exception as e:
            raise Exception("创建collection失败") from e
        finally:
            client.close()

    def update(self,text:str,type:int | TextType,id:str = None)->str:

        if isinstance(type,TextType):
            type = type.value

        if not id:
            id = str(uuid4())

        doc = Document(text = text,
                       id_=id,
                        metadata={"type":type,
                                  "id":id},
                        excluded_embed_metadata_keys=['type',"id"],
                    )
        self.index.update(document=doc)

    def delete(self,id:str):

        self.index.delete(doc_id = id)

    def search(self,query:str)->str:
        results = self.retriver.retrieve(query)
        results = self.postprocess.postprocess_nodes(results)

        for result in results:
            super_print(result.text,"result")
        return "success"





class EditCode:
    def __init__(self,py_path):
        self.py_path = py_path
        self.bx = BianXieAdapter()
        

    def edit(self,function_requirement:str):
        # 最小改动代码原则
        path = '/'.join(self.py_path.split('/')[:-1])
        with open(self.py_path,'r') as f:
            code = f.read()
        prompt = program_system_prompt.format(source_code=code,function_requirement=function_requirement)

        response = self.bx.product(prompt)
        response = extract_python(response)
        with open(self.py_path,'w') as f:
            f.write(response)
        print(f"代码已保存到{self.py_path}")

