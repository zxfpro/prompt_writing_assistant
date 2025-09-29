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


'''

### 文件结构
```
src/test_111/
├── __init__.py
├── adapter.py
├── cli.py
├── core.py
├── funs.py
├── log.py
├── mock.py
└── server.py
```

### 文件内容

1. **`adapter.py`**:
```python
# 做适配器的文件

# test_111/adapter.py
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, TextIO
from src.test_111.funs import RealDataReaderImpl, RealDataAnalyzerImpl, RealReportGeneratorImpl
from src.test_111.mock import MockDataReader, MockDataAnalyzer, MockReportGenerator

# --- 1. 定义接口 (Interfaces / Ports) ---
# ▼▼▼ MODIFICATION / ADDITION: New interfaces for CSV analysis capabilities ▼▼▼
class IDataReader(ABC):
    """
    数据读取器接口：定义从源读取数据的能力。
    """
    @abstractmethod
    def read_csv(self, source: str) -> List[Dict]:
        """从指定源读取CSV数据并返回字典列表。"""
        pass

class IDataAnalyzer(ABC):
    """
    数据分析器接口：定义对数据进行统计分析的能力。
    """
    @abstractmethod
    def analyze_column_stats(self, data: List[Dict], column_name: str) -> Any: # 返回IAnalysisStrategy
        """分析指定列的统计数据（例如：总和、平均值）。"""
        pass

class IReportGenerator(ABC):
    """
    报告生成器接口：定义根据分析结果生成报告的能力。
    """
    @abstractmethod
    def generate_report(self, analysis_results: Dict[str, Any]) -> str:
        """根据分析结果生成可读的报告字符串。"""
        pass

# --- 2. AdapterType 枚举 (用于工厂模式) ---
class AdapterType(Enum):
    MOCK = 'MOCK'
    REAL = 'REAL'

# --- 3. 适配器工厂 (AdapterFactory) ---
# ▼▼▼ YOUR FACTORY TEMPLATE IS USED HERE (Adapted to our needs) ▼▼▼
class AdapterFactory:
    @classmethod
    def new_data_reader(cls, type: AdapterType, **kwargs) -> IDataReader:
        if type == AdapterType.MOCK:
            return MockDataReader()
        elif type == AdapterType.REAL:
            return RealDataReader()
        else:
            raise ValueError(f"Unknown AdapterType for IDataReader: {type}")

    @classmethod
    def new_data_analyzer(cls, type: AdapterType, **kwargs) -> IDataAnalyzer:
        if type == AdapterType.MOCK:
            return MockDataAnalyzer()
        elif type == AdapterType.REAL:
            return RealDataAnalyzer()
        else:
            raise ValueError(f"Unknown AdapterType for IDataAnalyzer: {type}")

    @classmethod
    def new_report_generator(cls, type: AdapterType, **kwargs) -> IReportGenerator:
        if type == AdapterType.MOCK:
            return MockReportGenerator()
        elif type == AdapterType.REAL:
            return RealReportGenerator()
        else:
            raise ValueError(f"Unknown AdapterType for IReportGenerator: {type}")

# --- 4. 适配器具体实现 (Mock & Real) ---
# Real 适配器 (继承自funs.py中的实现)
class RealDataReader(IDataReader, RealDataReaderImpl):
    def __init__(self):
        super().__init__()

from src.test_111.core import IAnalysisStrategy # 导入IAnalysisStrategy

class RealDataAnalyzer(IDataAnalyzer):
    def __init__(self):
        # RealDataAnalyzerImpl 实际上就是我们需要的分析策略
        self._strategy = RealDataAnalyzerImpl()

    def analyze_column_stats(self, data: List[Dict], column_name: str) -> IAnalysisStrategy:
        # 这里不再执行分析，而是返回一个IAnalysisStrategy的实例
        return self._strategy

class MockDataAnalyzer(IDataAnalyzer):
    def __init__(self):
        # MockDataAnalyzerImpl 实际上就是我们需要的分析策略
        self._strategy = MockDataAnalyzerImpl()

    def analyze_column_stats(self, data: List[Dict], column_name: str) -> IAnalysisStrategy:
        # 这里不再执行分析，而是返回一个IAnalysisStrategy的实例
        return self._strategy

class RealReportGenerator(IReportGenerator, RealReportGeneratorImpl):
    def __init__(self):
        super().__init__()
```

2. **`core.py`**:
```python
# 编写核心使用代码的部分

# test_111/funs.py
from typing import Any, Dict, List
from abc import ABC, abstractmethod
from src.test_111.adapter import IDataReader, IDataAnalyzer, IReportGenerator, AdapterFactory, AdapterType

# 定义分析策略接口
class IAnalysisStrategy(ABC):
    """
    分析策略接口：定义不同的数据分析算法。
    """
    @abstractmethod
    def analyze(self, data: List[Dict], column_name: str) -> Dict[str, Any]:
        """执行具体的分析算法。"""
        pass

# 这里应该只包含设计模式相关的内容，而不包含具体的逻辑实现
# 具体的逻辑实现都应该从adapter导入
# 例如，你可以定义一个使用这些接口的Facade或者Service类
class CSVProcessor:
    def __init__(self, data_reader: IDataReader, data_analyzer: IDataAnalyzer, report_generator: IReportGenerator):
        self.data_reader = data_reader
        self.data_analyzer = data_analyzer
        self.report_generator = report_generator

    def process_csv_data(self, source: str, column_name: str) -> str:
        data = self.data_reader.read_csv(source)
        # 从data_analyzer获取IAnalysisStrategy的实例
        analysis_strategy: IAnalysisStrategy = self.data_analyzer.analyze_column_stats(data, column_name)
        analysis_results = analysis_strategy.analyze(data, column_name)
        report = self.report_generator.generate_report(analysis_results)
        return report

# 示例如何使用AdapterFactory来获取具体的实现
def get_real_csv_processor():
    reader = AdapterFactory.new_data_reader(AdapterType.REAL)
    analyzer = AdapterFactory.new_data_analyzer(AdapterType.REAL)
    generator = AdapterFactory.new_report_generator(AdapterType.REAL)
    return CSVProcessor(reader, analyzer, generator)

def get_mock_csv_processor():
    reader = AdapterFactory.new_data_reader(AdapterType.MOCK)
    analyzer = AdapterFactory.new_data_analyzer(AdapterType.MOCK)
    generator = AdapterFactory.new_report_generator(AdapterType.MOCK)
    return CSVProcessor(reader, analyzer, generator)
```

3. **`funs.py`**:
```python
# test_111/funs.py
from typing import Any, Dict, List
import csv # 用于真实读取CSV
import io # 用于真实读取CSV

# ▼▼▼ MODIFICATION / ADDITION: Real implementations of capabilities ▼▼▼
class RealDataReaderImpl:
    def __init__(self):
        print("RealDataReaderImpl: Initialized (reads actual CSV data).")

    def read_csv(self, source: str) -> List[Dict]:
        """从指定路径读取CSV数据并返回字典列表。"""
        print(f"RealDataReaderImpl: Reading actual CSV from {source}.")
        # 实际逻辑：从文件中读取
        # 为了演示，我们假设source是一个包含CSV数据的字符串或者一个文件路径
        # 这里为了不依赖实际文件，假设source就是CSV内容
        if source.endswith(".csv"):
             with open(source, 'r', encoding='utf-8') as f:
                 reader = csv.DictReader(f)
                 return list(reader)
        else: # 假设source直接是CSV内容的字符串
            f = io.StringIO(source)
            reader = csv.DictReader(f)
            return list(reader)

class RealDataAnalyzerImpl:
    def __init__(self):
        print("RealDataAnalyzerImpl: Initialized (performs actual analysis).")

    def analyze_column_stats(self, data: List[Dict], column_name: str) -> Dict[str, Any]:
        """对指定列进行实际的统计分析。"""
        print(f"RealDataAnalyzerImpl: Analyzing actual column '{column_name}'.")
        values = []
        for row in data:
            if column_name in row:
                try:
                    values.append(float(row[column_name])) # 尝试转换为数字进行计算
                except (ValueError, TypeError):
                    continue # 忽略非数字值

        if not values:
            return {"sum": 0.0, "avg": 0.0, "count": 0}

        total_sum = sum(values)
        return {"sum": total_sum, "avg": total_sum / len(values), "count": len(values)}

class RealReportGeneratorImpl:
    def __init__(self):
        print("RealReportGeneratorImpl: Initialized (generates actual report).")

    def generate_report(self, analysis_results: Dict[str, Any]) -> str:
        """根据分析结果生成可读的报告字符串。"""
        print(f"RealReportGeneratorImpl: Generating actual report for {analysis_results}.")
        report_str = f"--- CSV Analysis Report ---\n"
        for key, value in analysis_results.items():
            report_str += f"{key}: {value}\n"
        report_str += "---------------------------"
        return report_str
```

4. **`mock.py`**:
```python
# test_111/mock.py
from typing import Any, Dict, List

# ▼▼▼ MODIFICATION / ADDITION: Mock implementations of capabilities ▼▼▼
class MockDataReaderImpl: # 不直接继承接口，因为这里只是一个具体的实现，通过工厂注入
    def __init__(self):
        print("MockDataReaderImpl: Initialized (returns dummy CSV data).")

    def read_csv(self, source: str) -> List[Dict]:
        """模拟读取CSV数据，返回固定假数据。"""
        print(f"MockDataReaderImpl: Mock reading CSV from {source}.")
        return [
            {"id": "mock_1", "value": 10, "category": "X"},
            {"id": "mock_2", "value": 20, "category": "Y"},
            {"id": "mock_3", "value": 30, "category": "X"},
        ]

class MockDataAnalyzerImpl:
    def __init__(self):
        print("MockDataAnalyzerImpl: Initialized (returns dummy analysis).")

    def analyze_column_stats(self, data: List[Dict], column_name: str) -> Dict[str, Any]:
        """模拟分析指定列的统计数据，返回固定假结果。"""
        print(f"MockDataAnalyzerImpl: Mock analyzing column '{column_name}'.")
        return {"sum": 60, "avg": 20, "count": 3, "mocked": True} # 对应上面假数据10+20+30=60

class MockReportGeneratorImpl:
    def __init__(self):
        print("MockReportGeneratorImpl: Initialized (generates dummy report).")

    def generate_report(self, analysis_results: Dict[str, Any]) -> str:
        """模拟生成报告，返回固定假报告字符串。"""
        print(f"MockReportGeneratorImpl: Mock generating report for {analysis_results}.")
        return "--- MOCK REPORT ---\nSum: 60\nAvg: 20\n-------------------"
```

5. **`cli.py`**:
```python
# test_111/cli.py
import argparse
import sys
# ▼▼▼ MODIFICATION / ADDITION: Importing main for initialization and core for API ▼▼▼
from test_111.core import CoreAPI
from test_111.adapter import AdapterFactory, AdapterType # 需要AdapterFactory来设置adapter
from test_111.funs import RealDataReaderImpl, RealDataAnalyzerImpl, RealReportGeneratorImpl # 真实实现
from test_111.mock import MockDataReaderImpl, MockDataAnalyzerImpl, MockReportGeneratorImpl # 模拟实现

# ▼▼▼ MODIFICATION / ADDITION: Centralized initialization for CLI context ▼▼▼
def initialize_app_for_cli(env: str):
    print(f"\n--- CLI Initializing in '{env}' mode ---")
    
    # 决定使用哪个实现子线 (mock.py 还是 funs.py)
    if env == "dev":
        reader_impl = MockDataReaderImpl()
        analyzer_impl = MockDataAnalyzerImpl()
        report_gen_impl = MockReportGeneratorImpl()
    else: # env == "prod"
        reader_impl = RealDataReaderImpl()
        analyzer_impl = RealDataAnalyzerImpl()
        report_gen_impl = RealReportGeneratorImpl()
        
    # 通过AdapterFactory包装成接口类型
    # 注意：这里我们手动将Impl包装成AdapterFactory的返回类型，因为AdapterFactory的new方法返回的是接口
    # 在main.py中AdapterFactory的new方法会直接创建Mock/RealAdapter，此处为了演示将funs/mock作为单独的文件注入，需要稍微调整工厂逻辑
    
    # 假设AdapterFactory的new_*方法可以接收Impl实例，并将其包装为对应的接口
    # 或者，我们调整AdapterFactory，让它根据env直接返回funs/mock中的Impl
    
    # 【重要调整】：为了直接使用funs.py和mock.py作为插拔点，我们需要调整AdapterFactory
    #              使其不再直接实例化Mock/RealDataReader等，而是实例化mock.py或funs.py中的Impl类
    #              然后这些Impl类自身就是接口的实现者（或者被adapter.py中的Adapter包装）

    # 简化的处理方式：直接在工厂中选择funs/mock的Impl作为真正的接口实现
    data_reader_adapter = reader_impl # 假设这些Impl类本身就是IDataReader等接口的实现
    data_analyzer_adapter = analyzer_impl
    report_generator_adapter = report_gen_impl

    domain_service = CoreAPI.CoreDomainService(
        data_reader=data_reader_adapter,
        data_analyzer=data_analyzer_adapter,
        report_generator=report_generator_adapter
    )
    CoreAPI(domain_service=domain_service)
    print("--- CLI Initialization Complete ---\n")


def run_cli():
    parser = argparse.ArgumentParser(description="CSV Analysis CLI Tool.")
    parser.add_argument("command", choices=["analyze"], help="Command to execute.")
    parser.add_argument("--env", default="dev", choices=["dev", "prod"], help="Environment (dev for mock, prod for real).")
    parser.add_argument("--source", default="dummy_data.csv", help="CSV data source (file path or dummy content).")
    parser.add_argument("--column", default="value", help="Column name to analyze.")

    args = parser.parse_args()

    # 初始化应用
    initialize_app_for_cli(args.env)
    
    # 获取CoreAPI实例
    core_api = CoreAPI.get_instance()

    try:
        if args.command == "analyze":
            report = core_api.analyze_csv_and_generate_report(args.source, args.column)
            print("\n--- Final Report ---")
            print(report)
            print("--------------------")
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    # 为了演示，这里的if __name__ == '__main__'直接运行CLI
    run_cli()
```

6. **`server.py`**:
```python
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware

# from core.py


if __name__ == "__main__":
    # 这是一个标准的 Python 入口点惯用法
    # 当脚本直接运行时 (__name__ == "__main__")，这里的代码会被执行
    # 当通过 python -m YourPackageName 执行 __main__.py 时，__name__ 也是 "__main__"
    import argparse
    import uvicorn
    from .log import Log

    parser = argparse.ArgumentParser(
        description="Start a simple HTTP server similar to http.server."
    )
    parser.add_argument(
        'port',
        metavar='PORT',
        type=int,
        nargs='?', # 端口是可选的
        default=8008,
        help='Specify alternate port [default: 8000]'
    )

    parser.add_argument(
        '--env',
        type=str,
        default='dev', # 默认是开发环境
        choices=['dev', 'prod'],
        help='Set the environment (dev or prod) [default: dev]'
    )

    args = parser.parse_args()

    port = args.port
    print(args.env)
    if args.env == "dev":
        port += 100
        Log.reset_level('debug',env = args.env)
        reload = False
    elif args.env == "prod":
        Log.reset_level('info',env = args.env)# ['debug', 'info', 'warning', 'error', 'critical']
        reload = False
    else:
        reload = False

    # 使用 uvicorn.run() 来启动服务器
    # 参数对应于命令行选项
    uvicorn.run(
        app, # 要加载的应用，格式是 "module_name:variable_name"
        host="0.0.0.0",
        port=port,
        reload=reload  # 启用热重载
    )
```

以上是 `src/test_111` 目录下的文件结构和完整内容。




你是一个专业的python架构师, 请使用以下设计模式给定模版, 并针对具体问题灵活使用


# 模式模版

## 适配器模式
class NewPrinter(ABC):
    def print_content(self,content):
        raise NotImplementedError
    
class Adapter(NewPrinter):
    def __init__(self, old_function):
        self.old_function = old_function
        
    def print_content(self, content):
        self.old_function.print(content)

## 代理模式
class Image(ABC):
    @abstractmethod
    def display(self):
        pass

class RealImage(Image):
    def __init__(self, filename):
        self.filename = filename
        self.load_from_disk()

    def load_from_disk(self):
        print(f"Loading {self.filename}")

    def display(self):
        print(f"Displaying {self.filename}")

class ProxyImage(Image):
    def __init__(self, filename):
        self.filename = filename
        self.real_image = None

    def display(self):
        if self.real_image is None:
            self.real_image = RealImage(self.filename)
        self.real_image.display()

proxy_image = ProxyImage("test_image.jpg")

proxy_image.display()


## 外观模式
class TV:
    def on(self):
        print("TV is on")

    def off(self):
        print("TV is off")

class SoundSystem:
    def on(self):
        print("Sound system is on")

    def off(self):
        print("Sound system is off")

    def set_volume(self, volume):
        print(f"Sound system volume set to {volume}")

class DVDPlayer:
    def on(self):
        print("DVD player is on")

    def off(self):
        print("DVD player is off")

    def play(self, movie):
        print(f"Playing movie: {movie}")


class HomeTheaterFacade:
    def __init__(self, tv: TV, sound_system: SoundSystem, dvd_player: DVDPlayer):
        self._tv = tv
        self._sound_system = sound_system
        self._dvd_player = dvd_player

    def watch_movie(self, movie):
        print("Get ready to watch a movie...")
        self._tv.on()
        self._sound_system.on()
        self._sound_system.set_volume(20)
        self._dvd_player.on()
        self._dvd_player.play(movie)

    def end_movie(self):
        print("Shutting down the home theater...")
        self._tv.off()
        self._sound_system.off()
        self._dvd_player.off()

tv = TV()
sound_system = SoundSystem()
dvd_player = DVDPlayer()

home_theater = HomeTheaterFacade(tv, sound_system, dvd_player)
home_theater.watch_movie("Inception")
home_theater.end_movie()

## 组合模式
from abc import ABC, abstractmethod

class FileSystemComponent(ABC):
    @abstractmethod
    def operation(self):
        pass

class File(FileSystemComponent):
    def __init__(self, name):
        self.name = name

    def operation(self):
        return f"File: {self.name}"

class Directory(FileSystemComponent):
    def __init__(self, name):
        self.name = name
        self.children = []

    def add(self, component: FileSystemComponent):
        self.children.append(component)

    def remove(self, component: FileSystemComponent):
        self.children.remove(component)

    def operation(self):
        results = [f"Directory: {self.name}"]
        for child in self.children:
            results.append(child.operation())
        return "\n".join(results)

## 桥接模式
import abc

class DrawingAPI(abc.ABC):
    """
    定义绘图功能的接口，例如在不同操作系统上的具体绘图操作。
    """
    @abc.abstractmethod
    def draw_circle(self, x: int, y: int, radius: int):
        pass

    @abc.abstractmethod
    def draw_rectangle(self, x1: int, y1: int, x2: int, y2: int):
        pass

class WindowsDrawingAPI(DrawingAPI):
    def draw_circle(self, x: int, y: int, radius: int):
        print(f"在 Windows 系统上绘制圆形: 中心({x},{y}), 半径{radius}")

    def draw_rectangle(self, x1: int, y1: int, x2: int, y2: int):
        print(f"在 Windows 系统上绘制矩形: 左上({x1},{y1}), 右下({x2},{y2})")

class LinuxDrawingAPI(DrawingAPI):
    def draw_circle(self, x: int, y: int, radius: int):
        print(f"在 Linux 系统上绘制圆形: 中心({x},{y}), 半径{radius}")

    def draw_rectangle(self, x1: int, y1: int, x2: int, y2: int):
        print(f"在 Linux 系统上绘制矩形: 左上({x1},{y1}), 右下({x2},{y2})")

class MacOSDrawingAPI(DrawingAPI):
    def draw_circle(self, x: int, y: int, radius: int):
        print(f"在 MacOS 系统上绘制圆形: 中心({x},{y}), 半径{radius}")

    def draw_rectangle(self, x1: int, y1: int, x2: int, y2: int):
        print(f"在 MacOS 系统上绘制矩形: 左上({x1},{y1}), 右下({x2},{y2})")

class Shape(abc.ABC):
    """
    定义形状的抽象类，持有绘图API的引用。
    """
    def __init__(self, drawing_api: DrawingAPI):
        self._drawing_api = drawing_api

    @abc.abstractmethod
    def draw(self):
        """抽象的绘制方法"""
        pass

    @abc.abstractmethod
    def resize(self, factor: float):
        """抽象的调整大小方法"""
        pass

class Circle(Shape):
    def __init__(self, x: int, y: int, radius: int, drawing_api: DrawingAPI):
        super().__init__(drawing_api)
        self._x = x
        self._y = y
        self._radius = radius

    def draw(self):
        self._drawing_api.draw_circle(self._x, self._y, self._radius)

    def resize(self, factor: float):
        self._radius *= factor
        print(f"圆形半径调整为: {self._radius}")

class Rectangle(Shape):
    def __init__(self, x1: int, y1: int, x2: int, y2: int, drawing_api: DrawingAPI):
        super().__init__(drawing_api)
        self._x1 = x1
        self._y1 = y1
        self._x2 = x2
        self._y2 = y2

    def draw(self):
        self._drawing_api.draw_rectangle(self._x1, self._y1, self._x2, self._y2)

    def resize(self, factor: float):
        width = (self._x2 - self._x1) * factor
        height = (self._y2 - self._y1) * factor
        self._x2 = self._x1 + width
        self._y2 = self._y1 + height
        print(f"矩形调整为: ({self._x1},{self._y1}) 到 ({self._x2},{self._y2})")

if __name__ == "__main__":
    windows_api = WindowsDrawingAPI()
    circle_on_windows = Circle(10, 20, 5, windows_api)
    rect_on_windows = Rectangle(5, 5, 15, 10, windows_api)

    print("--- 在 Windows 上绘制 ---")
    circle_on_windows.draw()
    rect_on_windows.draw()
    circle_on_windows.resize(1.5)
    circle_on_windows.draw()

    print("\n--- 在 Linux 上绘制 ---")
    linux_api = LinuxDrawingAPI()
    circle_on_linux = Circle(30, 40, 8, linux_api)
    rect_on_linux = Rectangle(20, 20, 30, 25, linux_api)

    circle_on_linux.draw()
    rect_on_linux.draw()

    print("\n--- 在 MacOS 上绘制 ---")
    macos_api = MacOSDrawingAPI()
    circle_on_macos = Circle(50, 60, 10, macos_api)
    circle_on_macos.draw()



    
这是我们的两个设计模式模版, 我现在希望将其融合一下, 用工厂模式提供一些原料管理, 并用这些原料用建造者模式建造

## 工厂模式
from enum import Enum
from typing import List, Any

class {EnumClassName}(Enum):
    {Option1} = '{Option1}'
    {Option2} = '{Option2}'
    # 添加更多选项

class {FactoryClassName}:
    def __new__(cls, type: {EnumClassName}) -> Any:
        assert type.value in [i.value for i in {EnumClassName}]
        instance = None

        if type.value == '{Option1}':

            # instance = SomeClass(param1=value1, param2=value2)
            pass

        elif type.value == '{Option2}':

            # instance = AnotherClass(param1=value1, param2=value2)
            pass


        else:
            raise Exception('Unknown type')

        return instance

## 建造者模式

class ABCHouse():
    pass


class House(ABCHouse):
    def __init__(self):
        self.foundation = None
        self.structure = None
        self.roof = None
        self.interior = None

    def __str__(self):
        return f"House with self.foundation, self.structure, self.roof, and self.interior"

# 生成器接口
class HouseBuilder:
    def build_foundation(self):
        pass

    def build_structure(self):
        pass

    def build_roof(self):
        pass

    def build_interior(self):
        pass

    def get_house(self):
        pass

# 具体生成器
class ConcreteHouseBuilder(HouseBuilder):
    def __init__(self):
        self.house = House() # ABCHouse

    def build_foundation(self):
        self.house.foundation = "Concrete foundation"

    def build_structure(self):
        self.house.structure = "Wood and brick structure"

    def build_roof(self):
        self.house.roof = "Shingle roof"

    def build_interior(self):
        self.house.interior = "Modern interior"

    def get_house(self):
        return self.house

# 指挥者
class Director:
    def __init__(self, builder: HouseBuilder):
        self.builder = builder

    def construct_house(self):
        self.builder.build_foundation()
        self.builder.build_structure()
        self.builder.build_roof()
        self.builder.build_interior()
        return self.builder.get_house()

{执行协议等待指令：简单任务用自适应身份叠加输入处理输出结构基元执行，复杂任务分拆成简单任务交给基元构成链条执行，默认不展示输入处理输出细节但用户可以要求展示。}
'''

'''
code_snippet = """
import os
from typing import Dict, Any, Optional

class Memo(object):
    '''
    A class for managing memo data.
    '''
    def __init__(self, some_param: int = 0):
        '''
        初始化 Memo 对象。
        :param some_param: 某个参数的示例
        '''
        self.param = some_param

    def build_from_datas(self, datas: Dict[str, Any]) -> None:
        '''
        根据给定的数据字典构建 Memo 内部结构。
        :param datas: 包含构建信息的字典
        '''
        pass

    def get_info_from_tree(self, prompt: str) -> Optional[str]:
        '''
        从内部树结构中获取信息。
        :param prompt: 查询的提示字符串
        :return: 获取到的信息，如果不存在则返回 None
        '''
        return "some info"

    def eval_(self) -> str:
        '''
        这是一个长时间运行的同步操作。
        它可能会阻塞事件循环。
        '''
        return "sync eval result"

    async def aeval_(self, timeout: float = 10.0) -> str:
        '''
        这是一个长时间运行的异步操作, 作用同eval_。
        :param timeout: 超时时间
        '''
        await asyncio.sleep(1) # 模拟异步操作
        return "async eval result"

    def get_eval(self) -> str:
        '''
        获取 eval 的执行结果。
        '''
        return "result"

def standalone_function(arg1: int, arg2: str = "default"):
    '''
    这是一个独立的函数。
    '''
    pass

async def async_standalone_function():
    '''
    这是一个独立的异步函数。
    '''
    pass
"""

tree = ast.parse(code_snippet)
extractor = DefinitionExtractor()
extractor.visit(tree)

for definition in extractor.definitions:
    if definition["type"] == "class":
        print(f"Type: {definition['type']}")
        print(f"  Name: {definition['name']}")
        print(f"  Signature: {definition['signature']}")
        print(f"  Docstring: {definition['docstring']}")
        for method in definition["methods"]:
            print(f"  Method Name: {method['name']}")
            print(f"    Signature: {method['signature']}")
            print(f"    Docstring: {method['docstring']}")
            print(f"    Is Async: {method['is_async']}")
        print("-" * 20)
    else: # standalone function
        print(f"Type: {definition['type']}")
        print(f"  Name: {definition['name']}")
        print(f"  Signature: {definition['signature']}")
        print(f"  Docstring: {definition['docstring']}")
        print(f"  Is Async: {definition['is_async']}")
        print("-" * 20)


'''