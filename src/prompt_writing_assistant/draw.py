# 绘图板子
import json
from prompt_writing_assistant.llm import bx
from prompt_writing_assistant.unit import extract_json
import re
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
