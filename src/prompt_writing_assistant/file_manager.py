from prompt_writing_assistant.utils import get_adler32_hash, embedding_inputs
from db_help.mysql import MySQLManagerWithVersionControler
from db_help.qdrant import QdrantManager
from datetime import datetime
from enum import Enum


class TextType(Enum):
    Code = 0
    Prompt = 1


class ContentManager():
    def __init__(self,host=None, user=None, password=None, database=None):
        """
        """
        self.table_name = "content"
        self.mysql = MySQLManagerWithVersionControler(
            host = host,
            user = user,
            password = password,
            database = database,
            select = ["name", "version", "timestamp", "content", "type","embed_name_id"]
        )
        self.qdrant = QdrantManager(host = "localhost")
        self.neo = None

    
    def save_content(self,text:str,name:str,type:int | TextType)->str:
        # 数据库维护版本, 而向量库只保持最新

        # 1 存入到数据库中
        embed_name_id = get_adler32_hash(name)

        vector_list = embedding_inputs([text])
        if isinstance(type,TextType):
            type = type.value

        self.mysql.save_content(table_name=self.table_name,
                                data = {'name': name,
                                        "embed_name_id": embed_name_id,
                                        'timestamp': datetime.now(),
                                        "content":text,
                                        "type":type})

        # 2 存入到qdrant中
        self.qdrant.update(self.table_name,
                           data_list =[
                               {
                                "id":embed_name_id,
                                "vector":vector_list[0],
                                "payload":{
                                    "name": name, 
                                    'timestamp': datetime.now(),
                                    "content": text,
                                    "type":type}
                                }
                           ])
        
        
        # 考虑使用知识图 + 增加因果联系

    def save_to(self):
        #TODO 尝试另存为
        pass

    def search(self,name:str,version = None):
        result = self.mysql.get_content_by_version(
            target_name = name,
            table_name = self.table_name,
            target_version=version,
        )
        return result
    
    def similarity(self,content: str,limit: int =2):
        vector = embedding_inputs([content])[0]
        result = self.qdrant.select_by_vector(query_vector = vector,
                                collection_name = self.table_name,
                                limit = limit)
        return result


# from prompt_writing_assistant.prompt_helper import Intel, IntellectType

# from prompt_writing_assistant.utils import extract_json

# import json

# from llama_index.core.storage.docstore import BaseDocumentStore

# from llama_index.core import VectorStoreIndex,StorageContext

# intel = Intel(host = "127.0.0.1",user = 'root',password = "1234",database = "prompts")

# # 我希望做一个分流器, 将输入的内容分类为以下几类:
# # 1 经验心得
# # 2 提示词存储
# # 3 代码片段
# # 4 备忘录

# @intel.intellect(IntellectType.inference,
#                  prompt_id = "db_help_001",
#                  table_name ="prompts_data",
#                  demand="""
# 包名不要使用大写
# 这类的应该属于经验心得, 
# """)
# def diverter(content,wang):
#     # 根据需求修改程序框图
#     content = json.loads(extract_json(content))
#     # print(wang)
#     return content,wang

# diverter(demo1,"天天形式")

