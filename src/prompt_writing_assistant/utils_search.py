
from typing import Any, List
from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.embeddings import BaseEmbedding
from volcenginesdkarkruntime import Ark
from prompt_writing_assistant.unit import super_print

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

