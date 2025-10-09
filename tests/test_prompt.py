from prompt_writing_assistant.file_manager import ContentManager, TextType
from prompt_writing_assistant.prompt_helper import IntellectType,Intel, Base_Evals
from prompt_writing_assistant.prompt_helper import Base_Evals
from pydantic import BaseModel

from dotenv import load_dotenv
load_dotenv(".env", override=True)
import pytest

# 内容池
class Test_ContentManager():

    @pytest.fixture
    def content_manager(self):
        return ContentManager()
    
    def test_save_content(self, content_manager):
        result =content_manager.save_content(
            name = "数据库--001",
            type = TextType.Tips,
            text = '''
```python
如何使用数据库同步数据
''',
        )
        print(result,'result')

    def test_save_content_auto(self, content_manager):
        result =content_manager.save_content_auto(
            text = '''

''')

        print(result,'result')

    def test_search(self, content_manager):
        result = content_manager.search(
            name = "装饰器-struct",
        )
        print("=="*20)
        print(result["content"])
        print("=="*20)

    def test_similarity(self,content_manager):
        result = content_manager.similarity(
            content = "每日资讯还是要具备的, 你要知道的内容和111",
            limit = 2
        )
        print(result[0].payload.get('content'),'result')
        print(type(result[0]),'result22')


######## prompt_helper ##########
# 尝试构建主从数据库的防御结构


# 测试 

class Test_Intel():

    @pytest.fixture
    def intels(self):
        intels = Intel()
        return intels
    

    def test_get_prompts(self,intels):
        result = intels.get_prompts_from_sql(
            table_name = "prompts_data",
            prompt_id = "1231231")
        print(result)

    def test_get_prompts_version(self,intels):
        result = intels.get_prompts_from_sql(
            table_name = "prompts_data",
            prompt_id = "1231231",
            version="1.1"
            )
        print(result)

    def test_intellect(self,intels):
        import json
        from pydantic import BaseModel
        from prompt_writing_assistant.utils import extract_json
        
        @intels.intellect(IntellectType.train,
                        "数字人生王者001",
                        table_name ="prompts_data",
                        demand = "改为使用$符号")
        def prompts(input):
            # 后处理, 也可以编写后处理的逻辑 extract_json 等
            # 也可以使用pydnatic 做校验
            input = json.loads(extract_json(input))
            # class Output(BaseModel):
            #     name : str
            #     type : int
            #     note : str
            # Output(**input)
            return input

        result = prompts("你好, 我的电话号码是12343213123, 身份证是2454532345")
        print(result,'result')

    def test_intellect_2(self,intels):
        import json
        from pydantic import BaseModel
        from prompt_writing_assistant.utils import extract_json

        @intels.intellect_2(IntellectType.train,
                        "数字人生王者002",
                        table_name ="prompts_data",
                        demand = "改为使用$符号")
        def prompts(input):
            # 后处理, 也可以编写后处理的逻辑 extract_json 等
            # 也可以使用pydnatic 做校验
            input = json.loads(extract_json(input))
            # class Output(BaseModel):
            #     name : str
            #     type : int
            #     note : str
            # Output(**input)
            return input

        result = prompts(input = {"text":"你好, 我的电话号码是12343213123, 身份证是2454532345"}
                        )
        print(result,'result')

    def test_intellect_3(self,intels):
        import json
        from pydantic import BaseModel
        from prompt_writing_assistant.utils import extract_json

        test_cases = [
            (1, 2, 3),      # Pass
            (-1, 1, 0),     # Pass
            (0, 0, 0),      # Pass
            (100, 200, None), # Fail
            (-5, -3, -8)    # Pass
        ]
        intels.intellect_3(test_cases = test_cases,
                           prompt_id = "测试11",
                           )

        def prompts(input):
            # 后处理, 也可以编写后处理的逻辑 extract_json 等
            # 也可以使用pydnatic 做校验
            input = json.loads(extract_json(input))
            # class Output(BaseModel):
            #     name : str
            #     type : int
            #     note : str
            # Output(**input)
            return input

        result = prompts(input = {"text":"你好, 我的电话号码是12343213123, 身份证是2454532345"}
                        )
        print(result,'result')

    def test_prompt_finetune_to_sql():
        # prompt_finetune_to_sql
        intels = Intel()
        demand = "将这个fastapi的接口改为post方式,"
        result = intels.prompt_finetune_to_sql(
            prompt_id = "数字人生王者001",
            table_name ="prompts_data",
            demand=demand)
        
        print(result)



# Evals

class AEvals(Base_Evals):
    def __init__(self):
        self.MIN_SUCCESS_RATE = 00.0 # 这里定义通过阈值, 高于该比例则通过

    def _assert_eval_function(self,params):

        # prompt running
        assert params[0] + params[1] == params[-1]
        self.person_evals(params[:-1],params[-1],params[-1])
        self.rule_evals(params[-1],params[-1])
        print(params,'params111')

    def rule_evals(self):
        return super().rule_evals()
    
    def global_evals(self):
        pass

    def local_evals(self):
        pass
        

def test_evals():
    aeval = AEvals()
    test_cases = [
            (1, 2, 3),      # Pass
            (-1, 1, 0),     # Pass
            (0, 0, 0),      # Pass
            (100, 200, None), # Fail
            (-5, -3, -8)    # Pass
        ]
    aeval.get_success_rate(test_cases)
