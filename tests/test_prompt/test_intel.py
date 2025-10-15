from prompt_writing_assistant.prompt_helper import IntellectType,Intel
from dotenv import load_dotenv
load_dotenv(".env", override=True)
import pytest



######## prompt_helper ##########
# 尝试构建主从数据库的防御结构


# 测试 

class Test_Intel():

    @pytest.fixture
    def intels(self):
        intels = Intel()
        return intels
    
    def test_save_prompts(self,intels):

        result = intels.save_prompt_by_sql(
            prompt_id = "fsdf",
            new_prompt = "112334",
            input_data = "",
                )
        print(result)

    def test_push_info_by_use(self,intels):

        result = intels.push_info_by_use(
            prompt_id = "fsdf",
            demand = "这是一个不错的角度",
                )
        print(result)

        # save_prompt_by_control
    
    def test_get_prompts(self,intels):
        result = intels.get_prompts_from_sql(
            prompt_id = "intel_summary")
        print(result)

    def test_get_prompts_version(self,intels):
        result = intels.get_prompts_from_sql(
            prompt_id = "数字人生王者001",
            version="1.1"
            )
        print(result)

        

    def test_intellect(self,intels):
        import json
        from pydantic import BaseModel
        from prompt_writing_assistant.utils import extract_
        
        @intels.intellect(IntellectType.train,
                        "数字人生王者001",
                        table_name ="prompts_data",
                        demand = "改为使用$符号")
        def prompts(input):
            # 后处理, 也可以编写后处理的逻辑 extract_json 等
            # 也可以使用pydnatic 做校验

            # input = json.loads(extract_(input))

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

        @intels.intellect_2(IntellectType.inference,
                        "数字人生王者002",
                        demand = "改为使用$符号")
        def prompts(input):
            # 后处理, 也可以编写后处理的逻辑 extract_json 等
            # 也可以使用pydnatic 做校验
            # input = json.loads(extract_(input))
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

    def test_intellect_3_11(self,intels):

        result = intels.intellect_3_11(input =  "你好",
                              prompt_id = "fsdf"
                           )
 


        print(result,'result')

    def test_prompt_finetune_to_sql(self,intels):

        demand = "将这个fastapi的接口改为post方式,"
        result = intels.prompt_finetune_to_sql(
            prompt_id = "数字人生王者001",
            demand=demand)
        
        print(result)

