'''
Author: 823042332@qq.com 823042332@qq.com
Date: 2025-08-27 20:40:34
LastEditors: 823042332@qq.com 823042332@qq.com
LastEditTime: 2025-08-28 18:07:52
FilePath: /prompt_writing_assistant/src/prompt_writing_assistant/work.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''

from prompt_writing_assistant.unit import load_inpackage_file
from prompt_writing_assistant.unit import extract_json
from prompt_writing_assistant.prompts import evals_prompt
from prompt_writing_assistant.llm import bx
from llama_index.core import PromptTemplate
import json


import asyncio
import json
import concurrent.futures # 用于线程池，如果 bx.product 是同步的


"""

async def awrite_chapter(info,master = "马恪",material_all = "", number_words = 3000):
    created_material =""
    try:
        model_name = "gemini-2.5-flash-preview-05-20-nothinking"
        bx.model_pool.append(model_name)
        bx.set_model(model_name=model_name)
        material_prompt = prompt_get_infos.format(material= material_all,frame = json.dumps(aa), requirements = json.dumps(info))
        material = await asyncio.to_thread(bx.product, material_prompt) 
        # create_prompt = prompt_get_create.format(material= material_all, requirements = json.dumps(info))
        # created_material = await asyncio.to_thread(bx.product, create_prompt) 
        model_name = "gpt-5"
        bx.model_pool.append(model_name)
        bx.set_model(model_name=model_name)
        words = prompt_base.format(master = master, chapter = f'{info.get("chapter_number")} {info.get("title")}', topic = info.get("topic"),
                               number_words = number_words,material = material ,reference = "",port_chapter_summery = '' )
        article = await asyncio.to_thread(bx.product, words) # Python 3.9+
        return extract_python_code(article), material, created_material
    except Exception as e:
        print(f"Error processing chapter {info.get('chapter_number')}: {e}")
        return None 
    

"""

def generate(input_:list[str],prompt:str)->list[str]:
    prompt = PromptTemplate(prompt)
    result = bx.product(prompt.format(bl = input_))
    return result

async def _agenerate_single(prompt_format:str) -> str:
    try:
        model_name = "gemini-2.5-flash-preview-05-20-nothinking"
        bx.model_pool.append(model_name)
        bx.set_model(model_name=model_name)
        result = await asyncio.to_thread(bx.product, prompt_format)
        return result
    except Exception as e:
        print(f"Error processing from {e}")
        return None 

async def agenerate(input_:list[str],prompt:str)->list[str]:
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=20) # 根据需要调整并发度

    prompt = PromptTemplate(prompt)
    tasks = []
    for inp in input_:
        print(f"Creating task")
        prompt_format = prompt.format(bl = inp)
        tasks.append(_agenerate_single(prompt_format))

    results = await asyncio.gather(*tasks, return_exceptions=False) 

    executor.shutdown(wait=True)
    return results


def judge_eval_result(eval_result:str)->bool:
    if eval_result == "":
        return True

    # 进行评价逻辑
    return True

def adjust(eval_result:str, prompt:str, eval_reason:str ) -> str:
    # adjust_v1
    prompt = prompt
    return prompt

def evals(_input:list[str], llm_output:list[str], person_output:list[str],rule:str, pass_if:str) -> "eval_result-str, eval_reason-str":
    result = bx.product(evals_prompt.format(评分规则 =rule ,
    输入案例 = _input ,大模型生成内容 = llm_output,人类基准 = person_output,
                                     通过条件 = pass_if))
                                    #  eval_result,eval_reason

    """
    "passes": "<是否通过, True or False>",
    "reason": "<如果不通过, 基于驳回理由>",
    "suggestions_on_revision": 
    """
    result = json.loads(extract_json(result))
    return result.get('passes'), result.get('suggestions_on_revision')

    

async def node(input_:list[str],person_output:list[str] = [],timeid  = "start", pass_if:str = "",rule:str = "",type_ = "adjust")->str:
    """
    type : adjust(评价且修改) wait(稳定模式, 评价但不修改)   product(正常使用, 不评价也不修改)
    """
    # get_target_prompt
    target_prompt_file = "tests/temp_file/prompt_{timeid}.txt"
    prompt_file = target_prompt_file.format(timeid = timeid)
    with open(prompt_file,'r') as f:
        prompt = f.read()

    if type_ in ['adjust']:
        assert person_output != []
        for _ in range(3):
            # generate
            output_:list[str] = await agenerate(input_ = input_,prompt = prompt)
            # evals 
            eval_result,eval_reason = evals(input_,output_,person_output,rule,pass_if)
            if not judge_eval_result(eval_result):
                # adjust_v1 聊天情况只有1轮 可能要保留历史记录
                new_prompt:str = adjust(eval_result, prompt, eval_reason)
                prompt:str = new_prompt
            else:
                # save_prompt 
                with open(target_prompt_file.format(timeid = "1"),'w') as f:
                    f.write(prompt)
                return output_
        return "未能通过"

    elif type_ in ['wait']:
        output_:list[str] = await agenerate(input_ = input_,prompt = prompt)
        eval_result,eval_reason = evals(input_,output_,person_output,rule,pass_if)
        print(eval_result,"eval_result")
        print(eval_reason,"eval_reason")
        return output_

    elif type_ in ['product']:
        # generate
        output_:list[str] = await agenerate(input_ = input_,prompt = prompt)
        return output_

    else:
        return "type error"


