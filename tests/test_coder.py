
from prompt_writing_assistant.utils_search import Code_Manager,TextType


def test_Code_Manager_update():
    cm = Code_Manager()

    code = '''

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


    '''

    cm.update(text=code,
              type = TextType.Code
              )

def test_Code_Manager_update_prompt():
    cm = Code_Manager()

    text = '''


    '''

    cm.update(text=text,
              type = TextType.Prompt
              )

def test_Code_Manager():
    cm = Code_Manager()

    prompt = """
你好    
"""
    result = cm.search(prompt)



# 以后最好两头跑
# 1 做一个收集底层工具 的仓库
# 2 做一个从上之下的仓库
# 3 两者配合

def test__():

    # # 1. 导入必要的类和外部 LLM 调用函数
    from llm_councilz.meeting.core import MeetingOrganizer # 暂时使用模拟函数

    # 2. 创建会议组织者
    organizer = MeetingOrganizer()

    # 3. 设置外部 LLM 调用函数 (!!! 重要步骤，连接框架和您的能力)
    # organizer.set_llm_caller(call_your_llm_api) # 在实际使用时，取消注释并替换

    # 4. 添加参与者 (LLM)
    organizer.add_participant(name="专家A", model_name="gpt-4o")
    organizer.add_participant(name="专家B", model_name="gpt-4.1")
    # organizer.add_participant(name="专家C", model_name="model-gamma")

    # 5. 设置会议主题
    topic = "制定一个针对中小型企业的数字化转型方案"
    background = "考虑到成本和实施难度，方案应侧重于易于落地和快速见效。"
    organizer.set_topic(topic, background)

    # 6. 运行一轮简单的会议
    organizer.run_simple_round()

    # 7. 获取讨论历史和简单摘要
    organizer.display_history() # 打印格式化历史

    simple_summary = organizer.get_simple_summary()
    print("\nGenerated Simple Summary:")
    print(simple_summary)

from program_writing_assistant.core__2 import EditCode


def test_edit():
    py_file = 'tests/temp_file/file.py'
    with open(py_file,'w') as f:
        f.write("print('hello world')")
    ec = EditCode(py_file)
    ec.edit('改成 你好 世界')
    with open(py_file,'r') as f:
        tt = f.read()
    assert "你好" in tt
    




#############