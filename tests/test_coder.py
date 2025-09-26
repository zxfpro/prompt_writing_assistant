import pytest
from prompt_writing_assistant.coder import Code_Manager, TextType


# UPDATE CODE
def test_Code_Manager_update():
    cm = Code_Manager()

    code = '''

    @retry(
        wait=wait_exponential(
            multiplier=1, min=1, max=10
        ),  # 首次等待1秒，指数增长，最大10秒
        stop=stop_after_attempt(3),  # 最多尝试5次 (1次初始请求 + 4次重试)
        retry=(
            retry_if_exception_type(httpx.RequestError)  # 匹配网络错误 (连接超时, DNS, SSL等)
            | retry_if_exception_type(httpx.HTTPStatusError)  # 匹配HTTP状态码错误
        ),
        before_sleep=before_sleep_log(logger, logging.INFO),  # 每次重试前打印日志
        reraise=True,  # 如果所有重试都失败，重新抛出最后一个异常
    )
    async def arequest(self, params: dict) -> dict:
        """
        # 修改后的异步请求函数
        简单对话：直接调用 OpenAI API 并返回完整响应 (异步版本)
        """
        api_base = self.api_base
        logger.info("Async request running")
        try:
            time1 = time.time()
            # 使用 httpx.AsyncClient 进行异步请求
            async with httpx.AsyncClient(headers=self.headers) as client:
                response = await client.post(
                    api_base, json=params, timeout=60.0
                )  # 加上timeout是个好习惯
                response.raise_for_status()  # 检查HTTP状态码，如果不是2xx，会抛出异常

            time2 = time.time()
            logger.debug(f"Request took: {time2 - time1:.4f} seconds")

            return response.json()

        except httpx.HTTPStatusError as e:
            logger.error(
                f"HTTP error occurred: {e.response.status_code} - {e.response.text}"
            )
            # 这里不要重新抛出新的Exception类型，而是重新抛出原始异常，或者不捕获让tenacity处理
            # 方式一：直接 raise e，tenacity 会捕捉到 httpx.HTTPStatusError
            raise
        except httpx.RequestError as e:
            logger.error(f"An error occurred while requesting {e.request.url!r}: {e}")
            raise 
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")
            raise Exception(f"API request failed: {e}") from e

    注意raise 的方式
    '''

    cm.update(text=code,
              type = TextType.Code
              )
# UPDATE PROMPT
def test_Code_Manager_update_prompt():
    cm = Code_Manager()
    text = '''
发现问题了, 在pytest 中使用插件执行时, 日志就会失效
'''
    text = '''

    '''

    cm.update(text=text,
              type = TextType.Prompt
              )

def test_Code_Manager():
    cm = Code_Manager()

    prompt = """
with contents() as f:
 
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

# from program_writing_assistant.core__2 import EditCode


# def test_edit():
#     py_file = 'tests/temp_file/file.py'
#     with open(py_file,'w') as f:
#         f.write("print('hello world')")
#     ec = EditCode(py_file)
#     ec.edit('改成 你好 世界')
#     with open(py_file,'r') as f:
#         tt = f.read()
#     assert "你好" in tt
    




#############