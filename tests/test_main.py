
from prompt_writing_assistant.core import prompt_writer
from prompt_writing_assistant.unit import super_print
from prompt_writing_assistant.web_ import edit_server
import pytest
def test_prompt_writer():
    opinion = "将这个fastapi的接口改为post方式,"
    prompt = """

"""

    output = prompt_writer(prompt,
              opinion = opinion)

    super_print(output,"test_prompt_writer")


# 格式输出不稳定

def test_edit_server():
    """
    1 uv run python -m temp_web # 搭建起虚拟服务
    2 运行, 写入服务更新, 然后获得curl 命令, 验证是否通行
    """
    # uv run python -m temp_web
    ask = '''
# 记忆合并
class Memory_card_list_Request(BaseModel):
    memory_cards: list[str] = Field(..., description="要评分的记忆卡片内容")

@app.post("/memory_card/merge")
async def memory_card_merge_server(request: Memory_card_list_Request):
    """
    记忆卡片质量评分
    接收一个记忆卡片内容字符串，并返回其质量评分。
    """
    result = memory_card_merge(memory_cards=request.memory_cards)
    return {
        "message": "memory card merge successfully",
        "result": result
    }

帮我修改为post方法
'''
    
    out_curl = edit_server(ask,output_py = 'temp_web.py')
    super_print(out_curl,"test_edit_server_curl")

from prompt_writing_assistant.web_ import gener_post_server,gener_post_server_multi,gener_get_server



def test_get_server():
    code = '''
    def brief(self,memory_cards:list[str])->str:
        """
        数字分身介绍
        """
        return {"title":"我的数字分身标题","content":"我的数字分身简介"}
    '''
    result = gener_get_server(prompt=code)
    super_print(result,'gener_get_server')



def test_post_server():
    code = '''
    def brief(self,memory_cards:list[str])->str:
        """
        数字分身介绍
        """
        return {"title":"我的数字分身标题","content":"我的数字分身简介"}
    '''
    result = gener_post_server(prompt=code)
    super_print(result,'post_server')



def test_post_server_multi():
    code = '''
class Memo():
    def __init__(self):
        """
        初始化
        """
        
    def build_from_datas(datas:dict)->None:
        """
        构建数据
        """
        
    def get_info_from_tree(self, prompt:str)-> str:
        """
        获取信息
        """
        
    def eval_(self)-> str:
        """
        这是一个长时间运行的同步操作
        """
        
    async def aeval_(self)-> str:
        """
        这是一个长时间运行的异步操作, 作用同eval_
        """
        
    def get_eval(self)-> str:
        """
        获取eval 的执行结果
        """
    '''
    result = gener_post_server_multi(prompt=code)
    super_print(result,'post_server')



from prompt_writing_assistant.utils_search import Code_Manager,TextType


def test_Code_Manager_update():
    cm = Code_Manager()

    code = '''
from llmada.core import BianXieAdapter
bx = BianXieAdapter()
bx.model_name = "gemini-2.5-flash-preview-05-20-nothinking"
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




from prompt_writing_assistant.draw import Paper

def test_draw_paper():
    content = """我在凉山州做项目期间，曾替客户背过一次“黑锅”。那次是我们接处警的项目刚刚上线，因为项目是在凉山州做的，所以我们在西昌市驻扎。当地有两个公安局，一个是州公安局，一个是市公安局。上线当天，我们原定在州公安局开新闻发布会，市公安局这边会同步进行。但就在当天早上，我被客户叫到了市公安局，本该去参加发布会的我被临时调动。客户告诉我，市公安局的大领导、局长可能会来问些问题，让我再去给他讲讲我们的系统。于是我直接去了市公安局，没有去参加发布会。\n\n当时叫我去的是客户那边的对接人，一位指挥长。在他们开新闻发布会的时候，我们就在后面聊天，并给他讲一些系统的事情，因为我们关系都比较熟了。发布会结束后，公安局的局长说要让我调一份接警的录音出来。这事儿是因为他们下面有一个派出所警员在接警过程中，对接警员的态度不好，局长比较生气，说要调出证据去处罚他，认为他既然对自己的同志态度都不好，那对待民众的态度肯定会更不好。他只要那份录音文件，让我从我们的系统里导出来。\n\n局长要在他们全市的公安局领导面前播放这份录音，但是他们播放的设备不能直接登录我们的系统，所以我需要把录音文件导出来，再插到他们播放的电脑上去播放。我找到U盘后，因为我在那块待的时间比较长，比较熟，就想把音频文件直接导到他们的电脑上，然后播放出来。"""
    pp = Paper(content)
    pp.talk(' 帮我将上面文字中关于u盘的内容删除')
    print(pp.content)



# 测试提示词的稳定性
def test_add_basic_with_success_rate():
    """
    测试加法函数，并计算内部测试用例的通过比例。
    注意：这会使 pytest 将整个函数视为一个测试，
    而不是为每个数据组生成独立的测试报告。
    """
    test_cases = [
        (1, 2, 3),      # Pass
        (-1, 1, 0),     # Pass
        (0, 0, 0),      # Pass
        (100, 200, None), # Fail
        (-5, -3, -8)    # Pass
    ]
    
    def add(a,b):
        return a + b

    successful_assertions = 0
    total_assertions = len(test_cases)
    
    failed_cases = []

    for i, (input_a, input_b, expected_sum) in enumerate(test_cases):
        try:

            actual_sum = add(input_a, input_b) #待测试函数
            assert actual_sum == expected_sum
            successful_assertions += 1
        except AssertionError as e:
            failed_cases.append(f"Case {i+1} ({input_a}+{input_b}): FAILED. Expected {expected_sum}, Got {actual_sum}. Error: {e}")
        except Exception as e: # 捕获其他可能的错误
            failed_cases.append(f"Case {i+1} ({input_a}+{input_b}): ERROR. Input {input_a},{input_b}. Error: {e}")
            print(f"Case {i+1} ({input_a}+{input_b}): ERROR. Input {input_a},{input_b}. Error: {e}")

    success_rate = (successful_assertions / total_assertions) * 100
    print(f"\n--- Aggregated Results ---")
    print(f"Total test cases: {total_assertions}")
    print(f"Successful cases: {successful_assertions}")
    print(f"Failed cases count: {len(failed_cases)}")
    print(f"Success Rate: {success_rate:.2f}%")

    MIN_SUCCESS_RATE = 80.0 # 例如，设定80%的通过率
    assert success_rate >= MIN_SUCCESS_RATE, \
        f"Test failed: Success rate {success_rate:.2f}% is below required {MIN_SUCCESS_RATE:.2f}%." + \
        f"\nFailed cases details:\n" + "\n".join(failed_cases)


# 这是一种结构, 必须通过的结构, 
@pytest.mark.parametrize("input1, input2, input3",[
    ("我喜欢这样",1,2),
    ("我喜欢这样",1,2),
    ("我喜欢这样",1,2),
                         ])
def test_prompt_stablely(input1, input2, input3):
    print(input1,input2, input3)



















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



rule = """
'| | 维度 | 关键字段 | 分值 | 评审规则 | 要点扣分规则 | 备注 |\n|---:|:-----------------|:-------------------------------------------------------------------------------------------------------------------------|-------:|:-----------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------|:-------------------|\n| 0 | 基本信息（15分） | 姓名、性别、年龄、病历号、就诊日期 | 15 | | | 系统比对, 默认满分 |\n| 1 | 主诉（10分) | 症状+部位+时间，≤20字 | 10 | 1.含症状关键词（如“牙痛”“溃疡”）或有“口腔检查”等诉求字段2.含时间词（天/周）3.字符数≤20 | 超字数扣5分；缺症状或时间或口腔检查各扣5分 | |\n| 2 | 现病史（15分） | 发病时间、持续时间、诱因、进展、伴随症状、缓解因素 | 15 | 1.时间轴完整性（“×天前”→“持续×天”）2.诱因/缓解因素关键词 | 每缺1要素扣3分 | |\n| 3 | 口腔检查（25分） | 主诉疾病、硬组织，软组织、咬合 | 25 | 1.龋坏或其他口腔疾病关键词 2.部位体征 | 缺部位扣5分；疾病类型5分；软组织漏记扣5分 | |\n| 4 | 诊断（20分） | 明确诊断（如“左上乳磨牙深龋伴慢性牙髓炎”）或“待查”下方列出≥1个可能性最大的诊断（如“待查：1.疱疹性龈口炎？2.手足口病？”） | 20 | 检测到“待查”字样，自动检索下一行是否含“1.”“2.”等序号开头的诊断提示；若无，则视为“无进一步诊治措施”扣10分。 | 无诊断直接丙级（单否）。写“待查”但未给出可能诊断，扣10分。诊断用语模糊（如“牙齿问题”），扣5分 | |\n| 5 | 处置计划（15分） | 预约方式，治疗建议、注意事项 | 15 | 1.治疗建议 2.预约 3.注意事项 | 缺部位扣5分；疾病类型5分；软组织漏记扣5分 | |'
"""
pass_if = """
严格按照评分规则进行评价, 或者与人类基准分差在5分以内
"""

# 1
inputs = ["""
姓 名 :彭 浩 宇      科 室 :又 腔 治 疗 科
门 诊 号 :1 8 0 3 2 0 8 0 9 2
电 话 :1 8 9 2 9 3 0 8 3 7 5 性 别 :男
年 龄 :9 岁 7 月 就诊时间:2025-08-24, 10:43
主诉:全麻下行龋齿治疗术后3 年，依约复查
现病史:3 年前于我科行全麻下龋齿治疗术，有先天缺牙，无明显不适，今依约复 查既往史、个人史、家族史:不详
药物过敏史:否认药物过敏史
体 格 检 查 :8 5 见 冠 修 复 ，松 2 度 ，7 5 见 间 隙 保 持 器 ，3 4 已 萌 ， 丝 圈 位 于 3 4 颈 部 ， 未 及 松动。
轴助检查 :X线摄熙成像(又腔曲面体层成像):
初步诊断:西医诊断:1. 龋齿
处理 :无1. 建议定期复查，又腔卫生宣教。
注意事项:无
""",
"""
姓名:余宥安
科室:又腔治疗科
门诊号:221200006453 电话:13926520598
性别:男
年 龄 :3 岁 2 月 就诊时间:2 025- 08- 19, 09: 52
主诉:上前牙发黑数月 现病史:上前牙发照数月，经常性食物嵌塞，否认自发痛、夜间痛史，未予处理，今 至我科要求治疗。
既往史、个人史、家族史 :不详
药物过敏史 :否认药物过敏史 体格检查:51、61近中龋坏达牙本质浅层，色黑，余牙未见龋坏
轴助检查: 无
初 步 诊 断 : 西 医 诊 断 :1 . 5 1 、 6 1 龋 齿
处理: 暂观，半年复查，不适随诊。
注意事项:又腔健康宣教:
1. 戒夜奶，晚上刷牙后睡觉，白天喝奶后漱又
2 . 早 晚 刷 牙 ，每 次 2 - 3 分 钟 ，8 岁 前 须 由 家 长 辅 助 和 监 督 刷 牙 ， 每 天 使 用 牙 线 清 洁 牙 缝内嵌塞的食物
3.
推荐使用含氟牙膏，需控制用量:3岁前米粒大小/ 每次，3- 6 岁绿豆粒大小/ 每次 4.
每3 -6个月检查牙齿，涂氟保护剂
5.
6 岁后第 一颗恒磨牙萌出后推荐做窝沟封闭
"""
]

person_output = ["""
姓名彭浩宇(甲)维度关键字段扣分规则得分扣分原因
1.基本信息（15分）姓名、性别、年龄、病历号、就诊日期缺/错一项扣3分15/
2.主诉（10分）症状+部位+时间，≤20字或口腔检查超字数扣5分；缺症状或时间或口腔检查各扣5分10/3.现病史（15分）发病时间、持续时间、诱因、进展、伴随症状、缓解因素每缺1要素扣3分15/4.口腔检查（25分）主诉疾病、硬组织，软组织、咬合缺部位扣5分；疾病类型5分；软组织漏记扣5分20软组织漏记5.诊断（20分）明确诊断（如“左上乳磨牙深龋伴慢性牙髓炎”）或“待查”下方列出≥1个可能性最大的诊断（如“待查：1.疱疹性龈口炎？2.手足口病？”）无诊断直接丙级（单否）。写“待查”但未给出可能诊断，扣10分。诊断用语模糊（如“牙齿问题”），扣5分20/6.处置计划（15分）预约方式，治疗建议、注意事项缺治疗建议/注意事项/预约各扣5分15/总分90
""",
"""
姓名
余宥安    (甲)

维度
关键字段
扣分规则
得分
扣分原因

1. 基本信息（15分）
姓名、性别、年龄、病历号、就诊日期
缺/错一项扣3分
15
/

2. 主诉（10分）
症状+部位+时间，≤20字
或口腔检查
超字数扣5分；缺症状或时间或口腔检查各扣5分
10
/

3. 现病史（15分）
发病时间、持续时间、诱因、进展、伴随症状、缓解因素
每缺1要素扣3分
15
/

4. 口腔检查（25分）
主诉疾病、硬组织，软组织、咬合
缺部位扣5分；疾病类型5分；软组织漏记扣5分
20
软组织漏记

5. 诊断（20分）
明确诊断（如“左上乳磨牙深龋伴慢性牙髓炎”）
或“待查”下方列出 ≥1 个可能性最大的诊断（如“待查：1.疱疹性龈口炎？2.手足口病？”）
无诊断直接丙级（单否）。
写“待查”但未给出可能诊断，扣10分。
诊断用语模糊（如“牙齿问题”），扣5分
20
/

6. 处置计划（15分）
预约方式，治疗建议、注意事项
缺治疗建议/注意事项/预约各扣5分
15
/

总分
95

"""
]



# def test_evals():
#     resu = evals(rule, inputs, llm_output, person_output,pass_if)

from prompt_writing_assistant.work import node

async def test_node_adjust():
    output = await node(inputs,person_output,timeid="start",pass_if = pass_if,rule = rule,type_ = "adjust")
    print(output,'output')

# logger = Log.logger