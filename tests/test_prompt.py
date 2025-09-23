# 测试 
import pytest
from prompt_writing_assistant.prompt_helper import intellect,IntellectType,aintellect
from prompt_writing_assistant.prompt_helper import prompt_finetune_to_sql, get_prompts_from_sql, save_prompt_by_sql
from prompt_writing_assistant.utils import super_print
from prompt_writing_assistant.prompt_helper import Base_Evals
from dotenv import load_dotenv


def test_get_prompts_from_sql():
    # get_prompts_from_sql
    result = get_prompts_from_sql(prompt_id = "123134123")
    print(result)



def test_save_prompt_by_sql():
    new_prompt = """
一个测试的提示词
"""
    save_prompt_by_sql(prompt_id = "123134123",
                       new_prompt = new_prompt)


def test_prompt_finetune_to_sql():
    # prompt_finetune_to_sql
    opinion = "将这个fastapi的接口改为post方式,"
    prompt_finetune_to_sql(prompt_id = "123134123",
                           opinion=opinion)


# 半自动编写/优化提示词
def test_intellect():
    @intellect(IntellectType.train,"1231231","改为使用$符号")
    def prompts(input_):
        # 可以直接输出, 也可以编写后处理的逻辑 extract_json 等
        return input_

    result = prompts("你好, 我的电话号码是12343213123, 身份证是2454532345")
    print(result,'result')


# 半自动编写/优化提示词
async def test_aintellect():
    @aintellect(IntellectType.train,"1231231","改为使用$符号")
    def prompts(input_):
        # 可以直接输出, 也可以编写后处理的逻辑 extract_json 等
        return input_

    result = await prompts("你好, 我的电话号码是12343213123, 身份证是2454532345")
    print(result,'result')

# Evals

class AEvals(Base_Evals):
    def __init__(self):
        self.MIN_SUCCESS_RATE = 00.0 # 这里定义通过阈值, 高于该比例则通过

    def _assert_eval_function(self,params):
        print(params,'params')


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



#########--时空光年-数字人生--#############
"""
0098 数字分身简介
0099 数字分身性格提取
0100 # 数字分身信息脱敏
"""

load_dotenv()
import os

def test_work():
    host = os.getenv("MySQL_DB_HOST")
    user = os.getenv("MySQL_DB_USER")
    passward = os.getenv("MySQL_DB_PASSWORD")
    name = os.getenv("MySQL_DB_NAME")
    table = os.getenv("MySQL_DB_Table_Name")
    print(host)
    print()
    print(user)
    print()
    print(passward)
    print()
    print(name)
    print()
    print(table)
    print()

from datetime import datetime
from db_help.mysql import MySQLManager
def test_db_help():
    
    db_manager = MySQLManager(host=os.getenv("MySQL_DB_HOST"),
                 user=os.getenv("MySQL_DB_USER"),
                 password=os.getenv("MySQL_DB_PASSWORD"),
                 database=os.getenv("MySQL_DB_NAME")
                 )
    
    table_name = "llm_prompt"
    user1_id = db_manager.insert(table_name, {'prompt_id': '1234134', 'version': '1.0', 'timestamp': datetime.now(),"prompt":"你好"})
    user2_id = db_manager.insert(table_name, {'prompt_id': '2345234234', 'version': '1.1', 'timestamp': datetime.now(),"prompt":"你好2223"})
    db_manager.close()



memory_cards = ["我出生在东北辽宁葫芦岛下面的一个小村庄。小时候，那里的生活比较简单，人们日出而作，日落而息，生活节奏非常有规律，也非常美好。当时我们都是山里的野孩子，没有什么特别的兴趣爱好，就在山里各种疯跑。我小时候特别喜欢晚上看星星，那时的夜晚星星非常多，真的是那种突然就能看到漫天繁星的感觉。",
                  "在我高中的时候，我对天文物理和理论物理就非常感兴趣。我当时高中是在我们县城里读的，资源没有那么丰富，我们所有的精力都放在学科的学习上。",
                  "离开泸沽湖之后，我和我的义工搭子一起去了虎跳峡徒步。在徒步过程中，我们还看到了日照金山。当时我们徒步到半山腰的一个临时扎点时，正好看到这个美景，非常激动地拍了照。能在徒步途中遇到这样的美景，确实让人兴奋不已。在虎跳峡徒步之后，我们在丽江休整了一天，然后就各自前往下一个目的地了。",
                  "我第一份职业是产品经理，当时在国内一家做专网通信的龙头企业。目前我是一位AI产品经理。",
                  "我记得在我小的时候，大概是在我三四年级的时候，我和我的小伙伴一块去上山抓蝎子。小时候那会儿都比较皮，到处跑。我在山上去翻蝎子的时候，遇到了一条蛇！我和它对视过后，我就慌忙地转身逃跑。在我刚转身逃跑的时候，我看见山上的小路上有一个树枝倒下来了，倒在我的前面，然后我就跳了起来。在我跳起来的时候，那条蛇就在我的脚底下钻过去了，当时真的吓得我很害怕。如果不是那个树枝的话，我可能就被咬了。而且那条蛇，我记得它是三角脑袋，枯草色的，看着像是有毒的样子。之后我对山上有些阴影了，都不怎么去上山抓蝎子了。我现在上山还是会拿根棍子，到处敲一敲！说实话，真的有阴影。",
                  "我在凉山州做项目期间，曾替客户背过一次“黑锅”。那次是我们接处警的项目刚刚上线，因为项目是在凉山州做的，所以我们在西昌市驻扎。当地有两个公安局，一个是州公安局，一个是市公安局。上线当天，我们原定在州公安局开新闻发布会，市公安局这边会同步进行。但就在当天早上，我被客户叫到了市公安局，本该去参加发布会的我被临时调动。客户告诉我，市公安局的大领导、局长可能会来问些问题，让我再去给他讲讲我们的系统。于是我直接去了市公安局，没有去参加发布会。\n\n当时叫我去的是客户那边的对接人，一位指挥长。在他们开新闻发布会的时候，我们就在后面聊天，并给他讲一些系统的事情，因为我们关系都比较熟了。发布会结束后，公安局的局长说要让我调一份接警的录音出来。这事儿是因为他们下面有一个派出所警员在接警过程中，对接警员的态度不好，局长比较生气，说要调出证据去处罚他，认为他既然对自己的同志态度都不好，那对待民众的态度肯定会更不好。他只要那份录音文件，让我从我们的系统里导出来。\n\n局长要在他们全市的公安局领导面前播放这份录音，但是他们播放的设备不能直接登录我们的系统，所以我需要把录音文件导出来，再插到他们播放的电脑上去播放。我找到U盘后，因为我在那块待的时间比较长，比较熟，就想把音频文件直接导到他们的电脑上，然后播放出来。结果，他们的电脑比较老旧，U盘识别不了，没有驱动，导致播放不了。我尝试了各种方法也不行。就在这时，局长本来就在气头上，他更生气了。他一开始以为我是他们局内的人，是信息部的，就冲我发了脾气，说我办事不力，问我是哪个部门的。我告诉他，我是这个系统公司的人，不是他们公安局的人。\n\n局长当时已经放出话来，他说我们的系统做的不好，不能直接播放录音，但其实是能播放的，只不过是他们自己内部系统的一些问题。我想解释，但局长没有给我解释的机会，他就在那里一直守着，在所有公安局的人面前指着我的鼻子数落我，还让我把我的领导叫来。他甚至说，如果系统做得不好，可以不采用我们的系统。我当时内心并没有太多感受，因为我知道这不是我的问题，我只是替他们背了一个锅而已。后来他们的人接手了，开始处理他们电脑的问题。局长也发现是他们的人一直在处理问题，也意识到了是他们自己的问题，跟我们没关系。但他还是把我们的领导叫来了。在这个过程中，我们客户，也就是公安局的那个主任，还在旁边小声安慰我说：“没事没事，领导发完火就过去了。”其实我内心是不在乎的。\n\n我的领导也很懵，因为他们正在那边开新闻发布会，突然把我的领导，还有公安局和我对接的那些负责人都叫来了。他们以为出了什么问题，大领导发大火。他们比较懵，但是叫来之后，局长也没有说什么，说了一些无关紧要的东西，做了一些指示。就这样，我默默地背了一个锅。这件事情对我的工作选择没有什么影响，因为我知道这不是我的问题，领导们也知道我只是背了一个锅，这事就这么过去了。能被局长兼副市长指着鼻子骂，对我来说也是一种特殊的体验"]

# 半自动编写/优化提示词

def test_0098():
    # 098 数字分身简介
    demand = """
帮我提取这个人的简介, 100字左右, 以以下格式输出
    ```json
    {"title":"我的数字分身标题","content":"我的数字分身简介"}
    ```
"""
    @intellect(IntellectType.inference,"0098",demand = demand,table_name = "llm_prompt")
    def prompts(input_):
        # 可以直接输出, 也可以编写后处理的逻辑 extract_json 等
        return input_

    result = prompts("\n".join(memory_cards))
    print(result,'result')

def test_0099():
    # 099 数字分身性格提取
    demand = """
"我会给你一段题主的记忆描述, 你来帮我提取这个人的MBTI性格特征"
"""
    @intellect(IntellectType.inference,"0099",demand = demand,table_name = "llm_prompt")
    def prompts(input_):
        # 可以直接输出, 也可以编写后处理的逻辑 extract_json 等
        return input_

    result = prompts("\n".join(memory_cards))
    print(result,'result')

def test_0100():
    # 数字分身信息脱敏
    demand = "我会给你一段文本, 我希望你帮我去除掉其中的数字 使用XXX代替"
    @intellect(IntellectType.inference,"0100",demand = demand,table_name = "llm_prompt")
    def prompts(input_):
        # 可以直接输出, 也可以编写后处理的逻辑 extract_json 等
        return input_

    result = prompts("你好, 我的电话号码是12343213123, 身份证是2454532345")
    print(result,'result')
