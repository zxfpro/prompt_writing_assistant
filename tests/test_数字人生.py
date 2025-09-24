
import pytest
from prompt_writing_assistant.prompt_helper import intellect,IntellectType
from prompt_writing_assistant.utils import super_print
from prompt_writing_assistant.utils import extract_article,extract_json

from dotenv import load_dotenv
load_dotenv()


memory_cards = ["我出生在东北辽宁葫芦岛下面的一个小村庄。小时候，那里的生活比较简单，人们日出而作，日落而息，生活节奏非常有规律，也非常美好。当时我们都是山里的野孩子，没有什么特别的兴趣爱好，就在山里各种疯跑。我小时候特别喜欢晚上看星星，那时的夜晚星星非常多，真的是那种突然就能看到漫天繁星的感觉。",
                  "在我高中的时候，我对天文物理和理论物理就非常感兴趣。我当时高中是在我们县城里读的，资源没有那么丰富，我们所有的精力都放在学科的学习上。",
                  "离开泸沽湖之后，我和我的义工搭子一起去了虎跳峡徒步。在徒步过程中，我们还看到了日照金山。当时我们徒步到半山腰的一个临时扎点时，正好看到这个美景，非常激动地拍了照。能在徒步途中遇到这样的美景，确实让人兴奋不已。在虎跳峡徒步之后，我们在丽江休整了一天，然后就各自前往下一个目的地了。",
                  "我第一份职业是产品经理，当时在国内一家做专网通信的龙头企业。目前我是一位AI产品经理。",
                  "我记得在我小的时候，大概是在我三四年级的时候，我和我的小伙伴一块去上山抓蝎子。小时候那会儿都比较皮，到处跑。我在山上去翻蝎子的时候，遇到了一条蛇！我和它对视过后，我就慌忙地转身逃跑。在我刚转身逃跑的时候，我看见山上的小路上有一个树枝倒下来了，倒在我的前面，然后我就跳了起来。在我跳起来的时候，那条蛇就在我的脚底下钻过去了，当时真的吓得我很害怕。如果不是那个树枝的话，我可能就被咬了。而且那条蛇，我记得它是三角脑袋，枯草色的，看着像是有毒的样子。之后我对山上有些阴影了，都不怎么去上山抓蝎子了。我现在上山还是会拿根棍子，到处敲一敲！说实话，真的有阴影。",
                  "我在凉山州做项目期间，曾替客户背过一次“黑锅”。那次是我们接处警的项目刚刚上线，因为项目是在凉山州做的，所以我们在西昌市驻扎。当地有两个公安局，一个是州公安局，一个是市公安局。上线当天，我们原定在州公安局开新闻发布会，市公安局这边会同步进行。但就在当天早上，我被客户叫到了市公安局，本该去参加发布会的我被临时调动。客户告诉我，市公安局的大领导、局长可能会来问些问题，让我再去给他讲讲我们的系统。于是我直接去了市公安局，没有去参加发布会。\n\n当时叫我去的是客户那边的对接人，一位指挥长。在他们开新闻发布会的时候，我们就在后面聊天，并给他讲一些系统的事情，因为我们关系都比较熟了。发布会结束后，公安局的局长说要让我调一份接警的录音出来。这事儿是因为他们下面有一个派出所警员在接警过程中，对接警员的态度不好，局长比较生气，说要调出证据去处罚他，认为他既然对自己的同志态度都不好，那对待民众的态度肯定会更不好。他只要那份录音文件，让我从我们的系统里导出来。\n\n局长要在他们全市的公安局领导面前播放这份录音，但是他们播放的设备不能直接登录我们的系统，所以我需要把录音文件导出来，再插到他们播放的电脑上去播放。我找到U盘后，因为我在那块待的时间比较长，比较熟，就想把音频文件直接导到他们的电脑上，然后播放出来。结果，他们的电脑比较老旧，U盘识别不了，没有驱动，导致播放不了。我尝试了各种方法也不行。就在这时，局长本来就在气头上，他更生气了。他一开始以为我是他们局内的人，是信息部的，就冲我发了脾气，说我办事不力，问我是哪个部门的。我告诉他，我是这个系统公司的人，不是他们公安局的人。\n\n局长当时已经放出话来，他说我们的系统做的不好，不能直接播放录音，但其实是能播放的，只不过是他们自己内部系统的一些问题。我想解释，但局长没有给我解释的机会，他就在那里一直守着，在所有公安局的人面前指着我的鼻子数落我，还让我把我的领导叫来。他甚至说，如果系统做得不好，可以不采用我们的系统。我当时内心并没有太多感受，因为我知道这不是我的问题，我只是替他们背了一个锅而已。后来他们的人接手了，开始处理他们电脑的问题。局长也发现是他们的人一直在处理问题，也意识到了是他们自己的问题，跟我们没关系。但他还是把我们的领导叫来了。在这个过程中，我们客户，也就是公安局的那个主任，还在旁边小声安慰我说：“没事没事，领导发完火就过去了。”其实我内心是不在乎的。\n\n我的领导也很懵，因为他们正在那边开新闻发布会，突然把我的领导，还有公安局和我对接的那些负责人都叫来了。他们以为出了什么问题，大领导发大火。他们比较懵，但是叫来之后，局长也没有说什么，说了一些无关紧要的东西，做了一些指示。就这样，我默默地背了一个锅。这件事情对我的工作选择没有什么影响，因为我知道这不是我的问题，领导们也知道我只是背了一个锅，这事就这么过去了。能被局长兼副市长指着鼻子骂，对我来说也是一种特殊的体验"]


chat_history = 'ai:柴东升，在凉山州做项目期间有什么特别的经历吗？\nhuman:有一次在项目现场，我替客户背了黑锅。\nai:哦？是怎样的情况呢？愿意和我详细说说吗？\nhuman:这个说起来有点花长了。\nai:人生故事总是值得慢慢品味的，不着急，您慢慢说。\nhuman:那一次是我们那个接触景的项目刚刚上线，然后因为我们是在梁山州做的项目。 那我们是在他的西昌市去那个柱上，然后他那里面是有两个公安局，一个是州公安局，一个是市公安局。 然后上线当天我们是就准备开一个新媒发新闻发。 发布会的和公安局的领导们一起，然后他们是在州公安局进行开，然后市公安局这边会有一个同步，然后， 但在当天早上，我被客户叫到了。本来我也是要到周公安局去参加这个发布会的。嗯，然后我临时被客户调到了市公安局。 客户说，公安局的大领导、局长可能会来问。 点问题让我再去给他讲一讲他的那个咱们那个我们那个系统，然后我就直接去到了市公安局，没有去那个我们的发布会。\nai:当时的情况听起来还挺紧急的，您被临时调去市公安局后，又发生了什么呢？\nhuman:然后因为当时叫我去的是嗯，客户的那边，我们的对接人吗？是一个指挥长，然后在他们开发布会的过程中，我们就在后面。 唠嗑，然后并给他讲一些系统，因为我们关系都比较熟了。然后再看看发布会之后，局长。 公安局去找，说要让我掉一份。嗯，接警的录音出来。嗯。 这个是因为他们下面有一个警员在接警的过程中对接警员的态度不好局长比较生气说调出来证据 去那个处罚的说，他既然对自己的同志态度都这么不好，那对待民众又会态度指定会更不好。 他只要那份录音文件让我，再让我在我们系统里给导出来。\nai:原来如此，那您之后成功帮局长导出那份录音文件了吗？\nhuman:是的，我因为他们那个局长要在他们全市的全公安局的领导面前把这个录音播出来。 但是他们播放的那个设备是不能直接登我们这个系统的，所以说我需要把这种 u 盘，把这个录音给找出来。 倒出来之后，然后再插到他们那个播放的电脑上去播放。嗯，我倒出来之后就嗯。\nai:您这么快就找到了录音文件，真是厉害。之后在播放过程中没有出现什么问题吧？\nhuman:是的，出了问题，嗯，我找到 u 盘之后，然后我看也没有人接手，我就去，因为我在那块待的时间比较长，比较熟。 我就想把那个音频文件直接倒到他们的电脑上，然后我给拨出来，结果他们那个电脑比较老了。 u 盘识别不了，那有驱动。 就导致他们那播放不了，然后我捅破了以后一下。 尝试这些方法也不行，就在这会儿，因为那个局长本来就在气头上，他就生气了。他一开始以为我是他们局内的人。 那个是他们信息部的，然后就冲我发了脾气，说我办事不力，然后问我是哪个部门。 是要出发。 那个时候说话就问我是哪个部门的。 然后我说我是这个系统的。 公司的人，不是他们公安局的人。\nai:哎呀，当时局长发脾气，您一定很委屈吧。后来呢，他们有没有因为这个事情为难您？\nhuman:然后但是局长口子已经开出来了，然后他就说我们系统不能直接那个。 播放这个录音怎么样？但其实是能播放的，只不过他们自己内部系统的一些问题我想解释，但是。 居然没有给我解释的机会，他不让那个他就在那一直守着，我真的是在全关机。 人的面前指着我的鼻子数落我，然后让我把我的领导叫来。 还说他一句话，还说他可以，如果我系统做得不好，可以不采用我们的系统。\nai:局长这么强势，您当时一定很无助。那您之后是怎么应对这种情况的呢？\nhuman:其实我当时内心是没有什么太多的感受的，因为我知道这不是我的问题，我只是替他们背了一个锅而已。 然后来他们的人就接手了，嗯，开始处理他们电脑的问题，嗯，后来他们的局长也发现。 是他们的人一直在处理问题，也意识到了是他们系统的问题，跟我们没关系。 然后但是还是把我的我们的领导叫来了，然后在这个过程中，他们公安局，我们客户嘛，我们对接的那个主任在孩子旁边小型的安慰我说那个没事没事，那个他发完会，领导发完火就过去了，就没事。 开始其实我内心是不在乎的\nai:您能这么快调整心态，真的很了不起。那后来您的领导来了之后，又发生了什么呢？\nhuman:领导也很萌，因为他们正在那边开新闻发布会，把我的领导，还有公安局和我对接的那些负责人都叫来了。 他们以为出了什么问题，大领导发大火。 他们比较猛，但是叫来之后，共局长也没有收这个身，受了一些无关紧要的东西，做了一些知识。 就这样就默默地背了一个锅。\nai:这事儿还真是挺曲折的。那经历了这件事，您对工作是不是有了新的认识呀？\nhuman:还好，也没有什么新的认识，我认为这都算是一种经历。 能被局长兼副市长指着给他妈，也是一种特殊的体验。\nai:您的心态真的很好呢。那在凉山州的这段经历，对您之后的工作选择有影响吗？\nhuman:没有什么影响，你以为我知道这不是我的问题，他领导们也知道我是背了一个锅，这个事就这么过去了。\nai:如此甚好。那我们再聊聊您的旅行吧，离开泸沽湖后，您又去了哪里呢？\nhuman:离开泸沽湖之后，我和我的那个义工搭子，我们一起去了虎跳峡徒步。\nai:徒步虎跳峡呀，这可真是一项很有挑战性又有趣的活动。你们在徒步过程中有遇到什么难忘的风景或事情吗？\nhuman:有，当时我们在还看到了日照金山。\nai:日照金山的美景可不常见，您真的很幸运呢。那一幕一定非常震撼吧，有没有拍照记录下来呀？\nhuman:有，当时我们很激动的拍了照，因为当时正好是徒步到那个半山腰的时候。嗯，看到的到了一个，临时的一个。 制扎点看到了，当时还挺兴奋的。\nai:能在徒步途中遇到这样的美景，确实会让人兴奋不已。那之后你们又去了哪里呢？\nhuman:我们在虎秀峡徒步之后， 在丽江休整了一天，然后就。 就去各自的下一个目的地了\n'
# 半自动编写/优化提示词





# 简要版说法

def test_0080():
    # 0080 prompt_get_infos
    demand = """
prompt_get_infos
"""
    @intellect(IntellectType.inference,"0080",demand = demand,table_name = "llm_prompt")
    def prompts(input_):
        # 可以直接输出, 也可以编写后处理的逻辑 extract_json 等
        return input_
    user_name = "柴东升"
    vitae = 12312
    result = prompts(f"{user_name},{vitae},{"\n".join(memory_cards)}")
    print(result,'result')

def test_0081():
    # 0081 prompt_base
    demand = """
prompt_base
"""
    @intellect(IntellectType.inference,"0081",demand = demand,table_name = "llm_prompt")
    def prompts(input_):
        # 可以直接输出, 也可以编写后处理的逻辑 extract_json 等
        return input_
    user_name = "柴东升"
    vitae = 12312
    result = prompts(f"{user_name},{vitae},{"\n".join(memory_cards)}")
    print(result,'result')


def test_0082():
    # 0082 素材增量生成
    demand = """
素材增量生成
"""
    @intellect(IntellectType.inference,"0082",demand = demand,table_name = "llm_prompt")
    def prompts(input_):
        # 可以直接输出, 也可以编写后处理的逻辑 extract_json 等
        return input_
    user_name = "柴东升"
    vitae = 12312
    result = prompts(f"{user_name},{vitae},{"\n".join(memory_cards)}")
    print(result,'result')

def test_0083():
    # 0083 传记简介
    demand = """
记忆卡片打分
"""
    @intellect(IntellectType.inference,"0083",demand = demand,table_name = "llm_prompt")
    def prompts(input_):
        # 可以直接输出, 也可以编写后处理的逻辑 extract_json 等
        return input_
    user_name = "柴东升"
    vitae = 12312
    result = prompts(f"{user_name},{vitae},{"\n".join(memory_cards)}")
    print(result,'result')

def test_0084():
    # 0084 大纲生成
    demand = """
记忆卡片打分
"""
    @intellect(IntellectType.inference,"0084",demand = demand,table_name = "llm_prompt")
    def prompts(input_):
        # 可以直接输出, 也可以编写后处理的逻辑 extract_json 等
        return input_
    user_name = "柴东升"
    vitae = 12312
    result = prompts(f"{user_name},{vitae},{"\n".join(memory_cards)}")
    print(result,'result')

def test_0085():
    # 0085 素材整理
    demand = """
记忆卡片打分
"""
    @intellect(IntellectType.inference,"0085",demand = demand,table_name = "llm_prompt")
    def prompts(input_):
        # 可以直接输出, 也可以编写后处理的逻辑 extract_json 等
        return input_
    user_name = "柴东升"
    vitae = 12312
    result = prompts(f"{user_name},{vitae},{"\n".join(memory_cards)}")
    print(result,'result')

def test_0086():
    # 0086 提取地名
    demand = """
提取地名
"""
    @intellect(IntellectType.inference,"0086",demand = demand,table_name = "llm_prompt")
    def prompts(input_):
        # 可以直接输出, 也可以编写后处理的逻辑 extract_json 等
        return input_
    user_name = "柴东升"
    vitae = 12312
    result = prompts(f"{user_name},{vitae},{"\n".join(memory_cards)}")
    print(result,'result')

def test_0087():
    # 0087 提取人名
    demand = """
提取人名
"""
    @intellect(IntellectType.inference,"0087",demand = demand,table_name = "llm_prompt")
    def prompts(input_):
        # 可以直接输出, 也可以编写后处理的逻辑 extract_json 等
        return input_
    user_name = "柴东升"
    vitae = 12312
    result = prompts(f"{user_name},{vitae},{"\n".join(memory_cards)}")
    print(result,'result')



def test_0088():
    # 0088 记忆卡片打分
    demand = """
记忆卡片打分
"""
    @intellect(IntellectType.inference,"0088",demand = demand,table_name = "llm_prompt")
    def prompts(input_):
        # 可以直接输出, 也可以编写后处理的逻辑 extract_json 等
        return input_
    user_name = "柴东升"
    vitae = 12312
    result = prompts(f"{user_name},{vitae},{"\n".join(memory_cards)}")
    print(result,'result')


def test_0089():
    # 0089 记忆卡片合并
    demand = """
记忆卡片合并
"""
    @intellect(IntellectType.inference,"0089",demand = demand,table_name = "llm_prompt")
    def prompts(input_):
        # 可以直接输出, 也可以编写后处理的逻辑 extract_json 等
        return input_
    user_name = "柴东升"
    vitae = 12312
    result = prompts(f"{user_name},{vitae},{"\n".join(memory_cards)}")
    print(result,'result')


def test_0090():
    # 0090 记忆卡片润色
    demand = """
记忆卡片润色
"""
    @intellect(IntellectType.inference,"0090",demand = demand,table_name = "llm_prompt")
    def prompts(input_):
        # 可以直接输出, 也可以编写后处理的逻辑 extract_json 等
        return input_
    user_name = "柴东升"
    vitae = 12312
    result = prompts(f"{user_name},{vitae},{"\n".join(memory_cards)}")
    print(result,'result')


def test_0091():
    # 0091 通过文本生成记忆卡片-memory_card_system_prompt
    demand = """
通过文本生成记忆卡片
"""
    @intellect(IntellectType.inference,"0091",demand = demand,table_name = "llm_prompt")
    def prompts(input_):
        # 可以直接输出, 也可以编写后处理的逻辑 extract_json 等
        return input_
    user_name = "柴东升"
    vitae = 12312
    result = prompts(f"{user_name},{vitae},{"\n".join(memory_cards)}")
    print(result,'result')

def test_0092():
    # 0092 通过文本生成记忆卡片-time_prompt
    demand = """
通过文本生成记忆卡片
"""
    @intellect(IntellectType.inference,"0092",demand = demand,table_name = "llm_prompt")
    def prompts(input_):
        # 可以直接输出, 也可以编写后处理的逻辑 extract_json 等
        input_ = extract_json(input_)
        return input_
    user_name = "柴东升"
    vitae = 12312
    result = prompts(f"{user_name},{vitae},{"\n".join(memory_cards)}")
    print(result,'result')


def test_0093():
    # 0093 上传文件生成记忆卡片-memory_card_system_prompt
    demand = """
聊天历史生成记忆卡片
"""
    @intellect(IntellectType.inference,"0093",demand = demand,table_name = "llm_prompt")
    def prompts(input_):
        # 可以直接输出, 也可以编写后处理的逻辑 extract_json 等
        return input_
    user_name = "柴东升"
    vitae = 12312
    result = prompts(f"{user_name},{vitae},{"\n".join(memory_cards)}")
    print(result,'result')

def test_0094():
    # 0094 聊天历史生成记忆卡片-time_prompt
    demand = """
聊天历史生成记忆卡片
"""
    @intellect(IntellectType.inference,"0094",demand = demand,table_name = "llm_prompt")
    def prompts(input_):
        # 可以直接输出, 也可以编写后处理的逻辑 extract_json 等
        input_ = extract_json(input_)
        return input_
    user_name = "柴东升"
    vitae = 12312
    result = prompts(f"{user_name},{vitae},{"\n".join(memory_cards)}")
    print(result,'result')




def test_0095():
    # 0095 简要版说法
    demand = """
帮我利用下面这些信息,生成一篇个人传记:


输出格式:  
```article
<content>
``` 
"""
    @intellect(IntellectType.inference,"0095",demand = demand,table_name = "llm_prompt")
    def prompts(input_):
        # 可以直接输出, 也可以编写后处理的逻辑 extract_json 等
        input_ = extract_article(input_)
        return input_
    user_name = "柴东升"
    vitae = 12312
    result = prompts(f"{user_name},{vitae},{"\n".join(memory_cards)}")
    print(result,'result')

def test_0096():
    # 0096 生成用户概述
    demand = """
帮我提取这个人的用户概述, 大概100字
"""
    @intellect(IntellectType.inference,"0096",demand = demand,table_name = "llm_prompt")
    def prompts(input_):
        # 可以直接输出, 也可以编写后处理的逻辑 extract_json 等
        return input_
    old_dverview = "用户概述"
    result = prompts(old_dverview + "\n".join(memory_cards))
    print(result,'result')


def test_0097():
    # 0097 用户关系提取
    demand = """
我希望你可以从一段对话历史中, 提取如下的用户关系表, 如下

 {
            “小王”:  # 人物姓名
	    {
                “relationship”: "关系",
                "profession": "职业",
                "birthday”: "出生日期",
            },
            “大王”: {
                "relationship": "关系",
                "profession": "职业",
                "birthday": "出生日期",
            },
        }

我会提供给你新的聊天历史从新的聊天历史中提取信息

以
```json
内容

```
的格式输出
"""
    @intellect(IntellectType.inference,"0097",demand = demand,table_name = "llm_prompt")
    def prompts(input_):
        # 可以直接输出, 也可以编写后处理的逻辑 extract_json 等
        return input_

    result = prompts(chat_history)
    print(result,'result')

def test_0098():
    # 0098 数字分身简介
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
    # 0099 数字分身性格提取
    demand = """
你可以将处理后的文字使用```article 内容 ```的方式返回给我
"""
    @intellect(IntellectType.inference,"0099",demand = demand,table_name = "llm_prompt")
    def prompts(input_):
        # 可以直接输出, 也可以编写后处理的逻辑 extract_json 等
        return input_

    result = prompts("\n".join(memory_cards))
    print(result,'result')


def test_0100():
    # 0100 数字分身信息脱敏
    demand = "你可以将处理后的文字使用```article 内容 ```的方式返回给我"
    @intellect(IntellectType.inference,"0100",demand = demand,table_name = "llm_prompt")
    def prompts(input_):
        # 可以直接输出, 也可以编写后处理的逻辑 extract_json 等
        return extract_article(input_)

    result = prompts("我出生在东北辽宁葫芦岛下面的一个小村庄。小时候，那里的生活比较简单，人们日出而作，日落而息，生活节奏非常有规律，也非常美好。当时我们都是山里的野孩子，没有什么特别的兴趣爱好，就在山里各种疯跑。我小时候特别喜欢晚上看星星，那时的夜晚星星非常多，真的是那种突然就能看到漫天繁星的感觉。")
    
    super_print(result,'result')
