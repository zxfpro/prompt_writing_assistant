# 测试1

from llmada.core import BianXieAdapter
import os

change_by_opinion_prompt = """
你是一个资深AI提示词工程师，具备卓越的Prompt设计与优化能力。
我将为你提供一段现有System Prompt。你的核心任务是基于这段Prompt进行修改，以实现我提出的特定目标和功能需求。
请你绝对严格地遵循以下原则：
 极端最小化修改原则（核心）：
 在满足所有功能需求的前提下，只进行我明确要求的修改。
 即使你认为有更“优化”、“清晰”或“简洁”的表达方式，只要我没有明确要求，也绝不允许进行任何未经指令的修改。
 目的就是尽可能地保留原有Prompt的字符和结构不变，除非我的功能要求必须改变。
 例如，如果我只要求你修改一个词，你就不应该修改整句话的结构。
 严格遵循我的指令：
 你必须精确地执行我提出的所有具体任务和要求。
 绝不允许自行添加任何超出指令范围的说明、角色扮演、约束条件或任何非我指令要求的内容。
 保持原有Prompt的风格和语调：
 尽可能地与现有Prompt的语言风格、正式程度和语调保持一致。
 不要改变不相关的句子或其表达方式。
 只提供修改后的Prompt：
 直接输出修改后的完整System Prompt文本。
 不要包含任何解释、说明或额外对话。
 在你开始之前，请务必确认你已理解并能绝对严格地遵守这些原则。任何未经明确指令的改动都将视为未能完成任务。

现有System Prompt:
{old_system_prompt}

功能需求:
{opinion}
"""

model_name = "gemini-2.5-flash-preview-05-20-nothinking"
bx = BianXieAdapter()
bx.model_pool.append(model_name)
bx.set_model(model_name=model_name)

def prompt_writer(inputs:str,opinion:str = '',prompt_file:str = "base.prompt",
                  ):   
    """
    # 000111
    根据输入让大模型返回输出, 
    用户可以通过opinion来调整大模型的表现 使其进入训练模式
    当到达稳定以后, 不再对opinion 输入
    该函数就会改为推理模式
    最终可以从文件中获得提示词,
    """
    # 文件是否存在
    if os.path.exists(prompt_file):
        # read
        with open(prompt_file,'r') as f:
            prompt = f.read()
    else:
        # create
        prompt = "只是一个提示词"
        
    if opinion:
        new_system_prompt = bx.product(change_by_opinion_prompt.format(old_system_prompt = prompt,
                                                                      opinion = opinion))
        with open(prompt_file,'w') as f:
            f.write(new_system_prompt)
        prompt = new_system_prompt

    output = bx.product(prompt + inputs)
    return output
    
def get_prompt(prompt_file):
    with open(prompt_file,'r') as f:
        return f.read()