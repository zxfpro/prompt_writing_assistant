'''
Author: 823042332@qq.com 823042332@qq.com
Date: 2025-08-28 09:07:54
LastEditors: 823042332@qq.com 823042332@qq.com
LastEditTime: 2025-08-28 09:30:32
FilePath: /prompt_writing_assistant/src/prompt_writing_assistant/unit.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import re
import inspect
import importlib
import yaml

def extract_json(text: str) -> str:
    """从文本中提取python代码
    Args:
        text (str): 输入的文本。
    Returns:
        str: 提取出的python文本
    """
    pattern = r"```json([\s\S]*?)```"
    matches = re.findall(pattern, text)
    if matches:
        return matches[0].strip()  # 添加strip()去除首尾空白符
    else:
        return ""  # 返回空字符串或抛出异常，此处返回空字符串

def extract_json_multi(text: str) -> list[str]:
    """从文本中提取python代码
    Args:
        text (str): 输入的文本。
    Returns:
        str: 提取出的python文本
    """
    pattern = r"```json([\s\S]*?)```"
    matches = re.findall(pattern, text)
    if matches:
        return [match.strip() for match in matches]        
    else:
        return []  # 返回空字符串或抛出异常，此处返回空字符串


def extract_python(text: str) -> str:
    """从文本中提取python代码
    Args:
        text (str): 输入的文本。
    Returns:
        str: 提取出的python文本
    """
    pattern = r"```python([\s\S]*?)```"
    matches = re.findall(pattern, text)
    if matches:
        return matches[0].strip()  # 添加strip()去除首尾空白符
    else:
        return ""  # 返回空字符串或抛出异常，此处返回空字符串


def extract_curl(text: str) -> str:
    """从文本中提取python代码
    Args:
        text (str): 输入的文本。
    Returns:
        str: 提取出的python文本
    """
    pattern = r"```curl([\s\S]*?)```"
    matches = re.findall(pattern, text)
    if matches:
        return matches[0].strip()  # 添加strip()去除首尾空白符
    else:
        return ""  # 返回空字符串或抛出异常，此处返回空字符串

def extract_from_loaded_objects(obj_list):
    results = []
    for obj in obj_list:
        if inspect.isclass(obj):
            class_info = {
                "type": "class",
                "name": obj.__name__,
                "docstring": inspect.getdoc(obj),
                "signature": f"class {obj.__name__}{inspect.getclasstree([obj], unique=True)[0][0].__bases__}:" if inspect.getclasstree([obj], unique=True)[0][0].__bases__ != (object,) else f"class {obj.__name__}:", # 尝试获取基类
                "methods": []
            }
            # 遍历类的方法
            for name, member in inspect.getmembers(obj, predicate=inspect.isfunction):
                if name.startswith('__') and name != '__init__': # 过滤掉大多数魔术方法，但保留 __init__
                    continue
                
                # inspect.signature 可以获取更精确的签名
                sig = inspect.signature(member)
                is_async = inspect.iscoroutinefunction(member)

                method_info = {
                    "type": "method",
                    "name": name,
                    "docstring": inspect.getdoc(member),
                    "signature": f"{'async ' if is_async else ''}def {name}{sig}:",
                    "is_async": is_async
                }
                class_info["methods"].append(method_info)
            results.append(class_info)
        elif inspect.isfunction(obj) or inspect.iscoroutinefunction(obj):
            is_async = inspect.iscoroutinefunction(obj)
            sig = inspect.signature(obj)
            results.append({
                "type": "function",
                "name": obj.__name__,
                "docstring": inspect.getdoc(obj),
                "signature": f"{'async ' if is_async else ''}def {obj.__name__}{sig}:",
                "is_async": is_async
            })
    return results



def load_inpackage_file(package_name:str, file_name:str,file_type = 'yaml'):
    """ load config """
    with importlib.resources.open_text(package_name, file_name) as f:
        if file_type == 'yaml':
            return yaml.safe_load(f)
        else:
            return f.read()


def super_print(s,target:str):
    print()
    print()
    print("=="*21 + target + "=="*21)
    print()
    print("=="*50)
    print(type(s))
    print("=="*50)
    print(s)
    print("=="*50)
    print()

