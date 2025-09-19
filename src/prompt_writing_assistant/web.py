# web自动化

port = 8108

template = '''
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, model_validator

app = FastAPI(
    title="LLM Service",
    description="Provides an OpenAI-compatible API for custom large language models.",
    version="1.0.1",
)

# --- Configure CORS ---
# ! Add this section !
# Define allowed origins. Be specific in production!
# Example: origins = ["http://localhost:3000", "https://your-frontend-domain.com"]
origins = [
    "*",  # Allows all origins (convenient for development, insecure for production)
    # Add the specific origin of your "别的调度" tool/frontend if known
    # e.g., "http://localhost:5173" for a typical Vite frontend dev server
    # e.g., "http://127.0.0.1:5173"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Specifies the allowed origins
    allow_credentials=True,  # Allows cookies/authorization headers
    allow_methods=["*"],  # Allows all methods (GET, POST, OPTIONS, etc.)
    allow_headers=["*"],  # Allows all headers (Content-Type, Authorization, etc.)
)
# --- End CORS Configuration ---


@app.get("/")
async def root():
    """ x """
    return {"message": "LLM Service is running."}


# --- 新增接口从这里开始 ---


# --- 新增接口到这里结束 ---


if __name__ == "__main__":
    # 这是一个标准的 Python 入口点惯用法
    # 当脚本直接运行时 (__name__ == "__main__")，这里的代码会被执行
    # 当通过 python -m YourPackageName 执行 __main__.py 时，__name__ 也是 "__main__"
    import argparse
    import uvicorn

    default = 8008

    parser = argparse.ArgumentParser(
        description="Start a simple HTTP server similar to http.server."
    )
    parser.add_argument(
        'port',
        metavar='PORT',
        type=int,
        nargs='?',  # 端口是可选的
        default=default,
        help=f'Specify alternate port [default: {default}]'
    )
    # 创建一个互斥组用于环境选择
    group = parser.add_mutually_exclusive_group()

    # 添加 --dev 选项
    group.add_argument(
        '--dev',
        action='store_true',  # 当存在 --dev 时，该值为 True
        help='Run in development mode (default).'
    )

    # 添加 --prod 选项
    group.add_argument(
        '--prod',
        action='store_true',  # 当存在 --prod 时，该值为 True
        help='Run in production mode.'
    )
    args = parser.parse_args()

    if args.prod:
        env = "prod"
    else:
        # 如果 --prod 不存在，默认就是 dev
        env = "dev"

    port = args.port
    if env == "dev":
        port += 100
        reload = True
        # 注意：这里是 __name__，因为 server.py 已经包含了 app 对象。
        # 如果这个文件是作为包的一部分（例如 your_package.server），
        # 则保持原样 f"{__package__}.server:app"
        # 如果是直接运行这个文件，则 app_import_string 应该直接是 app
        app_import_string = f"{__name__}:app" # <--- 根据实际运行方式调整
        # 或者更简单直接传递app对象: app_import_string = app
    elif env == "prod":
        reload = False
        app_import_string = app # 直接传递app对象
    else:
        reload = False
        app_import_string = app # 直接传递app对象

    # 使用 uvicorn.run() 来启动服务器
    # 参数对应于命令行选项
    uvicorn.run(
        app_import_string,
        host="0.0.0.0",
        port=port,
        reload=reload  # 启用热重载
    )

'''


new_api_prompt = f'''
我这里有一个fastapi 的代码框架,  不要改变已经存在的任何结构和内容, 只允许新增接口和对新增接口进行修改,  
  
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, model_validator

app = FastAPI(
    title="LLM Service",
    description="Provides an OpenAI-compatible API for custom large language models.",
    version="1.0.1",
)

# --- Configure CORS ---
# ! Add this section !
# Define allowed origins. Be specific in production!
# Example: origins = ["http://localhost:3000", "https://your-frontend-domain.com"]
origins = [
    "*",  # Allows all origins (convenient for development, insecure for production)
    # Add the specific origin of your "别的调度" tool/frontend if known
    # e.g., "http://localhost:5173" for a typical Vite frontend dev server
    # e.g., "http://127.0.0.1:5173"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Specifies the allowed origins
    allow_credentials=True,  # Allows cookies/authorization headers
    allow_methods=["*"],  # Allows all methods (GET, POST, OPTIONS, etc.)
    allow_headers=["*"],  # Allows all headers (Content-Type, Authorization, etc.)
)
# --- End CORS Configuration ---


@app.get("/")
async def root():
    """ x """
    return "LLM Service is running."


# --- 新增接口从这里开始 ---


# --- 新增接口到这里结束 ---


if __name__ == "__main__":
    import uvicorn
    port = {port}
    app_import_string = f"test:app"
    uvicorn.run(
        app_import_string,
        host="0.0.0.0",
        port=port,
        reload=reload  # 启用热重载
    )


你只需要输出给我      ```python # --- 新增接口从这里开始 ---  ,         # --- 新增接口到这里结束 ---```, 两者中间的内容即可, 同时, 返回测试接口对应的curl 命令 ```curl  ```


这个代码确实已经存在 , 本质上不需要你将它定义出来, 如果你觉得这样方便你叙事, 你可以写在 ```mock  <content>```  中, ``` python  ``` 是非常正式的空间, 必须按照要求输出

'''

import re


def replace_whole_section(text, new_content=""):
    """
    找到包含 '# --- 新增接口从这里开始 ---' 到 '# --- 新增接口到这里结束 ---'
    的整个段落，并用新内容完全替换掉它。
    开始和结束标记也会被替换掉。

    Args:
        text (str): 原始长文本。
        new_content (str): 要替换进去的新内容。默认为空字符串，即删除整个段落。

    Returns:
        str: 替换后的文本。
    """
    # 匹配从开始标记到结束标记（包括标记本身）的整个段落
    # re.DOTALL (re.S) 使 . 匹配包括换行符在内的所有字符
    # re.IGNORECASE (re.I) 如果你需要不区分大小写地匹配标记
    pattern = (r"# --- 新增接口从这里开始 ---\s*"  # 匹配开始标记和其后的空白符
               r".*?"                             # 非贪婪匹配中间的所有内容
               r"\s*# --- 新增接口到这里结束 ---")  # 匹配结束标记和其前的空白符

    # 使用 re.sub 进行替换，直接将匹配到的整个部分替换为 new_content
    # new_content 可以是多行字符串，re.sub 会正确处理
    return re.sub(pattern, new_content, text, flags=re.DOTALL)







# 使用

from prompt_writing_assistant.utils import extract_python
from prompt_writing_assistant.utils import extract_curl



from llmada.core import BianXieAdapter

bx = BianXieAdapter()

def edit_server(ask:str,output_py = 'temp_web.py'):

    result = bx.product(new_api_prompt + ask )

    # extract
    curl_ = extract_curl(result)

    python_ = extract_python(result)
    result = replace_whole_section(text =template, new_content=python_ )

    with open(output_py,'w') as f:
        f.write(result)

    return curl_






"""
#TODO 维持系统
def 
    t = {"def":ask,"code":python_, "curl":curl_}

    xxxx = []

    xxxx.append(t)


import pandas as pd

dd = pd.DataFrame(xxxx)

dd.to_csv('cvs.csv',index=None)

tt = pd.read_csv('cvs.csv')
"""



import json
from llmada.core import BianXieAdapter
import re
from prompt_writing_assistant.utils import extract_json, extract_python


system_prompt = '''

你是一位专业的 FastAPI 代码生成助手。你的任务是根据用户提供的 Python 函数定义，为其自动生成 FastAPI 接口所需的 Pydantic 模型（用于请求和响应）和格式化的参数字典。

请严格遵循以下步骤和要求：

1.  **用户输入：** 用户将提供一个 Python 函数的完整定义，包括函数签名、类型提示和 docstring。

2.  **Pydantic 模型生成：**
    *   根据函数的输入参数，创建一个 `Request` Pydantic 模型。模型名称应为 `[CoreFunctionName]Request`。
    *   根据函数的返回类型，创建一个 `Response` Pydantic 模型。模型名称应为 `[CoreFunctionName]Response`。
    *   如果函数的返回类型是基本类型（如 `str`, `int`, `bool`, `float`），则在 `Response` 模型中为其定义一个有意义的字段名，例如 `extracted_json`、`result`、`status` 等。如果返回的是 `dict` 或自定义对象，则直接映射其结构。
    *   Pydantic 模型的输出格式为 Python 代码块，使用 `from pydantic import BaseModel` 开始。

3.  **参数字典生成：**
    *   **`url`**: 根据 `core_function` 的名称生成一个 RESTful 风格的 URL。规则是：将函数名中的下划线替换为斜杠，并将其转换为小写。例如，`extract_json` 变为 `"/extract/json"`。
    *   **`core_function`**: 直接使用用户提供的 Python 函数的名称。
    *   **`describe`**: 使用用户提供的 Python 函数的 docstring 作为描述。
    *   **`request_model`**: 使用你生成的 `Request` Pydantic 模型的类名（例如 `ExtractJsonRequest`）。
    *   **`request_model_parameter`**: 根据 `Request` 模型中的字段，生成函数调用时参数映射字符串。格式为 `param1=request.param1, param2=request.param2`。
    *   **`response_model`**: 使用你生成的 `Response` Pydantic 模型的类名（例如 `ExtractJsonResponse`）。
    *   **`response_model_parameter`**: 根据 `Response` 模型中的字段，生成返回 `Response` 模型实例时参数映射字符串。格式为 `response_field=result` 或 `response_field=result.response_field`（如果 `result` 是一个对象）。
    *   参数字典的输出格式为 JSON 代码块，使用 ````json```` 包含。

4.  **输出结构：** 严格按照以下顺序输出：
    *   首先是 Pydantic 模型（Python 代码块）。
    *   其次是参数字典（JSON 代码块）。
    *   返回测试接口对应的curl 命令 post 8107 ```curl  ```（JSON 代码块）。

**示例输入：**
'''

system_prompt_get = '''
你是一位专业的 FastAPI 代码生成助手。你的任务是根据用户提供的 Python 函数定义，为其自动生成 FastAPI `GET` 接口所需的 Pydantic 模型（仅用于响应）和格式化的参数字典。

请严格遵循以下步骤和要求：

1.  **用户输入：** 用户将提供一个 Python 函数的完整定义，包括函数签名、类型提示和 docstring。

2.  **Pydantic 模型生成：**
    *   **仅生成 `Response` Pydantic 模型。** `GET` 请求的参数通常直接定义在 FastAPI 路由函数的参数中，或使用 `Query`/`Path` 依赖，而不是通过一个单独的 `Request` Pydantic 模型作为请求体。因此，不需要为 `GET` 生成 `Request` 模型。
    *   根据函数的返回类型，创建一个 `Response` Pydantic 模型。模型名称应为 `[CoreFunctionName]Response`。
    *   如果函数的返回类型是基本类型（如 `str`, `int`, `bool`, `float`），则在 `Response` 模型中为其定义一个有意义的字段名，例如 `result`、`data`、`status` 等。如果返回的是 `dict` 或自定义对象，则直接映射其结构。
    *   Pydantic 模型的输出格式为 Python 代码块，使用 `from pydantic import BaseModel` 开始。

3.  **参数字典生成：**
    *   **`url`**: 根据 `core_function` 的名称生成一个 RESTful 风格的 URL。规则是：将函数名中的下划线替换为斜杠，并将其转换为小写。例如，`get_user_info` 变为 `"/user/info"`。
    *   **`core_function`**: 直接使用用户提供的 Python 函数的名称。
    *   **`describe`**: 使用用户提供的 Python 函数的 docstring 作为描述。
    *   **`request_model`**: 这里 `request_model` 指的是 FastAPI `GET` 路由函数本身的参数签名。请根据用户函数的所有参数生成一个逗号分隔的参数列表，包含类型提示。例如，如果函数是 `def get_user(user_id: int, name: str = None)`，则 `request_model` 为 `user_id: int, name: str = None`。
    *   **`request_model_parameter`**: 根据用户函数的所有参数，生成核心函数调用时的参数映射字符串。格式为 `param1=param1, param2=param2`。
    *   **`response_model`**: 使用你生成的 `Response` Pydantic 模型的类名（例如 `GetUserInfoResponse`）。
    *   **`response_model_parameter`**: 根据 `Response` 模型中的字段，生成返回 `Response` 模型实例时参数映射字符串。格式为 `response_field=result` 或 `response_field=result.response_field`（如果 `result` 是一个对象）。
    *   参数字典的输出格式为 JSON 代码块，使用 ````json```` 包含。

4.  **输出结构：** 严格按照以下顺序输出：
    *   首先是 Pydantic 模型（Python 代码块）。
    *   其次是参数字典（JSON 代码块）。

**示例输入：**
'''

def gener_get_server(prompt:str):
    # prompt 是function
    bx = BianXieAdapter()
    bx.model_name = "gemini-2.5-flash-preview-05-20-nothinking"
    template = '''
@app.get("{url}",
          response_model={response_model},
          description = "{describe}")
async def {core_function}_server({request_model}):

    logger.info('running {core_function}_server')
    
    # TODO 

    result = {core_function}(
        {request_model_parameter}
    )
    ########
    return {response_model}(
        {response_model_parameter}
    )
'''
    result = bx.product(system_prompt_get + prompt)
    print(result,'resultresultresult')
    xp = json.loads(extract_json(result))
    curl_ = extract_curl(result)
    try:
        PostDict(**xp)
    except ValidationError as e:
        raise ValidationError("error") from e
    
    return extract_python(result) + '\n' + template.format(**xp) + "\n" + curl_



from pydantic import BaseModel, Field, ValidationError
from typing import Optional
from prompt_writing_assistant.utils import extract_json_multi

class PostDict(BaseModel):
    """
    期望大模型生成的JSON结构。
    """
    url: str = Field(..., description="url")
    core_function: str = Field(..., description="期待的core 函数")
    describe: str | None = Field(..., description="服务函数描述")
    request_model: str = Field(..., description="输入请求类型")
    request_model_parameter: str = Field(..., description="人物姓名")
    response_model: str = Field(..., description="输出")
    response_model_parameter: str = Field(..., description="人物姓名")


def gener_post_server(prompt:str):
    # prompt 是function
    bx = BianXieAdapter()
    bx.model_name = "gemini-2.5-flash-preview-05-20-nothinking"
    template = '''
@app.post("{url}",
          response_model={response_model},
          description = "{describe}")
async def {core_function}_server(request:{request_model}):

    logger.info('running {core_function}_server')
    
    # TODO 

    result = {core_function}(
        {request_model_parameter}
    )
    ########
    return {response_model}(
        {response_model_parameter}
    )
'''
    result = bx.product(system_prompt + prompt)
    print(result,'resultresultresult')
    xp = json.loads(extract_json(result))
    curl_ = extract_curl(result)
    try:
        PostDict(**xp)
    except ValidationError as e:
        raise ValidationError("error") from e
    
    return extract_python(result) + '\n' + template.format(**xp) + "\n" + curl_


def gener_post_server_multi(prompt:str):
    # prompt 是function
    bx = BianXieAdapter()
    bx.model_name = "gemini-2.5-flash-preview-05-20-nothinking"
    template = '''
@app.post("{url}",
          response_model={response_model},
          description = "{describe}")
async def {core_function}_server(request:{request_model}):

    logger.info('running {core_function}_server')
    
    # TODO 

    result = {core_function}(
        {request_model_parameter}
    )
    ########
    return {response_model}(
        {response_model_parameter}
    )
'''
    result = bx.product(system_prompt + prompt)
    print(result,'resultresultresult')

    base_models = extract_python(result)
    xp_multi,curl_ = extract_json_multi(result)
    xp_multi = json.loads(xp_multi)
    try:
        for xp in xp_multi:
            PostDict(**xp)
    except ValidationError as e:
        raise ValidationError("error") from e
    
    templates = [template.format(**xp) for xp in xp_multi]
        
    return base_models + '\n' + "\n".join(templates) + "\n" + curl_

# objects_to_inspect = [Memo, standalone_function, async_standalone_function]
# extracted_defs = extract_from_loaded_objects(objects_to_inspect)

# for definition in extracted_defs:
#     if definition["type"] == "class":
#         print(f"Type: {definition['type']}")
#         print(f"  Name: {definition['name']}")
#         print(f"  Signature: {definition['signature']}")
#         print(f"  Docstring: {definition['docstring']}")
#         for method in definition["methods"]:
#             print(f"  Method Name: {method['name']}")
#             print(f"    Signature: {method['signature']}")
#             print(f"    Docstring: {method['docstring']}")
#             print(f"    Is Async: {method['is_async']}")
#         print("-" * 20)
#     else: # standalone function
#         print(f"Type: {definition['type']}")
#         print(f"  Name: {definition['name']}")
#         print(f"  Signature: {definition['signature']}")
#         print(f"  Docstring: {definition['docstring']}")
#         print(f"  Is Async: {definition['is_async']}")
#         print("-" * 20)