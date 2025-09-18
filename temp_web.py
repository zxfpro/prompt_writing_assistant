
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

# 记忆合并
class Memory_card_list_Request(BaseModel):
    memory_cards: list[str] = Field(..., description="要合并的记忆卡片内容列表")

# 假设 memory_card_merge 函数已经定义在某个地方
# 例如:
def memory_card_merge(memory_cards: list[str]) -> str:
    """
    模拟记忆卡片合并逻辑。
    实际应用中，这里会调用你的LLM或其他合并逻辑。
    """
    if not memory_cards:
        return ""
    # 简单地将所有卡片内容用换行符连接起来
    return "".join(memory_cards)

@app.post("/memory_card/merge")
async def memory_card_merge_server(request: Memory_card_list_Request):
    """
    合并记忆卡片。
    接收一个记忆卡片内容字符串列表，并返回合并后的内容。
    """
    if not request.memory_cards:
        raise HTTPException(status_code=400, detail="memory_cards cannot be empty")

    result = memory_card_merge(memory_cards=request.memory_cards)
    return {
        "message": "memory card merge successfully",
        "merged_content": result
    }

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

