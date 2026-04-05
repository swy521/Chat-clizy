import sys
import io
# 问db强制UTF-8输出，解决中文乱码
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import os
import json
import re
from pathlib import Path
from openai import OpenAI
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
import tiktoken

# 模型配置 dk+智谱
# API Key
MODELS = {
    "deepseek": {
        "api_key": "sk-5b31e90f996242d09fe85ee335a1755f",
        "base_url": "https://api.deepseek.com/v1",
        "model": "deepseek-chat",
        "max_tokens": 4096,
        "context_window": 128000
    },
    "zhipu": {
        "api_key": "ecdd9d264373429bb51bb11f4dc2a3d0.gMYBI9vyu5SzDvEh",  # 智谱Key
        "base_url": "https://open.bigmodel.cn/api/paas/v4/",
        "model": "glm-4-flash",  # 智谱免费模型
        "max_tokens": 4096,
        "context_window": 128000
    }
}
# 默认dk
CURRENT_MODEL = "deepseek"
# ======================================================================

# 配置目录
CONFIG_DIR = Path.home() / ".chat_cli"
CONFIG_PATH = CONFIG_DIR / "config.json"
HISTORY_PATH = CONFIG_DIR / "history.json"
INPUT_HISTORY_PATH = CONFIG_DIR / "input_history.txt"

def count_tokens(messages, model_name):
    """计算token量"""
    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except:
        encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = 0
    for msg in messages:
        num_tokens += 4
        for k, v in msg.items():
            num_tokens += len(encoding.encode(str(v)))
            if k == "name":
                num_tokens -= 1
    num_tokens += 2
    return num_tokens

def truncate_messages(messages, context_window, model_name):
    """截断历史，保留system消息"""
    system = [m for m in messages if m["role"] == "system"]
    user_assist = [m for m in messages if m["role"] != "system"]
    while True:
        total = count_tokens(system + user_assist, model_name)
        if total <= context_window or not user_assist:
            break
        user_assist.pop(0)
    return system + user_assist

def process_file_references(text):
    """识别部分文件名"""
    pattern = r'@"([^"]+)"|@(\S+)'
    matches = re.findall(pattern, text)
    new_text = text
    for match in matches:
        fp = match[0] if match[0] else match[1]
        try:
            fp = Path(fp).expanduser()
            if fp.suffix not in [".py", ".txt", ".md", ".json"]:
                print(f"仅支持 .py/.txt/.md/.json")
                continue
            with open(fp, 'r', encoding='utf-8') as f:
                content = f.read()
            old = f'@"{fp}"' if match[0] else f'@{fp}'
            new = f'\n--- 文件: {fp.name} ---\n{content}\n--- 文件结束 ---\n'
            new_text = new_text.replace(old, new)
        except Exception as e:
            print(f"读取文件失败 {fp}: {e}")
    return new_text

def handle_command(cmd, history):
    """处理命令：/model /exit /clear /reset /help"""
    global CURRENT_MODEL
    parts = cmd.strip().lower().split()
    if not parts:
        return True
    if parts[0] == "/model":
        if len(parts) < 2:
            print(f"用法：/model list | /model <模型名>")
            print(f"可用模型：{', '.join(MODELS.keys())}")
            return True
        sub = parts[1]
        if sub == "list":
            print("可用模型：")
            for name in MODELS:
                mark = "当前" if name == CURRENT_MODEL else ""
                print(f"  {name} - {MODELS[name]['model']} {mark}")
            return True
        if sub in MODELS:
            CURRENT_MODEL = sub
            print(f"已切换到模型：{CURRENT_MODEL} ({MODELS[CURRENT_MODEL]['model']})")
            return True
        else:
            print(f"未知模型：{sub}，可用：{', '.join(MODELS.keys())}")
            return True
    elif parts[0] == "/exit":
        print("再见！")
        exit(0)
    elif parts[0] == "/help":
        print("="*40)
        print("          命令帮助")
        print("="*40)
        print("  /model list    - 列出模型")
        print("  /model <name>  - 切换模型（如 /model zhipu）")
        print("  /clear         - 清空当前对话")
        print("  /reset         - 重置所有历史")
        print("  /help          - 帮助")
        print("  /exit          - 退出")
        print("  @filename      - 读取本地文件")
        print("="*40)
        return True
    elif parts[0] == "/clear":
        non_system = [m for m in history if m["role"] != "system"]
        if non_system:
            history[:] = [m for m in history if m["role"] == "system"]
            print("已清空当前对话")
            with open(HISTORY_PATH, 'w', encoding='utf-8') as f:
                json.dump(history, f, indent=2)
        else:
            print("对话已为空")
        return True
    elif parts[0] == "/reset":
        history[:] = [{"role": "system", "content": "You are a helpful assistant."}]
        if HISTORY_PATH.exists():
            HISTORY_PATH.unlink()
        print("已重置所有历史")
        return True
    else:
        print(f"未知命令：{parts[0]}，输入 /help 查看")
        return True

def main():
    CONFIG_DIR.mkdir(exist_ok=True, parents=True)
    # 加载历史
    if HISTORY_PATH.exists():
        with open(HISTORY_PATH, 'r', encoding='utf-8') as f:
            history = json.load(f)
    else:
        history = [{"role": "system", "content": "You are a helpful assistant."}]
    # 输入历史
    session = PromptSession(history=FileHistory(str(INPUT_HISTORY_PATH)))
    # 欢迎
    print("="*50)
    print(f"      Chat CLI - 当前模型：{CURRENT_MODEL}")
    print("="*50)
    print("输入 /help 查看命令 | /exit 退出")
    print("-"*50)
    # 主循环
    while True:
        try:
            user_input = session.prompt(f"You({CURRENT_MODEL})> ").strip()
            if not user_input:
                continue
            # 命令处理
            if user_input.startswith('/'):
                handle_command(user_input, history)
                continue
            # 处理文件引用
            processed = process_file_references(user_input)
            history.append({"role": "user", "content": processed})
            # 获取当前模型配置
            cfg = MODELS[CURRENT_MODEL]
            # 截断历史
            history[:] = truncate_messages(history, cfg["context_window"], cfg["model"])
            # 检查API Key
            if not cfg["api_key"]:
                print(f"请填写 {CURRENT_MODEL} 的API Key")
                history.pop()
                continue
            # 初始化客户端
            client = OpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"], timeout=60)
            # 调用
            print("AI> ", end='', flush=True)
            try:
                resp = client.chat.completions.create(
                    model=cfg["model"],
                    messages=history,
                    max_tokens=cfg["max_tokens"]
                )
                content = resp.choices[0].message.content
                print(content)
                history.append({"role": "assistant", "content": content})
                # 保存历史
                with open(HISTORY_PATH, 'w', encoding='utf-8') as f:
                    json.dump(history, f, indent=2)
            except Exception as e:
                print(f"\n调用失败：{e}")
                if history and history[-1]["role"] == "user":
                    history.pop()
        except KeyboardInterrupt:
            print("\n再见！")
            break
        except Exception as e:
            print(f"\n错误：{e}")
            if history and history[-1]["role"] == "user":
                history.pop()

if __name__ == "__main__":
    main()
