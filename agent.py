# agent.py
import asyncio
import os
from openai import AsyncOpenAI
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
import json
import dotenv

dotenv.load_dotenv()

MINERU_API_TOKEN = os.getenv("MINERU_TOKEN")
os.environ["MINERU_API_TOKEN"] = MINERU_API_TOKEN

client = AsyncOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
)

MCP_URL = os.getenv("MCP_URL", "https://mcp.mineru.net/mcp")

async def run_openai_mcp_agent():
    print(f"🔑 Token 前缀：{MINERU_API_TOKEN[:20] if MINERU_API_TOKEN else 'None'}...")
    print(f"🌐 MCP URL: {MCP_URL}")
    
    async with streamable_http_client(url=MCP_URL) as result:
        if len(result) == 3:
            read_stream, write_stream, get_session_id = result
        else:
            read_stream, write_stream = result
        
        async with ClientSession(read_stream, write_stream) as mcp_session:
            await mcp_session.initialize()
            print("✅ MCP 会话初始化成功")

            mineru_tools = await mcp_session.list_tools()
            print(f"📦 可用工具：{len(mineru_tools.tools)} 个")
            
            for tool in mineru_tools.tools:
                desc = tool.description if tool.description else "无描述"
                print(f"   - {tool.name}: {desc[:50]}...")
            
            # ✅ 修复：使用 inputSchema 而不是 parameters
            openai_tools = []
            for t in mineru_tools.tools:
                openai_tools.append({
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description if t.description else "No description",
                        "parameters": t.inputSchema  # ✅ 使用 inputSchema
                    }
                })

            # ⚠️ 注意：MinerU 无法访问本地文件，需要使用远程 URL 或上传 UI
            messages = [
                {"role": "system", "content": """你是一个学术论文图文数据提取助手。你的任务是从论文中提取图片数据，并按照 JSONL 格式输出。

重要提示：
- MinerU 服务器无法访问本地文件路径
- 如果用户提供本地文件路径，请调用 open_upload_ui 工具获取上传链接
- 如果用户有远程 URL，直接使用 parse_documents 工具

输出格式要求：
- 每行一个 JSON 对象
- 必须包含 "messages" 字段，值为数组
- 每个消息包含 "role" 和 "content" 字段
- content 是数组，可以包含 {"text": "..."} 或 {"image": "文件名"}
- 图片文件名和 JSON 文件放在同一文件夹

示例格式：
{"messages":[{"role":"user","content":[{"text":"这张图描述了什么？"},{"image":"image_1.png"}]},{"role":"assistant","content":[{"text":"这是一张显示实验结果的图表，展示了..."}]}]}

注意事项：
1. 为每张图片生成有意义的问答对
2. 问题和回答应该与论文内容相关
3. 图片文件名保持原样或使用描述性名称
4. 每张图片生成一个 JSON 对象"""},
                # ✅ 修改：提示用户需要上传文件或提供 URL
                {"role": "user", "content": """请从论文中提取所有图片，并为每张图片生成问答对，输出为 JSONL 格式。

注意：由于 MinerU 无法访问本地文件，请先调用 open_upload_ui 工具获取上传链接，让用户上传 data/test.pdf 文件。上传完成后，再使用 parse_documents 工具处理文件。

文件路径：data/test.pdf
语言：Chinese"""}
            ]

            print("\n🤖 正在调用大模型...")
            response = await client.chat.completions.create(
                model="deepseek-v3.1",
                messages=messages,
                tools=openai_tools,
                tool_choice="auto"
            )

            response_message = response.choices[0].message
            tool_calls = response_message.tool_calls

            if tool_calls:
                print(f"📞 模型决定调用 {len(tool_calls)} 个工具")
                messages.append(response_message)

                for tool_call in tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)

                    print(f"\n🛠️ 正在调用 MinerU 工具：{function_name}")
                    print(f"   参数：{function_args}")

                    tool_result = await mcp_session.call_tool(function_name, arguments=function_args)

                    messages.append({
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": str(tool_result.content)
                    })
                    print(f"   ✅ 工具调用完成")
                    print(f"   结果：{tool_result.content[:200]}...")

                print("\n🤖 正在获取最终响应...")
                final_response = await client.chat.completions.create(
                    model="deepseek-v3.1",
                    messages=messages,
                    tools=openai_tools
                )

                final_message = final_response.choices[0].message.content
                print(f"\n📊 最终响应:\n{final_message}")

                output_path = "/Users/ice/Desktop/shendi/data/data.jsonl"
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(final_message)

                print(f"\n✅ 结果已保存到：{output_path}")

            else:
                print("❌ 模型未调用任何工具")
                print(f"模型响应：{response_message.content}")


if __name__ == "__main__":
    asyncio.run(run_openai_mcp_agent())
