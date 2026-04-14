import asyncio
import os
import json
import re
import fitz  # PyMuPDF
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

client = AsyncOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
)

# JSONL 文件名
JSONL_FILE = "qa_data.jsonl"

# 从环境变量读取模型列表，支持 JSON 数组或逗号分隔的格式
def get_model_list():
    model_str = os.getenv("TEXT_PROCESSOR_MODEL", "glm-5")
    try:
        # 尝试解析为 JSON 数组
        models = json.loads(model_str)
        if isinstance(models, list):
            return models
    except:
        pass
    
    # 如果不是 JSON，尝试逗号分隔
    models = [m.strip() for m in model_str.split(',')]
    return models if models else ["glm-5"]

MODEL_LIST = get_model_list()

# 状态记录文件名
STATUS_FILE = "processing_status.json"

def load_status(output_dir: str) -> dict:
    """加载处理状态，返回已处理的PDF文件列表"""
    status_file = os.path.join(output_dir, STATUS_FILE)
    if os.path.exists(status_file):
        try:
            with open(status_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️ 无法加载状态文件：{e}")
            return {"processed_files": [], "last_update": None}
    return {"processed_files": [], "last_update": None}

def save_status(output_dir: str, processed_files: list):
    """保存处理状态"""
    status_file = os.path.join(output_dir, STATUS_FILE)
    from datetime import datetime
    status = {
        "processed_files": processed_files,
        "last_update": datetime.now().isoformat()
    }
    try:
        with open(status_file, 'w', encoding='utf-8') as f:
            json.dump(status, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"⚠️ 无法保存状态文件：{e}")

def is_file_processed(status: dict, pdf_file: str, input_dir: str) -> bool:
    """检查PDF文件是否已处理"""
    # 使用相对路径作为唯一标识
    file_key = os.path.join(input_dir, pdf_file)
    return file_key in status.get("processed_files", [])

def append_to_jsonl(output_dir: str, qa_pairs: list):
    """将问答对追加到JSONL文件"""
    jsonl_file = os.path.join(output_dir, JSONL_FILE)
    try:
        with open(jsonl_file, 'a', encoding='utf-8') as f:
            for qa in qa_pairs:
                f.write(json.dumps(qa, ensure_ascii=False) + '\n')
    except Exception as e:
        print(f"⚠️ 无法写入JSONL文件：{e}")

async def call_llm_with_fallback(messages: list, max_tokens: int = 2000, current_model_index: list = [0]):
    """
    调用 LLM API，支持模型切换和重试
    
    Args:
        messages: 消息列表
        max_tokens: 最大 token 数
        current_model_index: 当前使用的模型索引（列表形式以便修改）
    
    Returns:
        response: API 响应
    """
    max_retries = len(MODEL_LIST)  # 最多尝试所有模型
    
    for attempt in range(max_retries):
        # 循环使用模型列表
        model_index = (current_model_index[0] + attempt) % len(MODEL_LIST)
        model_name = MODEL_LIST[model_index]
        
        try:
            response = await client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=max_tokens
            )
            
            # 成功后更新模型索引
            current_model_index[0] = model_index
            return response
            
        except Exception as e:
            error_str = str(e).lower()

            # 检查是否是需要切换模型的错误类型
            is_model_error = any(keyword in error_str for keyword in [
                # Token/配额错误
                'token', 'quota', 'limit', '429', 'rate limit', 'insufficient',
                # 认证错误
                'unauthorized', '401', 'authentication', 'invalid api key',
                # 模型不支持错误
                'http call', 'invalid_request_error', 'invalid_parameter_error',
                'model not found', 'model not supported', 'invalid model',
                # 服务不可用
                'service unavailable', '503', '502', '500',
                # 其他客户端错误
                '400', 'bad request'
            ])

            if is_model_error:
                # 切换到下一个模型
                next_model_index = (model_index + 1) % len(MODEL_LIST)
                print(f"   ⚠️ 模型 {model_name} 不可用（{e}）")
                print(f"   🔄 切换到模型 {MODEL_LIST[next_model_index]}")
                current_model_index[0] = next_model_index
                continue
            else:
                # 其他错误，直接抛出
                raise
    
    # 所有模型都尝试失败
    raise Exception(f"所有模型都不可用：{MODEL_LIST}")

class PaperQAAgent:
    def __init__(self, pdf_path: str, pdf_name: str = None, chunk_size: int = 2000, overlap: int = 200):
        self.pdf_path = pdf_path
        self.pdf_name = pdf_name or os.path.splitext(os.path.basename(pdf_path))[0]
        self.doc = fitz.open(pdf_path)
        self.text_chunks = []
        self.qa_pairs = []
        self.chunk_size = chunk_size  # 每段文本的字符数
        self.overlap = overlap  #  chunk 之间的重叠字符数
        
    def _clean_text(self, text):
        """清理文本中的换行符和多余空格"""
        if not text:
            return ""
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        return text
    
    def _extract_text_from_pages(self):
        """从 PDF 中提取所有文本内容"""
        full_text = ""
        for page_num in range(len(self.doc)):
            page = self.doc[page_num]
            text = page.get_text("text")
            text = self._clean_text(text)
            if text:
                full_text += f"[第{page_num + 1}页] {text}\n"
        return full_text
    
    def _chunk_text(self, text):
        """将长文本分割成适合处理的片段"""
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            # 尽量在句子边界处切割
            if end < len(text):
                # 查找最近的句号、问号、感叹号或换行
                break_chars = ['.', '。', '!', '！', '?', '？', '\n']
                for char in break_chars:
                    last_pos = text.rfind(char, start, end)
                    if last_pos > start + self.chunk_size // 2:
                        end = last_pos + 1
                        break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - self.overlap  # 重叠部分
        
        self.text_chunks = chunks
        return chunks
    
    def extract_text(self):
        """提取 PDF 中的所有文本内容"""
        print(f"📄 处理文件：{os.path.basename(self.pdf_path)}")
        print(f"📊 总页数：{len(self.doc)}")
        print("="*60)

        # 提取全文
        full_text = self._extract_text_from_pages()

        # 分块
        chunks = self._chunk_text(full_text)

        print(f"📝 提取文本总长度：{len(full_text)} 字符")
        print(f"📦 分块数量：{len(chunks)}")
        print("="*60)

        return chunks, full_text
    
    async def generate_qa_pairs(self, questions_per_chunk: int = 1):
        """从文本片段中生成问答对，返回 qa_pairs 列表"""
        if not self.text_chunks:
            print("❌ 没有文本可处理")
            return []

        qa_pairs = []
        current_model_index = [0]  # 使用列表以便在函数中修改

        # 系统提示词
        system_instruction = "你是一个专业的学术论文助手，擅长从学术文本中提取关键信息并生成高质量的问答对。"
        
        print(f"🤖 可用模型列表：{MODEL_LIST}")
        print(f"🎯 当前使用模型：{MODEL_LIST[current_model_index[0]]}")
        
        for i, chunk in enumerate(self.text_chunks, 1):
            print(f"\n 处理文本块 {i}/{len(self.text_chunks)}")
            
            user_prompt = f"""请从以下学术文本中提取 {questions_per_chunk} 个有意义的问答对。

                            文本内容：
                            {chunk}

                            要求：
                            1. 问题要基于文本内容，有实际意义
                            2. 回答要准确、完整，引用文本中的信息
                            3. 使用中文输出
                            4. 只输出 JSON 格式，不要其他内容
                            5. 不要涉及到人名信息，

                            输出格式：
                            [
                            {{
                                "question": "问题",
                                "answer": "回答"
                            }}
                            ]"""
            
            messages = [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_prompt}
            ]
            
            try:
                # 使用支持模型切换的 API 调用函数
                response = await call_llm_with_fallback(
                    messages=messages,
                    max_tokens=2000,
                    current_model_index=current_model_index
                )
                
                # 显示当前使用的模型
                current_model = MODEL_LIST[current_model_index[0]]
                if i == 1 or current_model_index[0] != (current_model_index[0] - 1):
                    print(f"   🎯 使用模型：{current_model}")
                
                qa_content = response.choices[0].message.content
                
                # 解析 JSON
                try:
                    # 尝试提取 JSON 数组
                    match = re.search(r'\[.*\]', qa_content, re.DOTALL)
                    if match:
                        qa_list = json.loads(match.group())
                    else:
                        qa_list = json.loads(qa_content)
                    
                    # 转换为指定格式
                    for qa in qa_list:
                        question = qa.get("question", "")
                        answer = qa.get("answer", "")

                        if question and answer:
                            qa_pair = {
                                "instruction": question,
                                "input": "",
                                "output": answer
                            }
                            qa_pairs.append(qa_pair)
                    
                    print(f"   ✅ 生成 {len(qa_list)} 个问答对")
                    
                except Exception as e:
                    print(f"   ⚠️ JSON 解析失败：{e}")
                    # 尝试从返回内容中提取问答
                    qa_pair = {
                        "instruction": "请总结这段文本的主要内容",
                        "input": "",
                        "output": qa_content
                    }
                    qa_pairs.append(qa_pair)
                
            except Exception as e:
                print(f"   ❌ 生成失败：{e}")
                # 失败时也要保持格式一致
                qa_pair = {
                    "instruction": "请总结这段文本的主要内容",
                    "input": "",
                    "output": "抱歉，无法处理这段文本。"
                }
                qa_pairs.append(qa_pair)

        self.qa_pairs = qa_pairs
        print(f"\n✅ 共生成 {len(qa_pairs)} 个问答对")
        return qa_pairs
    
    def close(self):
        self.doc.close()


async def batch_process(base_dir: str, output_base_dir: str):
    """批量处理多个目录下的 PDF 文件，生成统一的 JSONL 文件"""
    print("="*60)
    print("🚀 文本批量处理工具")
    print("="*60)
    print(f"📁 基础输入目录：{base_dir}")
    print(f"📁 基础输出目录：{output_base_dir}")
    print("="*60)
    print(f"🤖 可用模型：{MODEL_LIST}")
    print(f"🎯 主模型：{MODEL_LIST[0]}")
    print(f"🔄 支持自动切换模型（token 用完时）")
    print("="*60)

    # 创建输出目录
    os.makedirs(output_base_dir, exist_ok=True)

    # 加载处理状态（断点重启支持）
    status = load_status(output_base_dir)
    processed_files = status.get("processed_files", [])

    if processed_files:
        print(f"🔄 检测到断点，已处理 {len(processed_files)} 个文件")
        if status.get("last_update"):
            print(f"⏰ 最后更新时间：{status['last_update']}")
        print(f"⏭️  将跳过已处理的文件")
        print("="*60)

    total_files = 0
    total_qa_pairs_count = 0
    failed_files = 0
    skipped_files = 0

    # 获取所有子目录
    subdirs = []
    if os.path.isdir(base_dir):
        # 检查是单个目录还是包含多个子目录
        subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

    if not subdirs:
        # 如果没有子目录，直接处理当前目录的 PDF 文件
        subdirs = [""]

    # 统计总文件数
    for subdir in subdirs:
        input_dir = os.path.join(base_dir, subdir) if subdir else base_dir
        pdf_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.pdf')]
        total_files += len(pdf_files)

    print(f"📊 找到 {total_files} 个 PDF 文件")
    print("="*60)

    # 处理每个子目录
    for i, subdir in enumerate(subdirs, 1):
        input_dir = os.path.join(base_dir, subdir) if subdir else base_dir

        # 获取所有 PDF 文件
        pdf_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith('.pdf')])

        print(f"\n📂 处理目录 {i}/{len(subdirs)}: {subdir or '根目录'} ({len(pdf_files)} 个文件)")
        print("-" * 60)

        # 处理每个 PDF 文件
        for j, pdf_file in enumerate(pdf_files, 1):
            print(f"\n[{i}/{len(subdirs)}] [{j}/{len(pdf_files)}] 处理文件：{pdf_file}")

            # 检查是否已处理
            if is_file_processed(status, pdf_file, input_dir):
                print(f"   ⏭️  已处理，跳过")
                skipped_files += 1
                continue

            pdf_path = os.path.join(input_dir, pdf_file)
            pdf_name = os.path.splitext(pdf_file)[0]

            try:
                agent = PaperQAAgent(
                    pdf_path=pdf_path,
                    pdf_name=pdf_name,
                    chunk_size=2000,
                    overlap=200
                )

                # 提取文本
                chunks, full_text = agent.extract_text()

                # 生成问答对
                qa_pairs = await agent.generate_qa_pairs(questions_per_chunk=1)

                if qa_pairs:
                    # ✅ 立即写入JSONL文件
                    append_to_jsonl(output_base_dir, qa_pairs)
                    total_qa_pairs_count += len(qa_pairs)
                    print(f"   ✅ 成功生成 {len(qa_pairs)} 个问答对，已写入 qa_data.jsonl")
                else:
                    print(f"   ⚠️ 未生成问答对")

                # ✅ 更新处理状态
                file_key = os.path.join(input_dir, pdf_file)
                processed_files.append(file_key)
                save_status(output_base_dir, processed_files)
                print(f"   💾 状态已更新")

                agent.close()

            except Exception as e:
                print(f"   ❌ 处理失败：{e}")
                failed_files += 1
                # 失败的文件也要记录到状态中，避免重复处理失败的文件
                file_key = os.path.join(input_dir, pdf_file)
                if file_key not in processed_files:
                    processed_files.append(file_key)
                    save_status(output_base_dir, processed_files)

    # 汇总统计
    print("\n" + "="*60)
    print("📊 批量处理完成！")
    print("="*60)
    print(f"📁 总文件数：{total_files}")
    print(f"✅ 成功处理：{total_files - failed_files - skipped_files}")
    print(f"⏭️  跳过已处理：{skipped_files}")
    print(f"❌ 失败文件：{failed_files}")
    print(f"💬 总问答对数：{total_qa_pairs_count}")
    print(f"📄 JSONL 文件：{os.path.join(output_base_dir, JSONL_FILE)}")
    print(f"💾 状态文件：{os.path.join(output_base_dir, STATUS_FILE)}")
    print("="*60)
    print(f"💡 提示：如果处理中断，再次运行程序将从断点继续")


async def main():
    # 配置参数
    base_dir = "/Users/ice/Desktop/shendi/shendi-data"
    output_base_dir = "/Users/ice/Desktop/shendi/shendi-data/Trainingdata_text"

    # 批量处理
    await batch_process(base_dir, output_base_dir)


if __name__ == "__main__":
    asyncio.run(main())
