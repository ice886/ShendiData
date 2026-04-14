# pdf_processor.py
import asyncio
import os
import json
import re
import fitz  # PyMuPDF
import base64
from dotenv import load_dotenv
from openai import AsyncOpenAI
from datetime import datetime

load_dotenv()

client = AsyncOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
)

# 状态记录文件名
STATUS_FILE = "processing_status.json"
JSONL_FILE = "data.jsonl"

# 从环境变量读取模型列表，支持 JSON 数组或逗号分隔的格式
def get_model_list():
    model_str = os.getenv("PDF_PROCESSOR_MODEL", "qwen3-vl-235b-a22b-thinking")
    try:
        # 尝试解析为 JSON 数组
        models = json.loads(model_str)
        if isinstance(models, list):
            return models
    except:
        pass

    # 如果不是 JSON，尝试逗号分隔
    models = [m.strip() for m in model_str.split(',')]
    return models if models else ["qwen3-vl-235b-a22b-thinking"]

MODEL_LIST = get_model_list()

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
    status = {
        "processed_files": processed_files,
        "last_update": datetime.now().isoformat()
    }
    try:
        with open(status_file, 'w', encoding='utf-8') as f:
            json.dump(status, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"⚠️ 无法保存状态文件：{e}")

def append_to_jsonl(output_dir: str, qa_pairs: list):
    """将问答对追加到JSONL文件"""
    jsonl_file = os.path.join(output_dir, JSONL_FILE)
    try:
        with open(jsonl_file, 'a', encoding='utf-8') as f:
            for qa in qa_pairs:
                f.write(json.dumps(qa, ensure_ascii=False) + '\n')
    except Exception as e:
        print(f"⚠️ 无法写入JSONL文件：{e}")

async def call_llm_with_fallback(messages: list, max_tokens: int = 1000, current_model_index: list = [0]):
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

def is_file_processed(status: dict, pdf_file: str, input_dir: str) -> bool:
    """检查PDF文件是否已处理"""
    # 使用相对路径作为唯一标识
    file_key = os.path.join(input_dir, pdf_file)
    return file_key in status.get("processed_files", [])

class PDFProcessor:
    def __init__(self, pdf_path: str, pdf_name: str = None, min_width: int = 100, min_height: int = 100):
        self.pdf_path = pdf_path
        self.pdf_name = pdf_name or os.path.splitext(os.path.basename(pdf_path))[0]
        self.doc = fitz.open(pdf_path)
        self.images = []
        self.min_width = min_width
        self.min_height = min_height
        
    def is_valid_image_size(self, image_bytes: bytes) -> bool:
        """检查图片尺寸是否有效"""
        try:
            img = fitz.open(stream=image_bytes, filetype="png")
            if img.page_count > 0:
                page = img[0]
                width = page.rect.width
                height = page.rect.height
                is_valid = width >= self.min_width and height >= self.min_height
                img.close()
                return is_valid
        except Exception as e:
            print(f"   ⚠️ 无法检查图片尺寸：{e}")
            return True
        return True
    
    def _clean_text(self, text):
        """清理文本中的换行符和多余空格"""
        if not text:
            return ""
        # 替换多个空格为单个空格
        text = re.sub(r'\s+', ' ', text)
        # 去除首尾空格
        text = text.strip()
        return text
    
    def _extract_figure_number_from_caption(self, caption):
        """从 caption 中提取图片编号"""
        if not caption:
            return None
        
        # 匹配 "Fig. 9", "Figure 9", "Fig 9" 等格式
        match = re.search(r'Fig(?:ure)?\.?\s*(\d+)', caption, re.IGNORECASE)
        if match:
            return match.group(1)
        
        # 匹配中文格式 "图 1", "图 1:", "图 1."
        match = re.search(r'图\s*(\d+)', caption)
        if match:
            return match.group(1)
        
        return None
    
    def _extract_caption_from_text(self, text):
        """从文本中提取图题"""
        if not text:
            return ""
        
        lines = text.split('\n')
        caption = ""
        
        # 查找包含 Figure/Fig. 或 图 的行
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            # 改进的正则表达式，匹配更多格式
            if re.search(r'^(Fig(?:ure)?\.?\s*\d+[\.\:\-]?\s*)', line, re.IGNORECASE) or \
               re.search(r'^(图\s*\d+[\.\:\-]?\s*)', line):
                # 提取该行及其后的几行作为完整的 caption
                caption = line
                # 继续读取后续行，直到遇到空行或另一个图题
                for j in range(i + 1, len(lines)):
                    next_line = lines[j].strip()
                    
                    # 遇到空行，停止
                    if not next_line:
                        break
                    
                    # 遇到另一个图题，停止
                    if re.match(r'^(Fig(?:ure)?\.?\s*\d+)', next_line, re.IGNORECASE) or \
                       re.match(r'^(图\s*\d+)', next_line):
                        break
                    
                    # 添加到 caption
                    caption += " " + next_line
                
                break
        
        # 清理 caption
        caption = self._clean_text(caption)
        
        return caption
    
    def _extract_caption_near_image(self, page, image_rect, figure_number=None):
        """在图片附近提取图题"""
        # 同时搜索图片上方和下方的文本区域
        
        # 上方搜索（有些论文的 caption 在图片上方）
        search_rect_above = fitz.Rect(
            image_rect.x0 - 200,
            max(0, image_rect.y0 - 200),
            image_rect.x1 + 200,
            image_rect.y0
        )
        
        # 下方搜索（通常 caption 在图片下方）
        search_rect_below = fitz.Rect(
            image_rect.x0 - 200,
            image_rect.y1,
            image_rect.x1 + 200,
            image_rect.y1 + 300
        )
        
        # 先搜索下方
        text = page.get_text("text", clip=search_rect_below)
        caption = self._extract_caption_from_text(text)
        
        if not caption:
            # 再搜索上方
            text = page.get_text("text", clip=search_rect_above)
            caption = self._extract_caption_from_text(text)
        
        return caption
    
    def extract_images_with_captions(self, output_dir: str, images_dir: str = None):
        """提取 PDF 中的所有图片及其题注"""
        # 如果提供了 images_dir，使用统一的图片目录
        if images_dir:
            image_output_dir = images_dir
        else:
            image_output_dir = output_dir

        # 创建输出目录
        os.makedirs(image_output_dir, exist_ok=True)

        print(f"📄 处理文件：{self.pdf_path}")
        print(f"📊 总页数：{len(self.doc)}")
        print(f"📁 输出目录：{output_dir}")
        print(f"🖼️  图片目录：{image_output_dir}")
        print(f"🔍 最小图片尺寸：{self.min_width}x{self.min_height} 像素")
        print("="*60)
        
        total_images = 0
        valid_images = 0
        filtered_count = 0
        
        for page_num in range(len(self.doc)):
            page = self.doc[page_num]
            image_list = page.get_images(full=True)
            
            for img_index, img in enumerate(image_list):
                xref = img[0]
                total_images += 1
                
                try:
                    base_image = self.doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]
                    
                    # 过滤小图片
                    if not self.is_valid_image_size(image_bytes):
                        filtered_count += 1
                        print(f"   ⚠️ 过滤小图片：page{page_num+1}_img{img_index+1}.{image_ext}")
                        continue
                    
                    valid_images += 1

                    # 使用 pdf_name 前缀生成唯一图片名
                    image_name = f"{self.pdf_name}_image_{len(self.images)+1}.{image_ext}"
                    image_path = os.path.join(image_output_dir, image_name)

                    with open(image_path, "wb") as f:
                        f.write(image_bytes)
                    
                    # 获取图片位置
                    img_rects = page.get_image_rects(xref)
                    img_rect = img_rects[0] if img_rects else None
                    
                    # ✅ 提取图题编号
                    figure_number = None
                    
                    # ✅ 在图片附近提取图题
                    caption = self._extract_caption_near_image(page, img_rect, figure_number)
                    
                    # 从 caption 中提取图题编号
                    if caption:
                        figure_number = self._extract_figure_number_from_caption(caption)
                    
                    self.images.append({
                        "page": page_num + 1,
                        "index": img_index + 1,
                        "path": image_path,
                        "name": image_name,
                        "caption": caption,
                        "figure_number": figure_number
                    })
                    
                    print(f"   ✅ 提取：{image_name}")
                    if caption:
                        caption_preview = caption[:60] + "..." if len(caption) > 60 else caption
                        print(f"      📝 题注：{caption_preview}")
                        if figure_number:
                            print(f"      🔢 编号：Fig. {figure_number}")
                    else:
                        print(f"      ⚠️ 未找到题注")
                    
                except Exception as e:
                    print(f"   ⚠️ 跳过图片：{e}")
        
        print("="*60)
        print(f"📊 统计:")
        print(f"   总图片数：{total_images}")
        print(f"   有效图片：{valid_images}")
        print(f"   过滤小图：{filtered_count}")
        print(f"   共提取 {len(self.images)} 张图片")
        
        with_captions = sum(1 for img in self.images if img.get('caption'))
        print(f"   有题注：{with_captions}/{len(self.images)}")
        print("="*60)
        
        return self.images
    
    async def generate_qa_pairs(self, output_dir: str = None):
        """根据题注生成问答对，返回 qa_pairs 列表"""
        if not self.images:
            print("❌ 没有图片可处理")
            return []

        qa_pairs = []
        current_model_index = [0]  # 使用列表以便在函数中修改

        print(f"🤖 可用模型列表：{MODEL_LIST}")
        print(f"🎯 当前使用模型：{MODEL_LIST[current_model_index[0]]}")

        for i, img_info in enumerate(self.images, 1):
            print(f"\n🤖 处理图片 {i}/{len(self.images)}: {img_info['name']}")

            caption_text = img_info.get('caption', '')
            figure_number = img_info.get('figure_number', '')

            # 根据是否有题注构建不同的提示
            if caption_text:
                figure_info = f"（图 {figure_number}）" if figure_number else ""
                system_prompt = f"""你是一个学术论文助手。请根据图片题注生成问答对。

                图片题注{figure_info}：{caption_text}

                输出格式（纯 JSON，不要其他内容）：
                {{
                "question": "基于题注提出的问题",
                "answer": "详细的回答"
                }}

                要求：
                - 问题要基于题注内容，有意义
                - 回答要准确、详细，可以扩展题注信息
                - 使用中文输出
                - 不要提及"题注"这个词，直接回答问题"""



                user_prompt = f"请根据以下题注生成问答对：{caption_text}"
            else:
                system_prompt = """
                你是一个学术论文助手。请分析这张图片并生成问答对。

                输出格式（纯 JSON，不要其他内容）：
                {
                "question": "关于这张图片的问题",
                "answer": "详细的回答"
                }

                要求：
                - 问题要有意义，与学术内容相关
                - 回答要准确、详细
                - 使用中文输出
                """

                user_prompt = "请分析这张学术论文中的图片，生成问答对。"

            # 读取图片并编码为 base64
            with open(img_info['path'], 'rb') as f:
                image_data = base64.b64encode(f.read()).decode('utf-8')

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_data}"}}
                ]}
            ]

            try:
                # 使用支持模型切换的 API 调用函数
                response = await call_llm_with_fallback(
                    messages=messages,
                    max_tokens=1000,
                    current_model_index=current_model_index
                )

                # 显示当前使用的模型
                current_model = MODEL_LIST[current_model_index[0]]
                if i == 1 or current_model_index[0] != (current_model_index[0] - 1):
                    print(f"   🎯 使用模型：{current_model}")

                qa_content = response.choices[0].message.content

                # 解析 JSON
                try:
                    qa_data = json.loads(qa_content)
                    question = qa_data.get("question", "这张图是什么内容")
                    answer = qa_data.get("answer", "无法生成回答")
                except:
                    question = "这张图是什么内容"
                    answer = qa_content

                # 严格按照要求的格式
                qa_pair = {
                    "messages": [
                        {
                            "role": "user",
                            "content": f"<image>{question}"
                        },
                        {
                            "role": "assistant",
                            "content": answer
                        }
                    ],
                    "images": [img_info['name']]
                }
                qa_pairs.append(qa_pair)
                print(f"   ✅ 生成成功")

            except Exception as e:
                print(f"   ❌ 生成失败：{e}")
                qa_pair = {
                    "messages": [
                        {
                            "role": "user",
                            "content": "<image>这张图是什么内容"
                        },
                        {
                            "role": "assistant",
                            "content": "抱歉，无法分析这张图片。"
                        }
                    ],
                    "images": [img_info['name']]
                }
                qa_pairs.append(qa_pair)

        print(f"\n✅ 生成了 {len(qa_pairs)} 个问答对")
        return qa_pairs
    
    def close(self):
        self.doc.close()

async def batch_process(base_dir: str, output_base_dir: str, pattern: str = "*.pdf"):
    """批量处理多个目录下的 PDF 文件，生成统一的 JSONL 文件，支持断点重启"""
    print("="*60)
    print("🚀 PDF 批量处理工具")
    print("="*60)
    print(f"📁 基础输入目录：{base_dir}")
    print(f"📁 基础输出目录：{output_base_dir}")
    print("="*60)

    # 创建统一的输出目录和图片目录
    os.makedirs(output_base_dir, exist_ok=True)
    images_dir = os.path.join(output_base_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    # 加载处理状态（断点重启支持）
    status = load_status(output_base_dir)
    processed_files = status.get("processed_files", [])
    
    if processed_files:
        print(f"🔄 检测到断点，已处理 {len(processed_files)} 个文件")
        if status.get("last_update"):
            print(f"⏰ 最后更新时间：{status['last_update']}")
        print(f"⏭️  将跳过已处理的文件")
        print("="*60)

    print(f"🤖 可用模型：{MODEL_LIST}")
    print(f"🎯 主模型：{MODEL_LIST[0]}")
    print(f"🔄 支持自动切换模型（token 用完时）")
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
    print(f"🖼️  图片将保存到：{images_dir}")
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
                processor = PDFProcessor(
                    pdf_path=pdf_path,
                    pdf_name=pdf_name,
                    min_width=200,
                    min_height=200
                )

                # 提取图片（保存到统一的 images 目录）
                processor.extract_images_with_captions(output_base_dir, images_dir)

                # 生成问答对
                qa_pairs = await processor.generate_qa_pairs()

                if qa_pairs:
                    # ✅ 立即写入JSONL文件
                    append_to_jsonl(output_base_dir, qa_pairs)
                    total_qa_pairs_count += len(qa_pairs)
                    print(f"   ✅ 成功生成 {len(qa_pairs)} 个问答对，已写入 data.jsonl")
                else:
                    print(f"   ⚠️ 未生成问答对")

                # ✅ 更新处理状态
                file_key = os.path.join(input_dir, pdf_file)
                processed_files.append(file_key)
                save_status(output_base_dir, processed_files)
                print(f"   💾 状态已更新")

                processor.close()

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
    print(f"🖼️  图片目录：{images_dir}")
    print(f"📄 JSONL 文件：{os.path.join(output_base_dir, JSONL_FILE)}")
    print(f"💾 状态文件：{os.path.join(output_base_dir, STATUS_FILE)}")
    print("="*60)
    print(f"💡 提示：如果处理中断，再次运行程序将从断点继续")

async def main():
    # 配置参数
    base_dir = "/Users/ice/Desktop/shendi/shendi-data"
    output_base_dir = "/Users/ice/Desktop/shendi/shendi-data/Trainingdata_vl"
    
    # 批量处理
    await batch_process(base_dir, output_base_dir)

if __name__ == "__main__":
    asyncio.run(main())