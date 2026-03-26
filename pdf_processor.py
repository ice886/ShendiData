# pdf_processor.py
import asyncio
import os
import json
import re
import fitz  # PyMuPDF
import base64
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

client = AsyncOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
)

class PDFProcessor:
    def __init__(self, pdf_path: str, min_width: int = 100, min_height: int = 100):
        self.pdf_path = pdf_path
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
    
    def extract_images_with_captions(self, output_dir: str):
        """提取 PDF 中的所有图片及其题注"""
        # 图片保存在 image 子文件夹
        image_dir = os.path.join(output_dir, "image")
        os.makedirs(image_dir, exist_ok=True)
        
        print(f"📄 处理文件：{self.pdf_path}")
        print(f"📊 总页数：{len(self.doc)}")
        print(f"📁 输出目录：{output_dir}")
        print(f"🖼️  图片目录：{image_dir}")
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
                    
                    image_name = f"image_{len(self.images)+1}.{image_ext}"
                    image_path = os.path.join(image_dir, image_name)
                    
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
    
    async def generate_qa_pairs(self, output_dir: str):
        """根据题注生成问答对"""
        if not self.images:
            print("❌ 没有图片可处理")
            return
        
        qa_pairs = []
        output_jsonl = os.path.join(output_dir, "data.jsonl")
        
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
                system_prompt = """你是一个学术论文助手。请分析这张图片并生成问答对。

输出格式（纯 JSON，不要其他内容）：
{
  "question": "关于这张图片的问题",
  "answer": "详细的回答"
}

要求：
- 问题要有意义，与学术内容相关
- 回答要准确、详细
- 使用中文输出"""
                
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
                response = await client.chat.completions.create(
                    model="qwen3-vl-235b-a22b-thinking",
                    messages=messages,
                    max_tokens=1000
                )
                
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
                            "content": [
                                {"text": question},
                                {"image": img_info['name']}
                            ]
                        },
                        {
                            "role": "assistant",
                            "content": [
                                {"text": answer}
                            ]
                        }
                    ]
                }
                qa_pairs.append(qa_pair)
                print(f"   ✅ 生成成功")
                
            except Exception as e:
                print(f"   ❌ 生成失败：{e}")
                qa_pair = {
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"text": "这张图是什么内容"},
                                {"image": img_info['name']}
                            ]
                        },
                        {
                            "role": "assistant",
                            "content": [
                                {"text": "抱歉，无法分析这张图片。"}
                            ]
                        }
                    ]
                }
                qa_pairs.append(qa_pair)
        
        # 保存 JSONL
        with open(output_jsonl, 'w', encoding='utf-8') as f:
            for qa in qa_pairs:
                f.write(json.dumps(qa, ensure_ascii=False) + '\n')
        
        print(f"\n✅ 结果已保存到：{output_jsonl}")
        return qa_pairs
    
    def close(self):
        self.doc.close()

async def main():
    pdf_path = "/Users/ice/Desktop/shendi/data/test.pdf"
    output_dir = "/Users/ice/Desktop/shendi/data/Trainingdata_vl"
    
    if not os.path.exists(pdf_path):
        print(f"❌ 文件不存在：{pdf_path}")
        return
    
    print("="*60)
    print("📄 PDF 图片提取工具（增强版图题提取）")
    print("="*60)
    print(f"📁 输入文件：{pdf_path}")
    print(f"📁 输出目录：{output_dir}")
    print("="*60)
    
    processor = PDFProcessor(
        pdf_path=pdf_path,
        min_width=200,
        min_height=200
    )
    
    try:
        # 1. 提取图片（含题注）
        processor.extract_images_with_captions(output_dir)
        
        # 2. 根据题注生成问答对
        await processor.generate_qa_pairs(output_dir)
        
        print("\n" + "="*60)
        print("✅ 处理完成！")
        print("="*60)
        print(f"📁 输出目录：{output_dir}")
        print(f"🖼️  图片目录：{os.path.join(output_dir, 'image')}")
        print(f"📊 图片数量：{len(processor.images)}")
        print(f"📄 JSONL 文件：{os.path.join(output_dir, 'data.jsonl')}")
        print("="*60)
        
    finally:
        processor.close()

if __name__ == "__main__":
    asyncio.run(main())
