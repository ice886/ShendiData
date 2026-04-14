import asyncio
import os
import json
import re
import fitz  # PyMuPDF
from dotenv import load_dotenv

load_dotenv()


class PretrainDataProcessor:
    def __init__(self, pdf_path: str, pdf_name: str = None):
        self.pdf_path = pdf_path
        self.pdf_name = pdf_name or os.path.splitext(os.path.basename(pdf_path))[0]
        self.doc = fitz.open(pdf_path)

    def _clean_text(self, text):
        """清理文本中的换行符和多余空格"""
        if not text:
            return ""
        # 替换多个空格为单个空格
        text = re.sub(r'\s+', ' ', text)
        # 去除首尾空格
        text = text.strip()
        return text

    def extract_text(self):
        """提取 PDF 中的所有文本内容"""
        print(f"📄 处理文件：{os.path.basename(self.pdf_path)}")
        print(f"📊 总页数：{len(self.doc)}")
        print("="*60)

        full_text = ""
        for page_num in range(len(self.doc)):
            page = self.doc[page_num]
            text = page.get_text("text")
            text = self._clean_text(text)
            if text:
                full_text += text + "\n"

        print(f"📝 提取文本总长度：{len(full_text)} 字符")
        print("="*60)

        return full_text

    def close(self):
        self.doc.close()


async def batch_process(base_dir: str, output_base_dir: str):
    """批量处理多个目录下的 PDF 文件，生成预训练数据"""
    print("="*60)
    print("🚀 预训练数据批量生成工具")
    print("="*60)
    print(f"📁 基础输入目录：{base_dir}")
    print(f"📁 基础输出目录：{output_base_dir}")
    print("="*60)

    # 创建输出目录
    os.makedirs(output_base_dir, exist_ok=True)

    total_files = 0
    pretrain_data = []
    failed_files = 0

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

            pdf_path = os.path.join(input_dir, pdf_file)
            pdf_name = os.path.splitext(pdf_file)[0]

            try:
                processor = PretrainDataProcessor(
                    pdf_path=pdf_path,
                    pdf_name=pdf_name
                )

                # 提取文本
                full_text = processor.extract_text()

                if full_text:
                    # 添加到预训练数据
                    pretrain_data.append({
                        "text": full_text
                    })
                    print(f"   ✅ 成功提取文本")
                else:
                    print(f"   ⚠️ 文本为空")

                processor.close()

            except Exception as e:
                print(f"   ❌ 处理失败：{e}")
                failed_files += 1

    # 保存 JSONL 文件
    output_jsonl = os.path.join(output_base_dir, "pretrain_data.jsonl")
    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for data in pretrain_data:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')

    # 汇总统计
    print("\n" + "="*60)
    print("📊 批量处理完成！")
    print("="*60)
    print(f"📁 处理文件数：{total_files}")
    print(f"✅ 成功处理：{total_files - failed_files}")
    print(f"❌ 失败文件：{failed_files}")
    print(f"📝 预训练数据条数：{len(pretrain_data)}")
    print(f"📄 JSONL 文件：{output_jsonl}")
    print("="*60)

    return pretrain_data


async def main():
    # 配置参数
    base_dir = "/Users/ice/Desktop/shendi/shendi-data"
    output_base_dir = "/Users/ice/Desktop/shendi/shendi-data/Trainingdata_pretrain"

    # 批量处理
    await batch_process(base_dir, output_base_dir)


if __name__ == "__main__":
    asyncio.run(main())