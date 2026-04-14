"""
中国矿产地质志钨矿卷上 扫描版PDF 转 Text 处理器
使用 OCR 技术从扫描版 PDF 中提取文本
"""

import fitz  # PyMuPDF
import os
import io
from PIL import Image
from paddleocr import PaddleOCR
import numpy as np


class ScannedPDFToTextConverter:
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.doc = fitz.open(pdf_path)
        # 初始化 PaddleOCR（支持中文）
        print("正在初始化 OCR 引擎...")
        self.ocr = PaddleOCR(use_textline_orientation=True, lang='ch')
        self.full_text = ""
        
    def _convert_page_to_image(self, page, zoom=2):
        """将 PDF 页面转换为高质量图片"""
        # 设置缩放比例以提高识别准确率
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        
        # 转换为 PIL Image
        img_bytes = pix.tobytes("png")
        img = Image.open(io.BytesIO(img_bytes))
        
        return img
    
    def extract_text_with_ocr(self, zoom=2, max_pages=None):
        """使用 OCR 从扫描版 PDF 中提取文本"""
        print(f"📄 处理文件：{os.path.basename(self.pdf_path)}")
        print(f"📊 总页数：{len(self.doc)}")
        if max_pages:
            print(f"🔬 测试模式：仅处理前 {max_pages} 页")
        print("="*60)

        full_text = ""
        total_pages = max_pages if max_pages else len(self.doc)
        
        for page_num in range(min(total_pages, len(self.doc))):
            page = self.doc[page_num]
            
            print(f"   处理第 {page_num + 1}/{total_pages} 页...")
            
            try:
                # 将页面转换为图片
                img = self._convert_page_to_image(page, zoom=zoom)
                
                # 转换为 numpy 数组供 OCR 使用
                img_np = np.array(img)
                
                # OCR 识别（使用 predict 方法）
                result = self.ocr.predict(img_np)
                
                # 提取识别的文本
                page_text = ""
                if result:
                    for line in result:
                        if len(line) >= 2:
                            page_text += line[1][0] + "\n"
                
                # 清理文本
                page_text = page_text.strip()
                
                if page_text:
                    # 添加页码标记
                    full_text += f"\n\n{'='*80}\n"
                    full_text += f"第 {page_num + 1} 页\n"
                    full_text += f"{'='*80}\n"
                    full_text += page_text
                
            except Exception as e:
                print(f"      ⚠️ 第 {page_num + 1} 页识别失败：{e}")
                full_text += f"\n\n{'='*80}\n"
                full_text += f"第 {page_num + 1} 页（识别失败）\n"
                full_text += f"{'='*80}\n"
        
        self.full_text = full_text
        
        print(f"✅ 提取完成")
        print(f"📝 文本总长度：{len(full_text)} 字符")
        print("="*60)
        
        return full_text
    
    def save_to_text(self, output_path: str):
        """将提取的文本保存到文件"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(self.full_text)
        
        print(f"💾 文本已保存到：{output_path}")
        print(f"📊 文件大小：{os.path.getsize(output_path) / 1024:.2f} KB")
        print("="*60)
    
    def close(self):
        self.doc.close()


def main():
    # 配置路径
    pdf_path = "/Users/ice/Desktop/shendi/shendi-data/data/中国矿产地质志钨矿卷上.pdf"
    output_path = "/Users/ice/Desktop/shendi/shendi-data/data/中国矿产地质志钨矿卷上_前15页.txt"
    
    # 测试模式：只处理前15页
    TEST_MODE = True
    MAX_PAGES = 30
    
    print("="*60)
    print("🚀 扫描版 PDF 转 Text 处理器（OCR）")
    if TEST_MODE:
        print(f"🔬 测试模式：仅处理前 {MAX_PAGES} 页")
    print("="*60)
    
    # 检查输入文件是否存在
    if not os.path.exists(pdf_path):
        print(f"❌ 错误：PDF 文件不存在")
        print(f"   路径：{pdf_path}")
        return
    
    # 创建转换器
    converter = ScannedPDFToTextConverter(pdf_path)
    
    try:
        # 使用 OCR 提取文本
        # zoom=2 提高分辨率以获得更好的识别效果
        # max_pages=MAX_PAGES 只处理前15页（测试模式）
        converter.extract_text_with_ocr(zoom=2, max_pages=MAX_PAGES if TEST_MODE else None)
        
        # 保存文本
        converter.save_to_text(output_path)
        
        print("✅ 处理完成！")
        
    except Exception as e:
        print(f"❌ 处理失败：{e}")
        import traceback
        traceback.print_exc()
    
    finally:
        converter.close()


if __name__ == "__main__":
    main()