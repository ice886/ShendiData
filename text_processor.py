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

class PaperQAAgent:
    def __init__(self, pdf_path: str, chunk_size: int = 2000, overlap: int = 200):
        self.pdf_path = pdf_path
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
    
    def extract_text(self, output_dir: str):
        """提取 PDF 中的所有文本内容"""
        print(f"📄 处理文件：{self.pdf_path}")
        print(f"📊 总页数：{len(self.doc)}")
        print(f"📁 输出目录：{output_dir}")
        print("="*60)
        
        # 提取全文
        full_text = self._extract_text_from_pages()
        
        # 保存原始文本
        text_file = os.path.join(output_dir, "full_text.txt")
        with open(text_file, 'w', encoding='utf-8') as f:
            f.write(full_text)
        
        # 分块
        chunks = self._chunk_text(full_text)
        
        print(f"📝 提取文本总长度：{len(full_text)} 字符")
        print(f"📦 分块数量：{len(chunks)}")
        print(f"📁 文本文件：{text_file}")
        print("="*60)
        
        return chunks
    
    async def generate_qa_pairs(self, output_dir: str, questions_per_chunk: int = 2):
        """从文本片段中生成问答对"""
        if not self.text_chunks:
            print("❌ 没有文本可处理")
            return
        
        qa_pairs = []
        output_jsonl = os.path.join(output_dir, "qa_data.jsonl")
        
        # 系统提示词
        system_instruction = "你是一个专业的学术论文助手，擅长从学术文本中提取关键信息并生成高质量的问答对。"
        
        for i, chunk in enumerate(self.text_chunks, 1):
            print(f"\n🤖 处理文本块 {i}/{len(self.text_chunks)}")
            
            user_prompt = f"""请从以下学术文本中提取 {questions_per_chunk} 个有意义的问答对。

文本内容：
{chunk}

要求：
1. 问题要基于文本内容，有实际意义
2. 回答要准确、完整，引用文本中的信息
3. 使用中文输出
4. 只输出 JSON 格式，不要其他内容

输出格式：
[
  {{
    "question": "问题 1",
    "answer": "回答 1"
  }},
  {{
    "question": "问题 2", 
    "answer": "回答 2"
  }}
]"""
            
            messages = [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_prompt}
            ]
            
            try:
                response = await client.chat.completions.create(
                    model="glm-5",
                    messages=messages,
                    max_tokens=2000
                )
                
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
                                "messages": [
                                    {
                                        "role": "system",
                                        "content": system_instruction
                                    },
                                    {
                                        "role": "user",
                                        "content": question
                                    },
                                    {
                                        "role": "assistant",
                                        "content": answer
                                    }
                                ]
                            }
                            qa_pairs.append(qa_pair)
                    
                    print(f"   ✅ 生成 {len(qa_list)} 个问答对")
                    
                except Exception as e:
                    print(f"   ⚠️ JSON 解析失败：{e}")
                    # 尝试从返回内容中提取问答
                    qa_pair = {
                        "messages": [
                            {
                                "role": "system",
                                "content": system_instruction
                            },
                            {
                                "role": "user",
                                "content": "请总结这段文本的主要内容"
                            },
                            {
                                "role": "assistant",
                                "content": qa_content
                            }
                        ]
                    }
                    qa_pairs.append(qa_pair)
                
            except Exception as e:
                print(f"   ❌ 生成失败：{e}")
                # 失败时也要保持格式一致
                qa_pair = {
                    "messages": [
                        {
                            "role": "system",
                            "content": system_instruction
                        },
                        {
                            "role": "user",
                            "content": "请总结这段文本的主要内容"
                        },
                        {
                            "role": "assistant",
                            "content": "抱歉，无法处理这段文本。"
                        }
                    ]
                }
                qa_pairs.append(qa_pair)
        
        # 保存 JSONL
        with open(output_jsonl, 'w', encoding='utf-8') as f:
            for qa in qa_pairs:
                f.write(json.dumps(qa, ensure_ascii=False) + '\n')
        
        self.qa_pairs = qa_pairs
        print(f"\n✅ 结果已保存到：{output_jsonl}")
        print(f"📊 共生成 {len(qa_pairs)} 个问答对")
        return qa_pairs
    
    def close(self):
        self.doc.close()


async def main():
    pdf_path = "/Users/ice/Desktop/shendi/data/test.pdf"
    output_dir = "/Users/ice/Desktop/shendi/data/Trainingdata_text"
    
    if not os.path.exists(pdf_path):
        print(f"❌ 文件不存在：{pdf_path}")
        return
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*60)
    print("📄 论文问答数据提取 Agent（文本版）")
    print("="*60)
    print(f"📁 输入文件：{pdf_path}")
    print(f"📁 输出目录：{output_dir}")
    print("="*60)
    
    agent = PaperQAAgent(
        pdf_path=pdf_path,
        chunk_size=2000,
        overlap=200
    )
    
    try:
        # 1. 提取文本
        agent.extract_text(output_dir)
        
        # 2. 从文本生成问答对
        await agent.generate_qa_pairs(output_dir, questions_per_chunk=2)
        
        print("\n" + "="*60)
        print("✅ 处理完成！")
        print("="*60)
        print(f"📁 输出目录：{output_dir}")
        print(f"📄 文本文件：{os.path.join(output_dir, 'full_text.txt')}")
        print(f"📊 问答对数量：{len(agent.qa_pairs)}")
        print(f"📄 JSONL 文件：{os.path.join(output_dir, 'qa_data.jsonl')}")
        print("="*60)
        
        # 显示示例
        if agent.qa_pairs:
            print("\n📋 示例问答对：")
            print(json.dumps(agent.qa_pairs[0], ensure_ascii=False, indent=2))
        
    finally:
        agent.close()


if __name__ == "__main__":
    asyncio.run(main())
