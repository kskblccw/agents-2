#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据库模块 (database.py)
======================
只负责与向量数据库和文件相关的操作

包含功能：
1. 读取职位数据 (jobs.json)
2. 初始化向量数据库 (Chroma)
3. 提供 Embeddings 封装类

注意：
- 不负责读取简历数据（由 resume_parser.py 负责）
- 不负责调用 GLM-4 API（由需要的地方自行调用）
"""

# 导入必要的库
import os           # 用于操作文件和环境变量
import json         # 用于处理 JSON 格式的数据
import re           # 用于正则表达式
import shutil       # 用于删除目录等文件操作

# LangChain 相关库
from langchain_community.vectorstores import Chroma  # 向量数据库
from langchain.embeddings.base import Embeddings     # Embeddings 基类
from langchain_core.documents import Document        # 文档类

# =============================================================================
# 第一部分：阿里云百炼 Embeddings 封装类
# =============================================================================

class AliyunEmbeddings(Embeddings):
    """
    阿里云百炼 Embeddings（基于 openai SDK 直连）

    使用 DashScope 的 OpenAI 兼容端点，调用 text-embedding-v2 模型。
    已验证：1536 维向量，单条和批量均可正常工作。

    为什么不用 langchain_openai.OpenAIEmbeddings？
    — 因为 langchain 封装的请求格式与百炼兼容端点不完全兼容
      （langchain 用内部 chunk 逻辑重组 input，导致 400 错误）。
      直接用 openai SDK 的 client.embeddings.create() 没有问题。
    """

    def __init__(self, api_key=None, model="text-embedding-v2", **kwargs):
        from openai import OpenAI

        # Embedding 仍使用 DashScope（DeepSeek 无公开 Embedding API）
        api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise ValueError("DASHSCOPE_API_KEY 必须在环境变量中设置（Embedding 使用 DashScope）")

        print(f"DEBUG [Embedding]: API Key = {api_key[:10]}... | base_url = https://dashscope.aliyuncs.com/compatible-mode/v1 | model = {model}")

        self.model = model
        self._client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

    def embed_documents(self, texts):
        """
        批量将文本转换为向量

        参数：
            texts: 文本列表，如 ["文本1", "文本2", ...]

        返回：
            向量列表，每个向量是浮点数列表（1536 维）
        """
        if not texts:
            return []

        embeddings = []
        # 逐条调用（百炼兼容端点对批量支持有限，逐条最稳定）
        for text in texts:
            if not isinstance(text, str) or not text.strip():
                # 空文本给零向量
                embeddings.append([0.0] * 1536)
                continue
            try:
                resp = self._client.embeddings.create(
                    model=self.model,
                    input=text,
                )
                embeddings.append(resp.data[0].embedding)
            except Exception as e:
                raise RuntimeError(f"Embedding API 调用失败: {e}")

        return embeddings

    def embed_query(self, text):
        """
        将单个查询文本转换为向量

        参数：
            text: 查询文本

        返回：
            向量（浮点数列表，1536 维）
        """
        if not text or not text.strip():
            return [0.0] * 1536

        resp = self._client.embeddings.create(
            model=self.model,
            input=text,
        )
        return resp.data[0].embedding


# =============================================================================
# 第二部分：数据加载函数（只负责 jobs.json）
# =============================================================================

def _parse_hard_requirements(requirements):
    """
    从任职要求中解析硬性指标
    
    参数：
        requirements: 任职要求列表（字符串列表）
    
    返回：
        dict: {
            'skills_text': 用于向量化的硬性指标文本,
            'core_skills': 核心技能列表,
            'experience_years': 经验要求（年数，整数）,
            'degree': 学历要求
        }
    """
    core_skills = []
    experience_years = 0
    degree = ""
    
    for req in requirements:
        req = req.strip()
        
        # 提取技能（通常是名词或技术栈名称）
        # 匹配常见的技术关键词模式
        skill_patterns = [
            r'(Python|Java|Go|C\+\+|JavaScript|TypeScript|React|Vue|Node\.js?|Django|Flask|Spring)',
            r'(MySQL|PostgreSQL|Redis|MongoDB|SQLite|Oracle)',
            r'(Git|Docker|Kubernetes|Jenkins|CI/CD)',
            r'(Linux|Unix|Windows)',
            r'(RESTful|gRPC|微服务|分布式)',
            r'(机器学习|深度学习|AI|NLP|计算机视觉)',
            r'(HTML|CSS|前端|后端|全栈)',
            r'(算法|数据结构|设计模式)',
            r'(TCP/IP|HTTP|网络协议)',
            r'(并发|线程|异步)',
        ]
        
        for pattern in skill_patterns:
            matches = re.findall(pattern, req)
            core_skills.extend(matches)
        
        # 提取经验要求
        exp_match = re.search(r'(\d+)\s*年\s*(经验|工作经验|开发经验)?', req)
        if exp_match:
            exp_years = int(exp_match.group(1))
            if exp_years > experience_years:
                experience_years = exp_years
        
        # 提取学历要求
        degree_keywords = ['本科', '硕士', '博士', '大专', '专科', '高中', '中专']
        for keyword in degree_keywords:
            if keyword in req:
                degree = keyword
                break
    
    # 去重技能列表
    core_skills = list(set(core_skills))
    
    # 构造技能文本（用于向量化）
    skills_text = f"职位: 技术岗位 | 核心技能: {', '.join(core_skills) if core_skills else '无'} | 经验要求: {experience_years}年 | 学历要求: {degree if degree else '不限'}"
    
    return {
        'skills_text': skills_text,
        'core_skills': core_skills,
        'experience_years': experience_years,
        'degree': degree
    }


def load_jobs(file_path):
    """
    从 JSON 文件加载职位数据
    
    参数：
        file_path: JSON 文件路径
    
    返回：
        职位列表，每个职位是一个字典
    
    数据格式示例：
        {
            "jobs": [
                {
                    "id": "1",
                    "title": "Python后端开发工程师",
                    "company": "创新科技有限公司",
                    "location": "北京",
                    "responsibilities": ["负责后端开发", "设计数据库"],
                    "requirements": ["3年经验", "熟悉Python"]
                }
            ]
        }
    """
    jobs = []
    
    try:
        # 打开文件并读取内容
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 检查数据格式是否正确
        if 'jobs' not in data:
            print(f"警告：{file_path} 中未找到 'jobs' 字段")
            return jobs
        
        # 遍历每个职位数据
        for idx, job_data in enumerate(data['jobs'], 1):
            try:
                # 提取职位信息，提供默认值防止数据缺失
                responsibilities = job_data.get('responsibilities', [])
                requirements = job_data.get('requirements', [])
                
                # 解析硬性指标
                hard_requirements = _parse_hard_requirements(requirements)
                
                job = {
                    'id': str(job_data.get('id', idx)),           # 职位ID
                    'title': job_data.get('title', '未命名职位'),   # 职位名称
                    'company': job_data.get('company', '未知公司'), # 公司名称
                    'location': job_data.get('location', '未知地点'),# 工作地点
                    'responsibilities': '\n'.join(responsibilities), # 岗位职责
                    'requirements': '\n'.join(requirements),       # 任职要求
                    'skills_text': hard_requirements['skills_text'], # 硬性指标文本（用于向量化）
                    'core_skills': hard_requirements['core_skills'], # 核心技能列表
                    'experience_years': hard_requirements['experience_years'], # 经验要求（年）
                    'degree': hard_requirements['degree']          # 学历要求
                }
                jobs.append(job)
            except Exception as e:
                # 如果某条职位数据解析失败，跳过并提示警告
                print(f"警告：职位 {job_data.get('title', f'第{idx}条')} 解析失败，已跳过: {str(e)}")
        
    except FileNotFoundError:
        print(f"错误：未找到文件 {file_path}")
    except json.JSONDecodeError:
        print(f"错误：{file_path} 不是有效的JSON格式")
    except Exception as e:
        print(f"错误：加载 {file_path} 时发生异常: {str(e)}")
    
    return jobs


# =============================================================================
# 第三部分：向量数据库相关函数
# =============================================================================

def init_vector_db(jobs_file, persist_directory="./chroma_db", force_reset=False):
    """
    初始化向量数据库
    
    将职位描述转换为向量存储到 Chroma 数据库中，方便后续快速检索
    
    参数：
        jobs_file: 职位数据文件路径
        persist_directory: 数据库存储目录（默认当前目录下的 chroma_db 文件夹）
        force_reset: 是否强制重新创建数据库
                      - True: 删除旧数据库，重新创建
                      - False: 使用已存在的数据库
    
    返回：
        tuple: (vector_db, embeddings) - 向量数据库对象和 Embeddings 对象
    
    注意事项：
        每个职位会存储两份：
        1. 完整描述（包含所有信息）
        2. 纯技能要求（只包含职位名和技能要求）
        这样可以提高匹配的准确性
    """
    # 创建阿里云百炼 Embeddings 对象
    embeddings = AliyunEmbeddings()
    
    # 检查数据库目录是否存在
    directory_exists = os.path.exists(persist_directory)
    
    # 情况1：目录存在且不需要重置，直接使用现有数据库
    if directory_exists and not force_reset:
        print(f"使用已存在的数据库目录 {persist_directory}")
        vector_db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
        return vector_db, embeddings
    
    # 情况2：需要创建新数据库
    if not directory_exists or force_reset:
        # 如果需要重置且目录存在，先删除旧目录
        if directory_exists and force_reset:
            print(f"检测到已存在的数据库目录 {persist_directory}，正在重置...")
            try:
                shutil.rmtree(persist_directory)
                print("数据库目录已删除")
            except Exception as e:
                print(f"警告：删除数据库目录失败: {str(e)}")
                # 如果删除失败，使用新的目录名
                persist_directory = persist_directory + "_new"
                print(f"使用新数据库目录: {persist_directory}")
        
        # 加载职位数据
        jobs = load_jobs(jobs_file)
        if not jobs:
            raise ValueError("未能加载任何职位数据，请检查数据源文件")
        
        print(f"正在创建新的向量数据库，共 {len(jobs)} 个职位...")
        documents = []
        
        # 为每个职位创建两份文档：完整描述 + 技能描述
        for job in jobs:
            # 1. 完整职位描述（包含所有信息）
            job_text = f"""职位：{job['title']}
公司：{job['company']}
工作地点：{job['location']}
岗位职责：
{job['responsibilities']}
任职要求：
{job['requirements']}"""
            documents.append(Document(page_content=job_text, metadata={
                'job_id': job['id'],
                'title': job['title'],
                'company': job['company'],
                'location': job['location'],
                'source': 'full_description'  # 标记来源为完整描述
            }))
            
            # 2. 纯硬性指标（用于硬性匹配）
            # 格式：职位: [职位名] | 核心技能: [技能1, 技能2] | 经验要求: [X年]
            skills_text = f"职位: {job['title']} | 核心技能: {', '.join(job['core_skills']) if job['core_skills'] else '无'} | 经验要求: {job['experience_years']}年 | 学历要求: {job['degree'] if job['degree'] else '不限'}"
            documents.append(Document(page_content=skills_text, metadata={
                'job_id': job['id'],
                'title': job['title'],
                'company': job['company'],
                'location': job['location'],
                'source': 'skills_only'  # 标记来源为纯技能
            }))
        
        # 创建向量数据库
        vector_db = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=persist_directory
        )
        print("向量数据库创建完成")
        
        return vector_db, embeddings
    
    raise RuntimeError("无法初始化向量数据库")


# =============================================================================
# 第四部分：辅助函数
# =============================================================================

def format_job_text(job):
    """
    将职位字典格式化为可读文本
    
    参数：
        job: 职位字典
    
    返回：
        str: 格式化的职位文本
    """
    return f"""职位：{job['title']}
公司：{job['company']}
工作地点：{job['location']}
岗位职责：
{job['responsibilities']}
任职要求：
{job['requirements']}"""


def format_resume_for_matching(resume):
    """
    将简历数据格式化为用于匹配的文本
    
    参数：
        resume: 简历字典（来自 resume_parser.py 的输出）
    
    返回：
        str: 格式化的简历文本
    """
    # 构建技能文本
    skills = resume.get('skills', [])
    if isinstance(skills, list):
        skills_text = ', '.join(skills)
    else:
        skills_text = str(skills)
    
    # 构建经历文本
    experience = resume.get('experience', [])
    experience_text = ""
    if isinstance(experience, list):
        for exp in experience:
            if isinstance(exp, dict):
                exp_str = f"{exp.get('company', '')} {exp.get('position', '')} ({exp.get('duration', '')})"
                desc = exp.get('description', '')
                if desc:
                    exp_str += f": {desc}"
                experience_text += exp_str + "\n"
            else:
                experience_text += str(exp) + "\n"
    else:
        experience_text = str(experience)
    
    return f"""姓名：{resume.get('name', '')}
技能：{skills_text}
工作经历：
{experience_text}"""
