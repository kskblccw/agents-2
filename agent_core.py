#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智能体模块 (agent_core.py)
=====================
定义简历与职位匹配智能体及相关工具函数

包含：
1. MatcherAgent - 简历与职位匹配智能体
2. evaluate_answer_tool - 回答评分工具
3. search_knowledge_base - 知识库检索工具

"""
from dotenv import load_dotenv
load_dotenv()
import time  # 用于添加延迟，让对话更自然
import re    # 用于正则表达式解析
import json  # 用于JSON格式解析
import os     # 用于环境变量
from typing import Dict, Any, List
from vector_db import (
    load_jobs,           # 加载职位数据
    init_vector_db,      # 初始化向量数据库
    format_job_text,     # 格式化职位文本
    format_resume_for_matching  # 格式化简历文本用于匹配
)
from resume_parser import ResumeParser  # 简历解析器

# RAG 知识库（延迟导入，避免循环依赖和初始化顺序问题）
_rag_chain = None


def _get_rag():
    """延迟获取 RAGChain 实例"""
    global _rag_chain
    if _rag_chain is None:
        try:
            from knowledge_base import get_rag_chain
            _rag_chain = get_rag_chain()
        except Exception as e:
            print(f"[RAG] Knowledge base init skipped: {e}")
            _rag_chain = False  # 标记为不可用
    return _rag_chain if _rag_chain is not False else None


# =============================================================================
# 第一部分：LLM 调用函数
# =============================================================================

def ask_llm(prompt: str, model: str = "deepseek-v4-pro") -> str:
    """
    调用 DeepSeek LLM API（OpenAI 兼容模式）

    参数：
        prompt: 提示词
        model: 模型名称

    返回：
        str: API 返回的回答
    """
    from openai import OpenAI

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("未设置 DEEPSEEK_API_KEY 环境变量")

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com/v1",
        timeout=120.0,
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
        )
        return response.choices[0].message.content
    except Exception as e:
        raise RuntimeError(f"LLM API 调用失败: {str(e)}")


# 保留旧名称作为别名，兼容现有调用
def ask_glm(prompt: str, model: str = "deepseek-v4-pro") -> str:
    """兼容旧接口，内部调用 ask_llm"""
    return ask_llm(prompt, model=model)


# =============================================================================
# 第二部分：MatcherAgent - 简历与职位匹配智能体
# =============================================================================

class MatcherAgent:
    """
    简历与职位匹配智能体

    职责：
    1. 加载和初始化向量数据库
    2. 计算简历与职位的匹配度
    3. 生成专业的推荐理由

    工作流程：
    输入简历 → 向量检索 → 计算相似度 → LLM评价 → 输出结果
    """

    def __init__(self):
        """
        初始化匹配器
        """
        self._vector_db = None  # 向量数据库
        self._jobs_data = None  # 职位数据
        self._initialized = False  # 是否已初始化

    def initialize(self, jobs_file='jobs.json', force_reset=True):
        """
        初始化智能体

        参数：
            jobs_file: 职位数据文件路径
            force_reset: 是否强制重建数据库

        示例：
            >>> agent = MatcherAgent()
            >>> agent.initialize('jobs.json')
            正在加载职位数据...
            已加载 3 个职位
        """
        if self._initialized:
            print("数据库已初始化，跳过重复初始化")
            return

        print("正在加载职位数据...")
        self._jobs_data = load_jobs(jobs_file)
        print(f"已加载 {len(self._jobs_data)} 个职位")

        print("\n正在初始化向量数据库...")
        self._vector_db, _ = init_vector_db(jobs_file, force_reset=force_reset)
        print("向量数据库初始化完成！")

        self._initialized = True

    def _calculate_similarity_score(self, distance, max_distance=30.0, baseline=0.0):
        """
        计算相似度分数

        参数：
            distance: 向量距离（越小越相似）
            max_distance: 最大可能距离
            baseline: 基线值（低于此值的相似度归零）

        返回：
            float: 0-100 的匹配分数
        """
        # 将距离转换为相似度
        similarity = max(0, min(1, (max_distance - distance) / max_distance))

        # 应用基线
        if similarity < baseline:
            return 0.0

        # 转换为百分比分数
        score = ((similarity - baseline) / (1.0 - baseline)) * 100
        return max(0, min(100, round(score, 2)))

    def match_resume(self, resume, top_k=3):
        """
        匹配单个简历与所有职位

        参数：
            resume: 简历字典
            top_k: 返回前几名匹配结果（默认前3）

        返回：
            list: 匹配结果列表，按匹配度降序排列

        示例：
            >>> agent.match_resume(resume)
            [
                {
                    'match_score': 85.5,
                    'job_title': 'Python后端开发工程师',
                    'company': '创新科技有限公司',
                    'reason': '技能匹配度高...'
                }
            ]
        """
        if not self._initialized:
            raise RuntimeError("MatcherAgent 尚未初始化，请先调用 initialize()")

        # 格式化简历文本用于匹配
        resume_text = format_resume_for_matching(resume)

        # 在向量数据库中搜索相似职位
        # k=top_k * 2 因为每个职位存了两份（完整描述+技能描述）
        results = self._vector_db.similarity_search_with_score(
            resume_text,
            k=top_k * 2
        )

        if not results:
            return []

        # 按职位ID分组
        job_scores = {}
        for doc, distance in results:
            # 安全获取元数据，使用.get()防止KeyError
            job_id = doc.metadata.get('job_id', 'unknown')

            if job_id not in job_scores:
                job_scores[job_id] = {
                    'job_id': job_id,
                    'job_title': doc.metadata.get('title', '未知职位'),
                    'company': doc.metadata.get('company', '未知公司'),
                    'location': doc.metadata.get('location', '未知地点'),
                    'job_content': doc.page_content,
                    'sources': [],
                    'distances': []
                }

            job_scores[job_id]['sources'].append(doc.metadata.get('source', 'full_description'))
            job_scores[job_id]['distances'].append(distance)

        # 合并并计算分数
        merged_results = []
        for job_id, data in job_scores.items():
            # 计算平均距离
            avg_distance = sum(data['distances']) / len(data['distances'])

            # 计算匹配分数
            match_score = self._calculate_similarity_score(avg_distance)

            # 找到对应的职位详细信息（安全比较，处理空格问题）
            job_info = None
            job_id_str = str(job_id).strip()  # 去除首尾空格

            for j in self._jobs_data:
                # 安全获取ID并去除空格
                j_id = str(j.get('id', '')).strip()
                if j_id == job_id_str:
                    job_info = j
                    break

            if job_info:
                # 生成推荐理由（传入匹配分数）
                reason_data = self._generate_reason(resume, job_info, match_score)

                # 将 reason 对象转换为字符串格式（适配前端期望）
                reason_str = f"【匹配度】{reason_data.get('match_level', '一般')}\n"
                reason_str += f"【技能匹配】{reason_data.get('skill_match', '')}\n"
                reason_str += f"【经验匹配】{reason_data.get('experience_match', '')}\n"
                reason_str += f"【综合评价】{reason_data.get('overall_evaluation', '')}\n"
                reason_str += f"【建议】{reason_data.get('suggestion', '')}"

                merged_results.append({
                    'id': job_id,  # 前端期望的字段名
                    'title': data['job_title'],  # 前端期望的字段名
                    'company': data['company'],
                    'location': data['location'],
                    'match_score': match_score,
                    'reason': reason_str,  # 转换为字符串
                    'responsibilities': job_info.get('responsibilities', ''),
                    'requirements': job_info.get('requirements', ''),
                    'job_info': job_info  # 保留完整职位信息供面试使用
                })

        # 按匹配分数降序排序
        merged_results.sort(key=lambda x: x['match_score'], reverse=True)

        return merged_results[:top_k]

    def _generate_reason(self, resume, job, match_score=None):
        """
        使用 LLM 生成推荐理由（返回结构化 JSON）

        参数：
            resume: 简历字典
            job: 职位字典
            match_score: 匹配分数（可选）

        返回：
            dict: 结构化的推荐理由字典
        """
        resume_text = format_resume_for_matching(resume)
        job_text = format_job_text(job)

        prompt = f"""请分析以下简历与职位的匹配度，并返回结构化的分析报告：

【简历】
{resume_text}

【职位】
{job_text}

【匹配分数】{match_score if match_score is not None else '未知'}

请返回JSON格式，包含以下字段：
{{
    "skill_match": "技能匹配分析（1-3句话）",
    "experience_match": "经验匹配分析（1-3句话）",
    "overall_evaluation": "综合评价（1-2句话）",
    "suggestion": "面试建议（1-2句话）",
    "match_level": "匹配等级：优秀/良好/一般/较差"
}}

注意：只输出JSON，不要有其他文字。
"""

        try:
            response = ask_glm(prompt)
            # 尝试解析JSON
            result = self._parse_json_response(response)
            if result:
                return result
            # 如果JSON解析失败，返回简单文本格式
            return {
                'skill_match': response[:100] + '...' if len(response) > 100 else response,
                'experience_match': '',
                'overall_evaluation': '',
                'suggestion': '',
                'match_level': '一般'
            }
        except Exception as e:
            print(f"警告：LLM调用失败: {str(e)}")
            return {
                'skill_match': '技能匹配分析中',
                'experience_match': '经验匹配分析中',
                'overall_evaluation': '基于简历与职位的分析',
                'suggestion': '建议深入了解职位需求',
                'match_level': '一般'
            }

    def _parse_json_response(self, response):
        """
        解析LLM返回的JSON响应

        参数：
            response: LLM返回的字符串

        返回：
            dict: 解析后的字典，如果解析失败返回None
        """
        try:
            # 尝试提取JSON部分（可能在markdown代码块中）
            if '```json' in response:
                json_str = response.split('```json')[1].split('```')[0].strip()
            elif '```' in response:
                json_str = response.split('```')[1].split('```')[0].strip()
            else:
                json_str = response.strip()

            # 尝试解析JSON
            return json.loads(json_str)
        except json.JSONDecodeError:
            # JSON解析失败，尝试提取引号内的内容作为备选
            try:
                # 提取所有双引号内的内容
                quoted_strings = re.findall(r'"([^"]+)"', response)
                if len(quoted_strings) >= 3:
                    return {
                        'skill_match': quoted_strings[0] if len(quoted_strings) > 0 else '',
                        'experience_match': quoted_strings[1] if len(quoted_strings) > 1 else '',
                        'overall_evaluation': quoted_strings[2] if len(quoted_strings) > 2 else '',
                        'suggestion': quoted_strings[3] if len(quoted_strings) > 3 else '',
                        'match_level': quoted_strings[4] if len(quoted_strings) > 4 else '一般'
                    }
            except Exception as e:
                print(f"警告：JSON解析失败，备选方案也失败: {str(e)}")
            return None
        except Exception as e:
            print(f"警告：解析响应失败: {str(e)}")
            return None

    def match_all_resumes(self, resumes_file):
        """
        批量匹配所有简历

        参数：
            resumes_file: 简历文件路径（支持 JSON、PDF、DOCX 格式）

        返回：
            list: 所有简历的匹配结果
        """
        if not self._initialized:
            raise RuntimeError("MatcherAgent 尚未初始化，请先调用 initialize()")

        # 根据文件扩展名选择加载方式
        ext = os.path.splitext(resumes_file)[1].lower()

        if ext in ['.pdf', '.docx', '.doc']:
            # 使用 resume_parser 解析文件
            try:
                parser = ResumeParser()
                parsed = parser.parse_file(resumes_file)
                resumes = [parsed]
            except Exception as e:
                print(f"警告：解析简历文件失败: {str(e)}")
                return []
        elif ext == '.json':
            # 从 JSON 文件加载
            try:
                with open(resumes_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                resumes = data.get('resumes', [])
            except Exception as e:
                print(f"警告：加载简历 JSON 文件失败: {str(e)}")
                return []
        else:
            print(f"警告：不支持的文件格式: {ext}")
            return []

        if not resumes:
            print("警告：未加载到任何简历数据")
            return []

        results = []
        total = len(resumes)

        print(f"\n正在匹配 {total} 份简历...")

        for idx, resume in enumerate(resumes, 1):
            try:
                print(f"\r匹配进度: {idx}/{total}", end='', flush=True)

                match_results = self.match_resume(resume)

                results.append({
                    'resume': resume,
                    'matches': match_results
                })

                # 添加小延迟，避免请求过快
                time.sleep(0.5)

            except Exception as e:
                print(f"\n警告：处理简历 {resume.get('name', f'简历{idx}')} 时发生错误: {str(e)}")

        print("\n匹配完成！")
        return results


# =============================================================================
# 第三部分：工具函数（LangChain Tool）
# =============================================================================

# 尝试导入 LangChain tools（如果未安装则设为 None）
try:
    from langchain_core.tools import tool
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    def tool(func):
        """如果 LangChain 未安装，返回原始函数"""
        return func
    print("⚠️  提示：未安装 langchain-core，将使用基础函数模式。如需完整 LangChain 支持，请运行：pip install langchain-core")


@tool
def evaluate_answer_tool(question: str, answer: str) -> str:
    """
    评价候选人对面试问题的回答。
    """
    prompt = f"""
你是一个严格的技术面试官。请评价候选人的回答。

面试问题：{question}
候选人的回答：{answer}

【评分标准】
- 1-3 分：回答完全错误、答非所问、或说"不知道/不清楚/不太会"
- 4-6 分：回答部分正确，但缺少关键细节、没有具体案例
- 7-8 分：回答基本正确，有核心要点，但深度不够
- 9-10 分：回答完整、有具体案例、有技术细节

输出格式（必须是 JSON）：
{{"score": 整数 1-10, "comment": "一句话评价"}}
"""
    try:
        response = ask_glm(prompt)
        # 解析 JSON
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            return json.dumps(result, ensure_ascii=False)
        return json.dumps({"score": 5, "comment": "评价解析失败"})
    except Exception as e:
        return json.dumps({"score": 5, "comment": f"评价出错：{str(e)}"})


@tool
def search_knowledge_base(query: str) -> str:
    """
    搜索知识库，可以查询任何信息，包括：面试题、参考答案、评分标准、追问策略、候选人个人信息（如姓名、生日、学校）等。

    当需要以下信息时调用此工具：
    - 生成更有针对性的面试问题
    - 评估候选人回答时，需要参考答案或评分标准
    - 判断是否需要追问时，需要追问策略参考
    - 需要了解某个技术栈的常见面试考点

    参数：
        query: 搜索查询文本，可以是技术关键词或问题描述。例如：
               - "Python 后端 面试问题"
               - "FastAPI 依赖注入 评分标准"
               - "行为面试 STAR 法则"
               - "如何评价技术问题回答"

    返回：
        str: 格式化的相关知识内容，包含参考问题和答案
    """
    try:
        from knowledge_base import get_rag_chain
        rag = get_rag_chain()

        # 使用通用检索：从所有知识库中检索
        docs_with_scores = rag.kb.retrieve_with_scores(
            query=query,
            top_k=5,
        )

        if not docs_with_scores:
            return "[知识库] 未找到相关知识。请基于你的专业知识继续回答。"

        # 组装结果
        parts = ["[知识库检索结果]\n"]
        for i, (doc, score) in enumerate(docs_with_scores, 1):
            source = doc.metadata.get('source_file', '未知')
            section = doc.metadata.get('section', '')
            parts.append(f"--- 参考 {i}（来源：{source}）---")
            parts.append(doc.page_content.strip())
            parts.append("")

        parts.append("请基于以上知识库内容辅助你的判断和回答。")
        return "\n".join(parts)

    except Exception as e:
        return f"[知识库] 检索失败: {str(e)}。请基于你的专业知识继续。"
