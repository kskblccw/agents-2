#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
面试知识库 RAG 模块 (knowledge_base.py)
======================================
基于 Chroma 向量数据库 + 智谱 Embedding 的知识检索增强生成（RAG）系统。

核心能力：
1. 文档加载与智能分块（Markdown / TXT）
2. 多 Collection 知识库管理（按技术方向隔离）
3. 语义检索 + 元数据过滤
4. RAGChain：检索 → 上下文组装 → LLM 增强生成

架构关系：
    interview_kb/*.md  ──加载──▶  KnowledgeBase  ──检索──▶  RAGChain  ──生成──▶  LLM
         (知识文档)        (向量存储+检索)        (上下文组装+生成)

使用示例：
    >>> kb = KnowledgeBase()
    >>> kb.load_directory("interview_kb/")
    >>> chain = RAGChain(kb)
    >>> context = chain.retrieve_for_question_generation("Python后端", resume, job)
    >>> result = chain.generate_questions(context, resume, job)
"""

import os
import re
import json
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

# LangChain 相关
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# 复用项目已有的阿里云百炼 Embeddings
from vector_db import AliyunEmbeddings


# =============================================================================
# 第一部分：KnowledgeBase — 知识库管理
# =============================================================================

class KnowledgeBase:
    """
    面试知识库管理器

    职责：
    1. 加载 Markdown/TXT 文档并智能分块
    2. 将文档块向量化存入 Chroma
    3. 支持多 Collection（按技术栈隔离）
    4. 提供语义检索 + 元数据过滤

    Collection 设计：
    ┌──────────────────────┬──────────────────────────────────┐
    │ Collection 名称       │ 存储内容                         │
    ├──────────────────────┼──────────────────────────────────┤
    │ python_backend       │ Python 后端面试题 + 参考答案       │
    │ fullstack            │ 全栈开发面试题 + 参考答案          │
    │ data_engineer        │ 数据工程面试题 + 参考答案          │
    │ behavioral           │ 行为面试题 + STAR 法则            │
    │ answer_guide         │ 评分标准 + 评价话术 + 追问策略     │
    └──────────────────────┴──────────────────────────────────┘
    """

    def __init__(self, persist_dir: str = "./chroma_db"):
        """
        初始化知识库管理器

        参数：
            persist_dir: Chroma 持久化目录（与 vector_db.py 共用）
        """
        self.persist_dir = persist_dir
        self.embeddings = AliyunEmbeddings()
        # 延迟导入 Chroma（避免循环导入）
        from langchain_community.vectorstores import Chroma
        self.Chroma = Chroma

        # 存储已加载的 collection 实例
        self._collections: Dict[str, Any] = {}

        # 文档分块器配置
        # chunk_size=500: 每个块约 500 字符（适配面试题粒度：一题 + 答案）
        # chunk_overlap=80: 保留 80 字符重叠，避免切断上下文
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=80,
            separators=["\n## ", "\n### ", "\n#### ", "\n", "。", ".", " ", ""],
            length_function=len,
        )

    # ---- 文档加载 ----

    def load_file(self, file_path: str) -> List[Document]:
        """
        加载单个 Markdown/TXT 文件并分块

        参数：
            file_path: 文件路径

        返回：
            List[Document]: 分块后的文档列表（每个块带元数据标记）
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")

        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()

        # 提取文件名作为 collection 名称
        file_name = os.path.splitext(os.path.basename(file_path))[0]

        # 提取文档中的二级标题，作为元数据标签
        # 例如 "## 一、Python 语言基础" → 标签 "Python 语言基础"
        sections = re.findall(r'^##\s+(.+?)$', text, re.MULTILINE)

        # 分块
        chunks = self.text_splitter.split_text(text)

        # 为每个块创建 Document 对象，附带元数据
        documents = []
        for i, chunk in enumerate(chunks):
            # 检测这个 chunk 属于哪个章节
            section_tag = self._detect_section(chunk, sections)

            doc = Document(
                page_content=chunk,
                metadata={
                    'source_file': file_name,       # 来源文件名
                    'chunk_index': i,                # 块序号
                    'section': section_tag,           # 所属章节
                    'created_at': datetime.now().isoformat(),
                }
            )
            documents.append(doc)

        return documents

    def load_directory(self, directory_path: str) -> Dict[str, int]:
        """
        加载目录下所有 Markdown 文件到知识库

        参数：
            directory_path: 知识库目录路径

        返回：
            Dict[str, int]: {collection_name: 文档块数量}

        示例：
            >>> kb = KnowledgeBase()
            >>> kb.load_directory("interview_kb/")
            {'python_backend': 25, 'fullstack': 20, ...}
        """
        if not os.path.isdir(directory_path):
            raise NotADirectoryError(f"目录不存在: {directory_path}")

        stats = {}

        for filename in os.listdir(directory_path):
            if filename.endswith(('.md', '.txt')):
                file_path = os.path.join(directory_path, filename)
                collection_name = os.path.splitext(filename)[0]

                try:
                    documents = self.load_file(file_path)
                    self._store_documents(collection_name, documents)
                    stats[collection_name] = len(documents)
                    print(f"  [OK] {filename} -> collection '{collection_name}' ({len(documents)} chunks)")
                except Exception as e:
                    print(f"  [ERR] {filename} load failed: {e}")

        return stats

    # ---- 存储 ----

    def _store_documents(self, collection_name: str, documents: List[Document], force_reset: bool = True):
        """
        将文档块存入 Chroma Collection

        参数：
            collection_name: Collection 名称
            documents: 文档块列表
            force_reset: 是否强制重建（True=每次清空重建，False=追加）
        """
        import shutil

        # 如果 collection 已加载到内存，跳过（避免重复初始化）
        if collection_name in self._collections:
            return

        # 每个 collection 使用独立的子目录
        collection_dir = os.path.join(self.persist_dir, f"kb_{collection_name}")

        # 如果强制重建，删除旧数据
        if force_reset and os.path.exists(collection_dir):
            try:
                shutil.rmtree(collection_dir)
            except PermissionError:
                # Windows 文件锁定：目录被其他进程占用
                print(f"  [WARN] Cannot reset {collection_name} (file locked), appending instead")
            except OSError:
                # 文件正在使用中
                pass

        # 创建/打开 Collection（如果目录已存在且有数据，Chroma 会自动加载）
        try:
            vectorstore = self.Chroma(
                collection_name=f"kb_{collection_name}",
                embedding_function=self.embeddings,
                persist_directory=collection_dir,
            )

            # 仅在新 collection 或空 collection 时添加文档
            existing_count = vectorstore._collection.count() if hasattr(vectorstore, '_collection') else 0
            if documents and existing_count == 0:
                vectorstore.add_documents(documents)

            self._collections[collection_name] = vectorstore
        except Exception as e:
            print(f"  [WARN] Failed to store {collection_name}: {e}")

    def get_collection(self, collection_name: str):
        """
        获取指定 Collection（如果未加载则从磁盘加载）

        参数：
            collection_name: Collection 名称

        返回：
            Chroma vectorstore 实例
        """
        if collection_name not in self._collections:
            collection_dir = os.path.join(self.persist_dir, f"kb_{collection_name}")
            if not os.path.exists(collection_dir):
                raise ValueError(
                    f"Collection '{collection_name}' 不存在。"
                    f"请先调用 load_directory() 加载知识库。"
                )
            self._collections[collection_name] = self.Chroma(
                collection_name=f"kb_{collection_name}",
                embedding_function=self.embeddings,
                persist_directory=collection_dir,
            )
        return self._collections[collection_name]

    # ---- 检索 ----

    def retrieve(
        self,
        query: str,
        collection_names: Optional[List[str]] = None,
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """
        语义检索：从知识库中检索与 query 最相关的文档块

        参数：
            query: 查询文本
            collection_names: 要检索的 Collection 列表（None=全部检索）
            top_k: 返回前 K 个结果
            filter_metadata: 元数据过滤条件（如 {'section': 'Python 语言基础'}）

        返回：
            List[Document]: 按相关度排序的文档块列表

        示例：
            >>> docs = kb.retrieve("Python 装饰器", collection_names=["python_backend"])
            >>> for doc in docs:
            ...     print(doc.metadata['section'], doc.page_content[:100])
        """
        if collection_names is None:
            collection_names = list(self._collections.keys())

        if not collection_names:
            raise ValueError("没有可检索的 Collection，请先调用 load_directory()")

        all_results = []

        for name in collection_names:
            try:
                vectorstore = self.get_collection(name)

                # 构建检索参数
                search_kwargs = {"k": top_k}
                if filter_metadata:
                    search_kwargs["filter"] = filter_metadata

                results = vectorstore.similarity_search(query, **search_kwargs)
                all_results.extend(results)

            except ValueError:
                continue  # Collection 不存在则跳过
            except Exception as e:
                print(f"警告：检索 Collection '{name}' 时出错: {e}")
                continue

        # 如果没有语义搜索结果，不进行排序
        if not all_results:
            return []

        # 尝试按相关度排序（如果检索结果包含距离信息）
        # 这里使用简单的去重 + 限制数量
        seen = set()
        unique_results = []
        for doc in all_results:
            key = doc.page_content[:100]  # 用前 100 字符去重
            if key not in seen:
                seen.add(key)
                unique_results.append(doc)

        return unique_results[:top_k]

    def retrieve_with_scores(
        self,
        query: str,
        collection_names: Optional[List[str]] = None,
        top_k: int = 5,
    ) -> List[Tuple[Document, float]]:
        """
        带相关度分数的检索

        参数：
            query: 查询文本
            collection_names: Collection 列表
            top_k: 返回前 K 个结果

        返回：
            List[Tuple[Document, float]]: (文档, 距离分数) 列表，分数越小越相关
        """
        if collection_names is None:
            collection_names = list(self._collections.keys())

        all_results = []

        for name in collection_names:
            try:
                vectorstore = self.get_collection(name)
                results = vectorstore.similarity_search_with_score(query, k=top_k)
                all_results.extend(results)
            except (ValueError, Exception):
                continue

        # 按分数排序（分数越小越相似）
        all_results.sort(key=lambda x: x[1])

        # 去重
        seen = set()
        unique_results = []
        for doc, score in all_results:
            key = doc.page_content[:100]
            if key not in seen:
                seen.add(key)
                unique_results.append((doc, score))

        return unique_results[:top_k]

    # ---- 辅助方法 ----

    def _detect_section(self, chunk: str, sections: List[str]) -> str:
        """
        检测文档块属于哪个章节

        参数：
            chunk: 文档块内容
            sections: 文档中所有二级标题列表

        返回：
            str: 章节名称（"未知章节" 如果无法确定）
        """
        for section in sections:
            if section in chunk:
                return section
        # 尝试匹配 chunk 中出现的第一个标题模式
        match = re.search(r'^#+\s+(.+?)$', chunk, re.MULTILINE)
        if match:
            return match.group(1)
        return "未知章节"

    def list_collections(self) -> List[str]:
        """列出所有已加载的 Collection 名称"""
        return list(self._collections.keys())

    def get_stats(self) -> Dict[str, Any]:
        """获取知识库统计信息"""
        stats = {}
        for name in self._collections:
            try:
                vs = self.get_collection(name)
                count = vs._collection.count() if hasattr(vs, '_collection') else '?'
                stats[name] = {'chunks': count}
            except Exception:
                stats[name] = {'chunks': 'error'}
        return stats


# =============================================================================
# 第二部分：RAGChain — 检索增强生成链
# =============================================================================

class RAGChain:
    """
    RAG 生成链：检索 → 上下文组装 → LLM 生成

    在面试场景中的三个核心应用：
    1. 面试题生成 —— 检索相关知识 → 生成更有针对性的问题
    2. 回答评估 —— 检索参考答案 + 评分标准 → 更准确的评分
    3. 追问决策 —— 检索相关深度问题 → 智能追问
    """

    def __init__(self, knowledge_base: KnowledgeBase):
        """
        初始化 RAG 链

        参数：
            knowledge_base: KnowledgeBase 实例
        """
        self.kb = knowledge_base

        # 延迟初始化 LLM 调用函数（避免循环导入）
        self._ask_glm = None

    def _get_llm(self):
        """延迟加载 LLM 调用函数"""
        if self._ask_glm is None:
            from agent_core import ask_glm
            self._ask_glm = ask_glm
        return self._ask_glm

    # ---- 应用 1：面试题生成增强 ----

    def retrieve_for_question_generation(
        self,
        job_title: str,
        resume_skills: List[str],
        top_k: int = 5,
    ) -> str:
        """
        检索与面试题生成相关的知识

        检索策略：
        1. 根据岗位名称匹配对应的技术知识库
        2. 根据简历技能补充相关技术考点
        3. 同时检索行为面试题通用库

        参数：
            job_title: 岗位名称（如 "Python后端开发工程师"）
            resume_skills: 简历中的技能列表
            top_k: 检索数量

        返回：
            str: 组装好的上下文文本（可直接注入 prompt）
        """
        # 确定要检索的 Collection
        collection_map = {
            'python': 'python_backend',
            '后端': 'python_backend',
            'backend': 'python_backend',
            '全栈': 'fullstack',
            'fullstack': 'fullstack',
            'full-stack': 'fullstack',
            '前端': 'fullstack',
            'frontend': 'fullstack',
            '数据': 'data_engineer',
            'data': 'data_engineer',
            '大数据': 'data_engineer',
            'etl': 'data_engineer',
        }

        tech_collections = set()
        for keyword, col_name in collection_map.items():
            if keyword.lower() in job_title.lower():
                tech_collections.add(col_name)

        if not tech_collections:
            # 默认检索所有技术库
            tech_collections = {'python_backend', 'fullstack', 'data_engineer'}

        # 始终包含行为面试和评分指南
        all_collections = list(tech_collections) + ['behavioral']

        # 构建检索查询：组合岗位 + 技能
        skills_str = ' '.join(resume_skills[:8]) if resume_skills else ''
        query = f"{job_title} {skills_str} 面试问题 技术考点"

        # 多路检索
        docs_with_scores = self.kb.retrieve_with_scores(
            query=query,
            collection_names=all_collections,
            top_k=top_k,
        )

        return self._assemble_context(docs_with_scores, "面试题生成参考")

    # ---- 应用 2：回答评估增强 ----

    def retrieve_for_answer_evaluation(
        self,
        question: str,
        answer: str,
        job_title: str = "",
        top_k: int = 4,
    ) -> str:
        """
        检索与回答评估相关的知识

        检索策略：
        1. 根据问题内容检索相关参考答案
        2. 检索评分标准和评价话术
        3. 检索常见的错误回答模式

        参数：
            question: 面试问题
            answer: 候选人回答
            job_title: 岗位名称（可选，用于缩小检索范围）

        返回：
            str: 组装好的评估参考上下文
        """
        # 从问题中提取关键词作为查询
        # 去除常见的面试引导词，提取核心技术术语
        clean_question = re.sub(r'请(描述|介绍|说明|谈谈|解释|回答)', '', question)
        clean_question = clean_question.strip()

        # 确定检索范围
        collections = ['answer_guide']  # 始终包含评分标准

        # 根据岗位添加技术库
        tech_keywords = {
            'python': 'python_backend', 'django': 'python_backend', 'flask': 'python_backend',
            'fastapi': 'python_backend', 'sql': 'python_backend',
            'react': 'fullstack', 'vue': 'fullstack', 'javascript': 'fullstack',
            'node': 'fullstack', 'css': 'fullstack', '前端': 'fullstack',
            'spark': 'data_engineer', 'hadoop': 'data_engineer', 'sql': 'data_engineer',
            '数据': 'data_engineer', 'etl': 'data_engineer', 'pandas': 'data_engineer',
        }
        for kw, col in tech_keywords.items():
            if kw.lower() in question.lower() or kw.lower() in job_title.lower():
                if col not in collections:
                    collections.append(col)

        # 两步检索：先检索参考答案，再检索评分标准
        ref_docs = self.kb.retrieve_with_scores(
            query=clean_question,
            collection_names=[c for c in collections if c != 'answer_guide'],
            top_k=top_k,
        )

        guide_docs = self.kb.retrieve_with_scores(
            query=f"{clean_question} 评分 评价",
            collection_names=['answer_guide'],
            top_k=2,
        )

        all_docs = ref_docs + guide_docs

        return self._assemble_context(all_docs, "回答评估参考")

    # ---- 应用 3：智能追问增强 ----

    def retrieve_for_follow_up(
        self,
        question: str,
        answer: str,
        evaluation: str,
        top_k: int = 3,
    ) -> str:
        """
        检索与追问决策相关的知识

        参数：
            question: 原问题
            answer: 候选人回答
            evaluation: 初步评价

        返回：
            str: 追问决策参考上下文
        """
        # 结合评价内容判断追问方向
        query = f"{question} {evaluation} 深入追问"

        docs_with_scores = self.kb.retrieve_with_scores(
            query=query,
            collection_names=['answer_guide', 'behavioral'],
            top_k=top_k,
        )

        return self._assemble_context(docs_with_scores, "追问策略参考")

    # ---- 通用方法 ----

    def generate_with_context(
        self,
        context: str,
        prompt_template: str,
        **kwargs,
    ) -> str:
        """
        带 RAG 上下文的 LLM 生成

        参数：
            context: 检索到的上下文文本
            prompt_template: 提示词模板（使用 {context} 占位符）
            **kwargs: 模板中的其他变量

        返回：
            str: LLM 生成的文本
        """
        ask_glm = self._get_llm()

        # 填充模板
        filled_prompt = prompt_template.format(context=context, **kwargs)

        try:
            response = ask_glm(filled_prompt)
            return response
        except Exception as e:
            # 降级：去掉 RAG 上下文重试
            print(f"警告：RAG 生成失败（{e}），降级为纯 LLM 生成")
            fallback_prompt = prompt_template.replace("{context}\n\n", "").format(**kwargs)
            return ask_glm(fallback_prompt)

    def _assemble_context(
        self,
        docs_with_scores: List[Tuple[Document, float]],
        label: str = "参考知识",
    ) -> str:
        """
        将检索结果组装为 LLM 可用的上下文字符串

        参数：
            docs_with_scores: (Document, score) 列表
            label: 上下文标签

        返回：
            str: 格式化的上下文字符串
        """
        if not docs_with_scores:
            return ""

        parts = [f"## {label}（从知识库检索）\n"]

        for i, (doc, score) in enumerate(docs_with_scores, 1):
            source = doc.metadata.get('source_file', '未知来源')
            section = doc.metadata.get('section', '')
            section_str = f" → {section}" if section and section != '未知章节' else ""

            parts.append(f"### 参考 {i}（来源：{source}{section_str}）")
            parts.append(doc.page_content.strip())
            parts.append("")

        parts.append("---")
        parts.append("请基于以上参考知识，结合候选人的具体情况，给出专业的分析和输出。")
        parts.append("")

        return "\n".join(parts)


# =============================================================================
# 第三部分：便捷函数
# =============================================================================

# 全局单例（延迟初始化）
_kb_instance: Optional[KnowledgeBase] = None
_rag_instance: Optional[RAGChain] = None


def get_knowledge_base(persist_dir: str = "./chroma_db", kb_dir: str = "interview_kb/") -> KnowledgeBase:
    """
    获取全局 KnowledgeBase 单例

    首次调用时自动加载知识库目录

    参数：
        persist_dir: Chroma 持久化目录
        kb_dir: 知识库 Markdown 文件目录

    返回：
        KnowledgeBase 实例
    """
    global _kb_instance
    if _kb_instance is None:
        _kb_instance = KnowledgeBase(persist_dir=persist_dir)
        if os.path.isdir(kb_dir):
            print(f"[KB] Loading knowledge base: {kb_dir}")
            try:
                stats = _kb_instance.load_directory(kb_dir)
                total = sum(stats.values())
                print(f"[KB] Knowledge base loaded: {len(stats)} collections, {total} chunks")
            except Exception as e:
                print(f"[KB] Load failed: {e}")
        else:
            print(f"[WARN] KB directory not found: {kb_dir}")
    return _kb_instance


def get_rag_chain() -> RAGChain:
    """
    获取全局 RAGChain 单例

    返回：
        RAGChain 实例
    """
    global _rag_instance
    if _rag_instance is None:
        kb = get_knowledge_base()
        _rag_instance = RAGChain(kb)
    return _rag_instance


def reset_knowledge_base():
    """重置知识库（用于测试或重建）"""
    global _kb_instance, _rag_instance
    _kb_instance = None
    _rag_instance = None


# =============================================================================
# 第四部分：CLI 测试入口
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("[TEST] Interview Knowledge Base RAG System - Self Check")
    print("=" * 60)

    # Load .env first
    from dotenv import load_dotenv
    load_dotenv()

    # 1. Initialize KB
    kb = KnowledgeBase()
    print("\n[1] Loading knowledge base documents...")
    stats = kb.load_directory("interview_kb/")
    print(f"    Total: {sum(stats.values())} chunks")

    # 2. Test retrieval
    print("\n[2] Testing semantic retrieval...")
    chain = RAGChain(kb)

    # Test 1: Retrieve Python-related questions
    results = kb.retrieve(
        query="Python decorator GIL multithreading",
        collection_names=["python_backend"],
        top_k=3,
    )
    print(f"    Query 'Python decorator GIL' -> {len(results)} results")
    for i, doc in enumerate(results, 1):
        source = doc.metadata.get('source_file', '?')
        section = doc.metadata.get('section', '?')
        print(f"   {i}. [{source}] {section}: {doc.page_content[:80]}...")

    # Test 2: Retrieve scoring standards
    results2 = kb.retrieve(
        query="how to evaluate answer quality",
        collection_names=["answer_guide"],
        top_k=2,
    )
    print(f"\n    Query 'evaluate answer quality' -> {len(results2)} results")
    for i, doc in enumerate(results2, 1):
        print(f"   {i}. {doc.page_content[:80]}...")

    # 3. Test RAGChain context assembly
    print("\n[3] Testing RAGChain context assembly...")
    context = chain.retrieve_for_question_generation(
        job_title="Python后端开发工程师",
        resume_skills=["Python", "FastAPI", "MySQL", "Docker"],
        top_k=3,
    )
    print(f"    Generated context length: {len(context)} chars")
    print(f"    Context preview:\n{context[:500]}...")

    print("\n" + "=" * 60)
    print("[OK] Knowledge Base RAG System self-check complete")
    print("=" * 60)
