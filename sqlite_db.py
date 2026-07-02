#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SQLite 数据库模块 (sqlite_db.py)
===============================
使用 SQLite 存储简历信息、面试记录和反馈报告

包含类：
1. InterviewDatabase - 数据库操作类

作者：AI助手
"""

import sqlite3
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Any

class InterviewDatabase:
    """
    SQLite 数据库操作类
    
    提供以下功能：
    1. 存储上传的简历信息
    2. 存储每一轮面试的问答记录
    3. 存储最终的评分和反馈报告
    """
    
    def __init__(self, db_path: str = "interview.db"):
        """
        初始化数据库连接
        
        参数：
            db_path: 数据库文件路径，默认为 interview.db
        """
        self.db_path = db_path
        self.conn = None
        self._connect()
        self._create_tables()
        self.conn.row_factory = sqlite3.Row

    def _connect(self):
        """建立数据库连接"""
        try:
            self.conn = sqlite3.connect(self.db_path)
            self.conn.row_factory = sqlite3.Row  # 使查询结果可以通过列名访问
        except sqlite3.Error as e:
            raise RuntimeError(f"数据库连接失败: {str(e)}")
    
    def _create_tables(self):
        """创建必要的数据表"""
        cursor = self.conn.cursor()
        
        # 创建简历表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS resumes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                phone TEXT,
                email TEXT,
                location TEXT,
                skills TEXT,
                experience TEXT,
                education TEXT,
                source_file TEXT,
                file_type TEXT,
                parsed_data TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # 创建面试记录表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS interviews (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                resume_id INTEGER NOT NULL,
                job_title TEXT NOT NULL,
                company TEXT,
                match_score INTEGER,
                status TEXT DEFAULT 'in_progress',
                started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                ended_at TIMESTAMP,
                FOREIGN KEY (resume_id) REFERENCES resumes(id)
            )
        ''')
        
        # 创建问答记录表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS qa_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                interview_id INTEGER NOT NULL,
                question_index INTEGER NOT NULL,
                question TEXT NOT NULL,
                answer TEXT,
                evaluation TEXT,
                score INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (interview_id) REFERENCES interviews(id)
            )
        ''')
        
        # 创建反馈报告表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                interview_id INTEGER NOT NULL,
                overall_score INTEGER,
                summary TEXT,
                strengths TEXT,
                weaknesses TEXT,
                suggestions TEXT,
                report_json TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (interview_id) REFERENCES interviews(id)
            )
        ''')
        
        self.conn.commit()
    
    def close(self):
        """关闭数据库连接"""
        if self.conn:
            self.conn.close()
    
    # ==================== 简历操作 ====================
    
    def insert_resume(self, resume_data: Dict[str, Any]) -> int:
        """
        插入简历信息
        
        参数：
            resume_data: 简历数据字典
        
        返回：
            int: 插入的简历ID
        """
        cursor = self.conn.cursor()
        
        # 将列表类型转换为 JSON 字符串存储
        skills = json.dumps(resume_data.get('skills', []))
        experience = json.dumps(resume_data.get('experience', []))
        education = json.dumps(resume_data.get('education', []))
        parsed_data = json.dumps(resume_data)
        
        cursor.execute('''
            INSERT 
            INTO resumes (
                name, phone, email, location, skills,
                experience, education, source_file, file_type, parsed_data
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            resume_data.get('name', ''),
            resume_data.get('phone', ''),
            resume_data.get('email', ''),
            resume_data.get('location', ''),
            skills,
            experience,
            education,
            resume_data.get('meta', {}).get('source_file', ''),
            resume_data.get('meta', {}).get('file_type', ''),
            parsed_data
        ))
        
        self.conn.commit()
        return cursor.lastrowid
    
    def get_resume(self, resume_id: int) -> Optional[Dict[str, Any]]:
        """
        获取简历信息
        
        参数：
            resume_id: 简历ID
        
        返回：
            dict: 简历数据，不存在返回 None
        """
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM resumes WHERE id = ?', (resume_id,))
        row = cursor.fetchone()
        
        if row:
            return self._row_to_dict(row)
        return None
    
    def get_all_resumes(self) -> List[Dict[str, Any]]:
        """
        获取所有简历列表
        
        返回：
            list: 简历列表
        """
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM resumes ORDER BY created_at DESC')
        rows = cursor.fetchall()
        
        return [self._row_to_dict(row) for row in rows]
    
    def get_resume_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """
        根据姓名查询简历（模糊匹配）
        
        参数：
            name: 姓名
        
        返回：
            dict: 简历数据，不存在返回 None
        """
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM resumes WHERE name LIKE ? ORDER BY created_at DESC', (f'%{name}%',))
        row = cursor.fetchone()
        
        if row:
            return self._row_to_dict(row)
        return None
    
    def update_resume(self, resume_id: int, resume_data: Dict[str, Any]) -> bool:
        """
        更新简历信息
        
        参数：
            resume_id: 简历ID
            resume_data: 更新的简历数据
        
        返回：
            bool: 是否更新成功
        """
        cursor = self.conn.cursor()
        
        updates = []
        params = []
        
        if 'name' in resume_data:
            updates.append('name = ?')
            params.append(resume_data['name'])
        if 'phone' in resume_data:
            updates.append('phone = ?')
            params.append(resume_data['phone'])
        if 'email' in resume_data:
            updates.append('email = ?')
            params.append(resume_data['email'])
        if 'location' in resume_data:
            updates.append('location = ?')
            params.append(resume_data['location'])
        if 'skills' in resume_data:
            updates.append('skills = ?')
            params.append(json.dumps(resume_data['skills']))
        if 'experience' in resume_data:
            updates.append('experience = ?')
            params.append(json.dumps(resume_data['experience']))
        if 'education' in resume_data:
            updates.append('education = ?')
            params.append(json.dumps(resume_data['education']))
        
        if not updates:
            return False
        
        params.append(resume_id)
        query = f'UPDATE resumes SET {", ".join(updates)} WHERE id = ?'
        
        cursor.execute(query, params)
        self.conn.commit()
        
        return cursor.rowcount > 0
    
    def delete_resume(self, resume_id: int) -> bool:
        """
        删除简历（级联删除相关的面试记录）
        
        参数：
            resume_id: 简历ID
        
        返回：
            bool: 是否删除成功
        """
        cursor = self.conn.cursor()
        
        # 先删除相关的问答记录、报告、面试记录
        cursor.execute('SELECT id FROM interviews WHERE resume_id = ?', (resume_id,))
        interview_ids = [row['id'] for row in cursor.fetchall()]
        
        for interview_id in interview_ids:
            cursor.execute('DELETE FROM qa_records WHERE interview_id = ?', (interview_id,))
            cursor.execute('DELETE FROM reports WHERE interview_id = ?', (interview_id,))
            cursor.execute('DELETE FROM interviews WHERE id = ?', (interview_id,))
        
        cursor.execute('DELETE FROM resumes WHERE id = ?', (resume_id,))
        self.conn.commit()
        
        return cursor.rowcount > 0
    
    # ==================== 面试操作 ====================
    
    def create_interview(self, resume_id: int, job_title: str, company: str = "", 
                        match_score: int = 0) -> int:
        """
        创建面试记录
        
        参数：
            resume_id: 简历ID
            job_title: 职位名称
            company: 公司名称
            match_score: 匹配分数
        
        返回：
            int: 面试ID
        """
        cursor = self.conn.cursor()
        
        cursor.execute('''
            INSERT INTO interviews (resume_id, job_title, company, match_score)
            VALUES (?, ?, ?, ?)
        ''', (resume_id, job_title, company, match_score))
        
        self.conn.commit()
        return cursor.lastrowid
    
    def get_interview(self, interview_id: int) -> Optional[Dict[str, Any]]:
        """
        获取面试记录
        
        参数：
            interview_id: 面试ID
        
        返回：
            dict: 面试数据，不存在返回 None
        """
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM interviews WHERE id = ?', (interview_id,))
        row = cursor.fetchone()
        
        if row:
            return self._row_to_dict(row)
        return None
    
    def get_interviews_by_resume(self, resume_id: int) -> List[Dict[str, Any]]:
        """
        获取指定简历的所有面试记录
        
        参数：
            resume_id: 简历ID
        
        返回：
            list: 面试记录列表
        """
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM interviews WHERE resume_id = ? ORDER BY started_at DESC', (resume_id,))
        rows = cursor.fetchall()
        
        return [self._row_to_dict(row) for row in rows]
    
    def update_interview_status(self, interview_id: int, status: str) -> bool:
        """
        更新面试状态
        
        参数：
            interview_id: 面试ID
            status: 状态（in_progress, completed, cancelled）
        
        返回：
            bool: 是否更新成功
        """
        cursor = self.conn.cursor()
        
        if status == 'completed':
            cursor.execute('''
                UPDATE interviews 
                SET status = ?, ended_at = CURRENT_TIMESTAMP 
                WHERE id = ?
            ''', (status, interview_id))
        else:
            cursor.execute('UPDATE interviews SET status = ? WHERE id = ?', (status, interview_id))
        
        self.conn.commit()
        return cursor.rowcount > 0
    
    # ==================== 问答记录操作 ====================
    
    def insert_qa_record(self, interview_id: int, question_index: int, 
                        question: str, answer: str = "", 
                        evaluation: str = "", score: int = 0) -> int:
        """
        插入问答记录
        
        参数：
            interview_id: 面试ID
            question_index: 问题序号
            question: 问题内容
            answer: 回答内容
            evaluation: 评价内容
            score: 评分
        
        返回：
            int: 记录ID
        """
        cursor = self.conn.cursor()
        
        cursor.execute('''
            INSERT INTO qa_records (interview_id, question_index, question, answer, evaluation, score)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (interview_id, question_index, question, answer, evaluation, score))
        
        self.conn.commit()
        return cursor.lastrowid
    
    def get_qa_records(self, interview_id: int) -> List[Dict[str, Any]]:
        """
        获取面试的所有问答记录
        
        参数：
            interview_id: 面试ID
        
        返回：
            list: 问答记录列表（按问题序号排序）
        """
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM qa_records WHERE interview_id = ? ORDER BY question_index', (interview_id,))
        rows = cursor.fetchall()
        
        return [self._row_to_dict(row) for row in rows]
    
    def update_qa_record(self, record_id: int, **kwargs) -> bool:
        """
        更新问答记录
        
        参数：
            record_id: 记录ID
            kwargs: 要更新的字段（answer, evaluation, score）
        
        返回：
            bool: 是否更新成功
        """
        cursor = self.conn.cursor()
        
        updates = []
        params = []
        
        if 'answer' in kwargs:
            updates.append('answer = ?')
            params.append(kwargs['answer'])
        if 'evaluation' in kwargs:
            updates.append('evaluation = ?')
            params.append(kwargs['evaluation'])
        if 'score' in kwargs:
            updates.append('score = ?')
            params.append(kwargs['score'])
        
        if not updates:
            return False
        
        params.append(record_id)
        query = f'UPDATE qa_records SET {", ".join(updates)} WHERE id = ?'
        
        cursor.execute(query, params)
        self.conn.commit()
        
        return cursor.rowcount > 0
    
    # ==================== 面试记录操作 ====================
    
    def insert_interview(self, candidate_name: str, job_title: str, company: str,
                        status: str = 'completed', started_at: datetime = None,
                        resume_id: int = None) -> int:
        """
        插入面试主记录
        
        参数：
            candidate_name: 候选人姓名
            job_title: 目标职位
            company: 公司名称
            status: 面试状态（默认 'completed'）
            started_at: 开始时间（默认当前时间）
            resume_id: 简历ID（可选，默认None，为None时使用1）
        
        返回：
            int: 面试ID
        """
        cursor = self.conn.cursor()
        
        if started_at is None:
            started_at = datetime.now()
        
        # 如果 resume_id 为 None，使用默认值 1（兼容旧数据）
        if resume_id is None:
            resume_id = 1
        
        cursor.execute('''
            INSERT INTO interviews (resume_id, job_title, company, status, started_at)
            VALUES (?, ?, ?, ?, ?)
        ''', (resume_id, job_title, company, status, started_at))
        
        self.conn.commit()
        return cursor.lastrowid
    
    # ==================== 反馈报告操作 ====================
    
    def insert_report(self, interview_id: int, overall_score: int, 
                    summary: str, strengths: str, weaknesses: str, 
                    suggestions: str, report_json: str = "") -> int:
        """
        插入反馈报告
        
        参数：
            interview_id: 面试ID
            overall_score: 总体评分
            summary: 总结
            strengths: 优点
            weaknesses: 缺点
            suggestions: 建议
            report_json: 完整报告的JSON字符串
        
        返回：
            int: 报告ID
        """
        cursor = self.conn.cursor()
        
        cursor.execute('''
            INSERT INTO reports (interview_id, overall_score, summary, strengths, weaknesses, suggestions, report_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (interview_id, overall_score, summary, strengths, weaknesses, suggestions, report_json))
        
        self.conn.commit()
        return cursor.lastrowid
    
    def get_report(self, interview_id: int) -> Optional[Dict[str, Any]]:
        """
        获取面试的反馈报告
        
        参数：
            interview_id: 面试ID
        
        返回：
            dict: 报告数据，不存在返回 None
        """
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM reports WHERE interview_id = ?', (interview_id,))
        row = cursor.fetchone()
        
        if row:
            return self._row_to_dict(row)
        return None
    
    def get_reports_by_resume(self, resume_id: int) -> List[Dict[str, Any]]:
        """
        获取指定简历的所有反馈报告
        
        参数：
            resume_id: 简历ID
        
        返回：
            list: 报告列表
        """
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT r.* FROM reports r
            JOIN interviews i ON r.interview_id = i.id
            WHERE i.resume_id = ?
            ORDER BY r.created_at DESC
        ''', (resume_id,))
        rows = cursor.fetchall()
        
        return [self._row_to_dict(row) for row in rows]
    
    def get_all_reports(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        获取所有反馈报告（按时间倒序）
        
        参数：
            limit: 返回的最大数量，默认50
        
        返回：
            list: 报告列表
        """

        cursor = self.conn.cursor()
        cursor.execute('''
    SELECT
        r.id, r.overall_score, r.summary, r.strengths, r.weaknesses, r.suggestions,
        COALESCE(res.name, '未知候选人') as candidate_name,  -- 姓名在 resumes 表里，没有则用默认值
        i.job_title, i.company, i.started_at as interview_date
FROM reports r
JOIN interviews i ON r.interview_id = i.id  -- 关联 interviews 表 (复数)
LEFT JOIN resumes res ON i.resume_id = res.id    -- 使用 LEFT JOIN，即使没有简历记录也能返回
ORDER BY r.created_at DESC
LIMIT ?
''', (limit,))
        rows = cursor.fetchall()
        
        results = []
        for row in rows:
            report_dict = dict(row)
            results.append(report_dict)
        
        return results
    
    # ==================== 辅助方法 ====================
    
    def _row_to_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
        """
        将数据库行转换为字典
        
        参数：
            row: SQLite 行对象
        
        返回：
            dict: 字典形式的数据
        """
        result = dict(row)
        
        # 将 JSON 字符串转换回原始类型
        if 'skills' in result and result['skills']:
            try:
                result['skills'] = json.loads(result['skills'])
            except:
                pass
        if 'experience' in result and result['experience']:
            try:
                result['experience'] = json.loads(result['experience'])
            except:
                pass
        if 'education' in result and result['education']:
            try:
                result['education'] = json.loads(result['education'])
            except:
                pass
        if 'parsed_data' in result and result['parsed_data']:
            try:
                result['parsed_data'] = json.loads(result['parsed_data'])
            except:
                pass
        if 'report_json' in result and result['report_json']:
            try:
                result['report_json'] = json.loads(result['report_json'])
            except:
                pass
        
        # 将时间戳转换为可读格式（使用 strptime 兼容更多格式）
        def format_datetime(dt_str):
            if not dt_str:
                return dt_str
            # 尝试多种常见格式
            formats = ['%Y-%m-%d %H:%M:%S.%f', '%Y-%m-%d %H:%M:%S', '%Y-%m-%dT%H:%M:%S', '%Y-%m-%dT%H:%M:%S.%f']
            for fmt in formats:
                try:
                    return datetime.strptime(dt_str, fmt).strftime('%Y-%m-%d %H:%M:%S')
                except ValueError:
                    continue
            # 如果都失败，返回原始字符串
            return dt_str
        
        if 'created_at' in result:
            result['created_at'] = format_datetime(result['created_at'])
        if 'started_at' in result:
            result['started_at'] = format_datetime(result['started_at'])
        if 'ended_at' in result:
            result['ended_at'] = format_datetime(result['ended_at'])
        
        return result
    
    def get_statistics(self) -> Dict[str, int]:
        """
        获取数据库统计信息
        
        返回：
            dict: 统计数据
        """
        cursor = self.conn.cursor()
        
        cursor.execute('SELECT COUNT(*) as count FROM resumes')
        resume_count = cursor.fetchone()['count']
        
        cursor.execute('SELECT COUNT(*) as count FROM interviews')
        interview_count = cursor.fetchone()['count']
        
        cursor.execute('SELECT COUNT(*) as count FROM qa_records')
        qa_count = cursor.fetchone()['count']
        
        cursor.execute('SELECT COUNT(*) as count FROM reports')
        report_count = cursor.fetchone()['count']
        
        cursor.execute('SELECT COUNT(*) as count FROM interviews WHERE status = "completed"')
        completed_count = cursor.fetchone()['count']
        
        return {
            'total_resumes': resume_count,
            'total_interviews': interview_count,
            'total_qa_records': qa_count,
            'total_reports': report_count,
            'completed_interviews': completed_count
        }


# ==================== 示例用法 ====================
def demo():
    """
    演示数据库操作
    """
    # 创建数据库连接
    db = InterviewDatabase()
    
    try:
        # 插入简历
        resume_id = db.insert_resume({
            'name': '张三',
            'phone': '13800138000',
            'email': 'zhangsan@example.com',
            'location': '北京',
            'skills': ['Python', 'Java', 'SQL'],
            'experience': [{'company': 'XX公司', 'position': '工程师', 'duration': '2年'}],
            'education': [{'school': 'XX大学', 'degree': '本科', 'major': '计算机科学'}],
            'meta': {'source_file': 'zhangsan.pdf', 'file_type': '.pdf'}
        })
        print(f"插入简历成功，ID: {resume_id}")
        
        # 创建面试
        interview_id = db.create_interview(resume_id, 'Python工程师', 'XX科技公司', 85)
        print(f"创建面试成功，ID: {interview_id}")
        
        # 插入问答记录
        qa_id1 = db.insert_qa_record(interview_id, 1, '请介绍一下你自己', 
                                    '我是张三，有2年开发经验', '回答清晰，信息完整', 90)
        qa_id2 = db.insert_qa_record(interview_id, 2, '你为什么想加入我们公司？',
                                    '贵公司技术氛围好', '回答合理', 85)
        print(f"插入问答记录成功，ID: {qa_id1}, {qa_id2}")
        
        # 插入报告
        report_id = db.insert_report(interview_id, 88, 
                                    '整体表现优秀，技术能力扎实',
                                    '技术基础扎实，沟通能力强',
                                    '项目经验较少',
                                    '建议多参与大型项目',
                                    json.dumps({'score': 88, 'details': '...'}))
        print(f"插入报告成功，ID: {report_id}")
        
        # 更新面试状态
        db.update_interview_status(interview_id, 'completed')
        print("更新面试状态成功")
        
        # 查询简历
        resume = db.get_resume(resume_id)
        print(f"\n查询简历: {resume['name']}")
        
        # 查询问答记录
        qa_records = db.get_qa_records(interview_id)
        print(f"\n问答记录 ({len(qa_records)} 条):")
        for qa in qa_records:
            print(f"  Q{qa['question_index']}: {qa['question']}")
            print(f"  A: {qa['answer']}")
            print(f"  评价: {qa['evaluation']} (得分: {qa['score']})")
        
        # 查询统计信息
        stats = db.get_statistics()
        print(f"\n统计信息: {stats}")
        
    finally:
        db.close()


if __name__ == "__main__":
    demo()
