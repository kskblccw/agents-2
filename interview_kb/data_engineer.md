# 数据工程师 — 面试题库

## 一、数据处理与 ETL

### Q1: 大数据处理中，MapReduce 的 Shuffle 阶段做了什么？
**参考答案**：
Shuffle 是 MapReduce 中最关键的阶段，发生在 Map 和 Reduce 之间：
1. Map 端：将 Map 输出按 key 分区（Partition）→ 内存缓冲 → 溢写磁盘 → 归并排序
2. Reduce 端：从各 Map 节点拉取对应分区的数据 → 归并排序 → 传给 Reduce 函数
3. 优化：Combiner（Map 端预聚合，减少网络传输）、调整 partition 数量、压缩中间数据

### Q2: Spark 的宽依赖和窄依赖的区别？
**参考答案**：
- 窄依赖：父 RDD 的每个分区最多被一个子 RDD 分区使用（map、filter、union）
  - 支持 pipeline 执行，不需要 shuffle
- 宽依赖：父 RDD 的分区被多个子 RDD 分区使用（groupByKey、reduceByKey、join）
  - 需要 shuffle，会触发 Stage 划分
  - 优化：用 reduceByKey 代替 groupByKey（Map 端预聚合）

### Q3: ETL 流程设计中如何处理数据质量问题？
**参考答案**：
- 数据完整性：非空校验、主键唯一性
- 数据一致性：格式校验（日期、电话号码正则）
- 数据准确性：范围校验（年龄 0-150）、关联校验（省市区对应）
- 数据时效性：数据延迟监控、SLA 告警
- 脏数据处理策略：丢弃 / 默认值填充 / 回溯修复 / 写入错误表

## 二、SQL 深入

### Q4: SQL 窗口函数有哪些？ROW_NUMBER vs RANK vs DENSE_RANK 的区别？
**参考答案**：
常见窗口函数：
- 排名：ROW_NUMBER / RANK / DENSE_RANK / NTILE
- 聚合：SUM / AVG / COUNT / MAX / MIN + OVER
- 偏移：LAG / LEAD / FIRST_VALUE / LAST_VALUE

ROW_NUMBER：1,2,3,4,5（唯一序号，并列也连续）
RANK：1,1,3,4,5（并列跳号）
DENSE_RANK：1,1,2,3,4（并列不跳号）

### Q5: SQL 查询优化思路？
**参考答案**：
1. EXPLAIN 分析执行计划 → 关注 type 列（ALL=全表扫描，ref=索引查找）
2. 索引优化：覆盖索引、最左前缀原则、避免索引失效（函数/计算/类型转换）
3. 避免 SELECT *，只取需要的列
4. 大表 JOIN 优化：小表驱动大表、避免笛卡尔积
5. 子查询改写为 JOIN（相关子查询性能差）
6. 分库分表：水平拆分（按 ID 哈希）、垂直拆分（按业务）

## 三、数据仓库

### Q6: 星型模型和雪花模型的区别？维度建模的核心思想？
**参考答案**：
- 星型模型：事实表 + 一层维度表（维度表有冗余，查询快）
- 雪花模型：事实表 + 多层维度表（维度表规范化，节省存储，查询需多 JOIN）
- 维度建模：先确定业务过程 → 声明粒度 → 设计维度 → 确定事实
- 缓慢变化维（SCD）处理：Type1 覆盖、Type2 新增行、Type3 新增列

### Q7: 数据湖 vs 数据仓库 vs 湖仓一体？
**参考答案**：
- 数据仓库：结构化数据、Schema-on-Write、适合 BI 报表（如 Snowflake、Redshift）
- 数据湖：所有格式原始数据、Schema-on-Read、适合数据科学和 ML（如 S3 + Spark）
- 湖仓一体（Lakehouse）：Data Lake 的存储 + Warehouse 的事务和 SQL 能力（如 Databricks Delta Lake、Apache Iceberg、Apache Hudi）

## 四、Python 数据处理

### Q8: Pandas 中 apply vs transform vs agg 的区别？
**参考答案**：
- apply：对 DataFrame 行/列应用任意函数，返回结果形状可变
- transform：返回与原数据相同形状的结果（适合分组后填充/标准化）
- agg（aggregate）：聚合操作，返回标量（适合分组后统计）

大数据场景优先用向量化操作（pandas 内置方法），避免逐行 apply。

### Q9: 如何优化 Python 数据处理性能？
**参考答案**：
- 向量化操作代替循环（pandas/numpy 内置方法）
- 使用合适的数据类型（category 代替 object，int8/16/32 代替 int64）
- 分块读取大文件（pd.read_csv(chunksize=10000)）
- 并行处理（multiprocessing / dask / pyspark）
- 避免不必要的数据拷贝（inplace=True、视图 vs 拷贝）
