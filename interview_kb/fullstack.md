# 全栈开发工程师 — 面试题库

## 一、前端基础

### Q1: React 的虚拟 DOM 原理？Diff 算法如何工作？
**参考答案**：
虚拟 DOM 是真实 DOM 的 JS 对象映射。当状态变化时：
1. 创建新的虚拟 DOM 树
2. 与旧的虚拟 DOM 树进行 Diff 比较（O(n) 启发式算法）
3. 将差异批量更新到真实 DOM

React Diff 三条策略：
- tree diff：跨层级移动很少，只对同层级节点比较
- component diff：同类型组件继续 diff，不同类型直接替换
- element diff：通过 key 标识节点的增删移

### Q2: 前端性能优化你做过哪些？
**参考答案**：
- 代码分割（React.lazy + Suspense，路由级/组件级懒加载）
- 图片优化（WebP 格式、懒加载、CDN、响应式图片）
- 缓存策略（Service Worker、HTTP 缓存、LocalStorage）
- 渲染优化（useMemo/useCallback、虚拟列表、防抖节流）
- Bundle 优化（Tree Shaking、Code Splitting、压缩）
- 网络优化（HTTP/2、资源预加载 prefetch/preload）

### Q3: CSS 布局方案对比：Flexbox vs Grid vs 传统布局？
**参考答案**：
- Flexbox：一维布局（行或列），适合组件内部布局
- Grid：二维布局（行和列同时），适合页面级布局
- 传统布局（float/position）：兼容性好但代码复杂
- 实际项目中通常结合使用

## 二、前后端交互

### Q4: JWT Token 的认证流程？如何防止 Token 被盗用？
**参考答案**：
流程：用户登录 → 服务端生成 JWT（Header.Payload.Signature）→ 客户端存储 → 每次请求携带
安全措施：
- Access Token 短期有效（15-30分钟）+ Refresh Token 长期有效
- HTTPS 传输，防止中间人攻击
- HttpOnly Cookie 存储（防 XSS）
- 黑名单机制（Redis 存储已注销 Token）
- 不要将敏感信息放在 Payload 中（Payload 只是 Base64 编码，非加密）

### Q5: 跨域问题怎么解决？CORS 的原理是什么？
**参考答案**：
浏览器同源策略限制跨域请求。CORS（跨域资源共享）是服务端方案：
- 简单请求：浏览器自动加 Origin 头，服务端返回 Access-Control-Allow-Origin
- 预检请求（OPTIONS）：非简单请求先发 OPTIONS 询问，服务端返回允许的方法和头
- Cookie 跨域：需设置 `withCredentials` + `Access-Control-Allow-Credentials: true`

## 三、后端与数据库

### Q6: Node.js 的事件循环机制？
**参考答案**：
Node.js 基于 libuv 实现事件循环，6 个阶段：
1. timers（setTimeout/setInterval 回调）
2. pending callbacks（系统操作回调）
3. idle/prepare（内部使用）
4. poll（获取新的 I/O 事件）
5. check（setImmediate 回调）
6. close callbacks（关闭回调）

微任务（process.nextTick、Promise）在每个阶段之间优先执行。

### Q7: 数据库事务的 ACID 特性？隔离级别？
**参考答案**：
- Atomicity（原子性）：事务要么全执行，要么全不执行
- Consistency（一致性）：事务前后数据完整性约束不变
- Isolation（隔离性）：并发事务互不干扰
- Durability（持久性）：已提交事务永久保存

隔离级别（从低到高）：READ UNCOMMITTED → READ COMMITTED → REPEATABLE READ → SERIALIZABLE
MySQL 默认 REPEATABLE READ（通过 MVCC + Gap Lock 防止幻读）

## 四、DevOps 与部署

### Q8: Docker 的核心概念？Dockerfile 的优化技巧？
**参考答案**：
核心概念：镜像（Image）、容器（Container）、Dockerfile（构建脚本）、Docker Compose（多容器编排）
Dockerfile 优化：
- 选择轻量基础镜像（alpine）
- 合并 RUN 命令减少层数
- 利用构建缓存（先 COPY 依赖文件，再 RUN 安装）
- 多阶段构建（build stage + runtime stage）
- .dockerignore 排除无用文件

### Q9: CI/CD 流水线应该包含哪些步骤？
**参考答案**：
- Lint 检查（代码风格）
- 单元测试 + 覆盖率检查
- 安全扫描（依赖漏洞检查）
- 构建（编译/打包）
- 集成测试
- 部署到 staging 环境
- 冒烟测试
- 部署到 production（灰度/蓝绿/滚动发布）

## 五、综合能力

### Q10: 从 0 到 1 设计一个全栈项目，你会怎么规划？
**参考答案**：
1. 需求分析 → 技术选型 → 数据库设计（ER 图）
2. 项目结构设计（monorepo / 前后端分离）
3. 先搭建 CI/CD 和开发环境（Docker Compose）
4. API 设计先行（OpenAPI/Swagger 文档 → 前后端并行开发）
5. 核心功能 MVP → 迭代完善 → 性能优化
6. 监控和日志（前端 Sentry + 后端 ELK/Prometheus）
