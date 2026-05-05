# Decisions

本文档记录重要技术决策。格式保持简短，便于后续回看为什么选择当前方案。

## 2026-05-05: 使用 MediaPipe Pose 做本地姿势识别

Decision:

- 使用 `mediapipe==0.10.9` 的 Pose 模型做人体姿势关键点识别。

Reason:

- MediaPipe Pose 能直接输出 33 个 normalized landmarks。
- 适合本地 MVP，不需要额外部署推理服务。
- 当前业务重点是先验证姿势分析和入库链路。

Rejected / Deferred:

- 暂不使用云端姿势识别 API，避免早期依赖外部服务。
- 暂不训练自定义姿势模型。

## 2026-05-05: 使用 CLIP zero-shot 做背景分类

Decision:

- 使用 `openai/clip-vit-base-patch32` 做场景 zero-shot 分类。
- 使用描述性 prompt，而不是单词 label。
- 只保留 CLIP 分类，不再保留 heuristic RGB 背景分类。

Reason:

- CLIP 能同时理解图像内容和文本 prompt，适合识别街道、咖啡馆、海边、公园、室内等开放背景。
- 与手写颜色/亮度规则相比，CLIP 对背景语义的识别更准确，也更容易通过 prompt 扩展类别。
- 描述性 prompt 对 CLIP 更稳定。
- 不需要先准备标注数据。

Rejected / Deferred:

- 暂不训练专用 scene classifier。
- 暂不把背景分类结果写入数据库。
- 删除 heuristic RGB fallback，因为它只能基于颜色和亮度做粗略判断，容易误判背景语义。

## 2026-05-05: 使用 MySQL + SQLAlchemy 2.x + PyMySQL

Decision:

- 数据库存储使用 MySQL。
- ORM 使用 SQLAlchemy 2.x。
- MySQL driver 使用 PyMySQL。
- 连接从 `DATABASE_URL` 环境变量读取。

Reason:

- MySQL 是维护者当前确认的存储方案。
- SQLAlchemy 能让模型、repository、service 分层更清晰。
- `DATABASE_URL` 方便本地、测试、部署环境切换。

Rejected / Deferred:

- 暂不引入 Alembic。MVP 阶段先用 `Base.metadata.create_all(engine)`。
- 暂不把姿势库 `poses.json` 迁移到 MySQL。

## 2026-05-05: HTTP 主链路收敛到 `/analyze_image_path`

Decision:

- 当前唯一 HTTP 主业务入口是 `POST /analyze_image_path`。
- 入参使用图片路径和 `should_store`。
- 接口无论是否入库，都返回姿势分析和背景分析。

Reason:

- 避免同时存在“只分析”“只姿势入库”“姿势+背景分析”多条业务链路。
- 维护者可以更容易追踪调用链。
- 当前产品图片已经有存储路径，先不处理 multipart 上传。

Rejected / Deferred:

- 暂不保留 multipart `/analyze_image` 作为主入口。
- 暂不支持远程 URL 下载。

## 2026-05-05: 背景分析暂不落库

Decision:

- 背景分析在 API 响应中返回，但不写 MySQL。

Reason:

- 维护者目前只确认三张表：图片、姿势、姿势骨架关键点。
- 背景字段如何存储仍未完全确认，先避免过早固化 schema。

Rejected / Deferred:

- 暂不新增 `scene_analysis` 表。
- 暂不在 `app_images` 添加背景字段。
