# Architecture Rules

当前项目的目标是让主业务路径清晰、可读、可手动修改：

```text
HTTP -> services -> vision -> repositories -> database
```

## Allowed

- `core`: 配置和数据库连接。
- `models`: SQLAlchemy ORM。
- `repositories`: 数据库读写，只处理 SQLAlchemy session 和 ORM 对象。
- `services`: 业务流程编排，例如图片路径分析、背景分析、可选入库。
- `vision`: 图片分析能力，例如 MediaPipe Pose 和 CLIP scene classification。
- `pose_library`: 姿势库 JSON 读取和维护。
- `features`: 从姿势关键点派生特征。
- `docs`: schema、架构和交接文档。

## Current Main Flow

```text
POST /analyze_image_path
  -> app.main.analyze_image_path_endpoint
  -> app.services.image_analysis_service.analyze_image_path
  -> app.vision.pose_estimator.PoseEstimator.estimate
  -> app.vision.scene_classifier.SceneClassifier.classify
  -> if should_store:
       app.repositories.image_repository.create_image
       app.repositories.pose_repository.create_pose
       app.repositories.pose_repository.create_keypoints
```

## Rules

- HTTP 层只做请求/响应转换和错误码映射。
- `services` 层负责业务流程，决定是否入库。
- `vision` 层不允许依赖数据库。
- `repositories` 层不允许调用模型推理。
- 背景分析本次只返回，不落库。
- MySQL 表结构暂时只确认三张表：图片、姿势、姿势骨架关键点。
- 不新增推荐 API、LLM 摄影建议、实时摄像头、pose card 渲染。

## Files and Responsibilities

### `app/main.py`

FastAPI 应用入口，同时提供少量 CLI 命令。

- `GET /health`: 健康检查。
- `POST /analyze_image_path`: 当前唯一 HTTP 主业务入口。
- `python -m app.main init-db`: 初始化 MySQL 表。
- `python -m app.main analyze --image-url ...`: 走同一个 `analyze_image_path` service，并强制入库。
- `python -m app.main list-library-poses`: 列出 JSON 姿势库。

### `app/schemas.py`

Pydantic schema 定义。

- API 请求：`AnalyzeImagePathRequest`。
- API 响应：`AnalyzeImagePathResponse`。
- 图片信息：`ImageInfo`。
- 姿势结果：`PoseEstimationResult`、`Landmark`。
- 背景分类：`SceneClassificationResult`、`SceneCandidate`。
- 入库结果：`ImageStorageResult`。

### `app/core/config.py`

读取环境变量配置。

- 当前只读取 `DATABASE_URL`。
- 未配置时抛出清晰错误。

### `app/core/database.py`

SQLAlchemy 基础设施。

- `Base`: ORM declarative base。
- `create_app_engine()`: 创建 engine。
- `create_session_factory()`: 创建 session factory。
- `session_scope()`: service 默认使用的 session context。
- `init_db()`: MVP 阶段用 `Base.metadata.create_all()` 初始化表。

### `app/models/`

SQLAlchemy ORM。

- `image.py`: `app_images` 表。
- `pose.py`: `poses` 表。
- `pose_keypoint.py`: `pose_keypoints` 表。

### `app/repositories/`

数据库读写函数。

- `image_repository.py`: 创建图片、更新分析状态、读取图片。
- `pose_repository.py`: 创建姿势、创建关键点、按图片读取姿势、读取单个姿势及关键点。

Repositories 不做模型推理，不处理 HTTP，不做业务编排。

### `app/services/`

业务流程层。

- `image_analysis_service.py`: 当前主业务流程。负责读取图片路径、调用姿势分析、调用背景分析、根据 `should_store` 可选写入三张表。
- `pose_library_service.py`: 读取和搜索 JSON 姿势库。

### `app/vision/`

图片分析能力。

- `image_loader.py`: multipart 上传图片读取工具。当前主 HTTP 入口不使用，但测试和未来上传入口可复用。
- `pose_estimator.py`: MediaPipe Pose 分析，输出 normalized landmarks。
- `scene_classifier.py`: CLIP zero-shot 背景分类。CLIP 通过图像和描述性文本 prompt 的相似度识别背景，适合当前开放场景分类需求。

### `app/pose_library/`

JSON 姿势库。

- `poses.json`: 当前姿势库数据源。
- `loader.py`: 读取和校验 JSON 姿势库。
- `updater.py`: 使用参考图片更新某个 pose 的 `landmark_template`、`scene_analysis`、`embedding_document`。

### `app/features/`

从姿势关键点派生特征。

- `skeleton_features.py`: 计算 bbox、人体中心、手臂抬起、手靠近脸、是否坐姿、是否全身等。

### `scripts/`

手动工具脚本，不参与 HTTP 运行时主链路，但仍然有维护作用。

- `ingest_pose_image.py`: 将参考图片写入 JSON 姿势库。
- `test_scene_classifier.py`: 单图调试 CLIP 背景分类。

### `tests/`

回归测试，不是运行时依赖，但对长期维护有效，不应删除。

- `test_analyze_image.py`: HTTP 主接口、图片读取、姿势估计基础行为。
- `test_database_services.py`: 图片分析、背景分析、可选入库 service。
- `test_pose_library_updater.py`: JSON 姿势库参考图 ingest。
- `test_skeleton_features.py`: 骨架派生特征。
