# ai_camera

`ai_camera` 是一个 Python/FastAPI 图片姿势分析后端。当前 MVP 聚焦一条主链路：

```text
图片路径 -> 姿势分析 -> 背景分析 -> 可选 MySQL 入库
```

当前能力：

- 读取一张本地图片路径。
- 使用 MediaPipe Pose 分析人体姿势骨架。
- 使用 CLIP zero-shot 做背景/场景分类。CLIP 能根据图像和文本 prompt 的相似度识别开放背景场景，比手写颜色规则更准确。
- 根据 `should_store` 控制是否写入 MySQL。
- MySQL 当前只存三类数据：图片、姿势、姿势骨架关键点。
- 背景分析当前只返回，不落库。
- 姿势库 MVP 仍读取 `app/pose_library/poses.json`。

## Install

所有 Python 依赖都安装到项目 `venv` 中：

```bash
venv/bin/pip install -r requirement.txt
```

如果新增或升级依赖，必须同步更新：

- `requirement.txt`
- `pyproject.toml`

## Run

启动 FastAPI 服务：

```bash
venv/bin/uvicorn app.main:app --reload
```

健康检查：

```bash
curl http://127.0.0.1:8000/health
```

## Core API

主接口：

```text
POST /analyze_image_path
```

示例：

```bash
curl -X POST "http://127.0.0.1:8000/analyze_image_path" \
  -H "Content-Type: application/json" \
  -d '{"image_url":"images/example.jpg","should_store":true}'
```

请求字段：

- `image_url`: 图片存储路径。当前实现按本地文件路径读取。
- `should_store`: 是否写入 MySQL。
- `source`: 图片来源，默认 `upload`。
- `detector`: 姿势检测器名称，默认 `local_pose_estimator`。

接口无论是否入库，都会在图片分析成功时返回姿势分析和背景分析结果。入库失败时，分析结果仍会返回，入库错误写在 `storage.error`。

## Database

配置 MySQL：

```bash
export DATABASE_URL='mysql+pymysql://user:password@localhost:3306/pose_app?charset=utf8mb4'
```

初始化表：

```bash
venv/bin/python -m app.main init-db
```

当前 MVP 使用 `Base.metadata.create_all(engine)` 建表，暂未引入 Alembic。

## CLI

分析图片并入库：

```bash
venv/bin/python -m app.main analyze --image-url images/example.jpg
```

读取 JSON 姿势库：

```bash
venv/bin/python -m app.main list-library-poses
```

把参考图片写入 JSON 姿势库：

```bash
venv/bin/python scripts/ingest_pose_image.py \
  --pose-id street_side_hair_touch_002 \
  --image images/example.jpg \
  --poses-json app/pose_library/poses.json \
  --overwrite
```

单独测试 CLIP 场景分类：

```bash
venv/bin/python scripts/test_scene_classifier.py images/example.jpg
```

`scripts/` 是手动工具目录，不参与 HTTP 运行时主链路；保留用于姿势库维护和 CLIP 调试。`tests/` 是回归测试目录，当前仍然有效，不能删除。

## Test

运行全量测试：

```bash
venv/bin/python -m pytest
```

常用定向测试：

```bash
venv/bin/python -m pytest tests/test_analyze_image.py
venv/bin/python -m pytest tests/test_database_services.py
venv/bin/python -m pytest tests/test_pose_library_updater.py
venv/bin/python -m pytest tests/test_skeleton_features.py
```

## Documentation

- `AGENTS.md`: AI coding agent 必须遵守的协作和架构规则。
- `CLAUDE.md`: Claude Code 专用说明。当前任务要求不修改它。
- `docs/AI_HANDOFF.md`: coding agent 交接说明。
- `docs/ARCHITECTURE.md`: 长期稳定的项目分层和文件职责。
- `docs/SCHEMAS.md`: API、数据库、JSON 数据结构说明。
- `docs/ROADMAP.md`: 当前阶段、下一步、长期方向。
- `docs/DECISIONS.md`: 技术决策记录。
