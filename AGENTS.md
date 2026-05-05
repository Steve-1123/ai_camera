# Agent Rules

本文件是所有 AI coding agent 修改本项目时必须遵守的规则。开始编码前先读本文件，再按任务需要阅读 `README.md` 和 `docs/` 下的文档。

## Core Rule

当前 MVP 只有一条主业务链路：

```text
HTTP -> image path analysis -> pose analysis + scene analysis -> optional DB storage
```

主入口是：

- `POST /analyze_image_path`
- `app.services.image_analysis_service.analyze_image_path`

改代码时优先维护这条链路的清晰度。不要重新引入多条相互竞争的“分析入口”。

## Architecture Boundaries

- `core`: 配置和数据库连接。只负责 `DATABASE_URL`、SQLAlchemy engine/session、`init_db()`。
- `models`: SQLAlchemy ORM。只描述数据库表和关系。
- `repositories`: 数据库读写。只接收 SQLAlchemy `Session` 和 ORM 对象，不调用模型推理。
- `services`: 业务流程。负责串联图片读取、姿势分析、背景分析、可选入库。
- `vision`: 图片分析能力。负责 MediaPipe Pose、CLIP/heuristic scene classification、图片读取/转换，不依赖数据库。
- `pose_library`: JSON 姿势库读取和维护。MVP 阶段继续读 `app/pose_library/poses.json`。
- `features`: 从姿势关键点派生特征。
- `docs`: 记录架构、schema、路线图、决策和 agent 交接。

## Do Not

- 不要把数据库写入放进 `vision/`。
- 不要让 `repositories/` 调用 MediaPipe、CLIP 或业务流程。
- 不要新增推荐 API、LLM 摄影建议、实时摄像头、pose card 渲染，除非用户明确要求。
- 不要把背景/场景分析写入 MySQL。当前产品只确认三张表：图片、姿势、姿势骨架关键点。
- 不要提交 secrets、tokens、`.env`、`venv/`、图片样本、模型缓存、`prompt.txt`。
- 不要修改 `CLAUDE.md`，除非用户明确要求。
- 不要大规模重写项目结构。优先小步、可读、可测试的改动。

## Required Before Coding

1. 阅读 `prompt.txt`，确认本次任务目标。
2. 查看 `git status --short --untracked-files=all`，识别已有未提交改动。
3. 阅读相关文档：
   - `README.md`: 项目是什么、如何安装、运行、测试、核心命令。
   - `docs/ARCHITECTURE.md`: 当前分层边界和各文件职责。涉及目录边界时必须读。
   - `docs/SCHEMAS.md`: API、数据库、JSON 字段含义。涉及接口或存储字段时必须读。
   - `docs/AI_HANDOFF.md`: 最近状态、已知限制、下一个 agent 接手注意事项。继续他人工作时必须读。
   - `docs/ROADMAP.md`: 当前阶段和下一步方向。涉及产品方向时必须读。
   - `docs/DECISIONS.md`: 已做技术决策及原因。涉及技术选型时必须读。
   - `CLAUDE.md`: Claude Code 专用上下文。除非用户要求，不修改。
4. 阅读相关代码，不要凭文档假设代码一定一致。
5. 如果新增文件/目录，判断是否应该加入 `.gitignore`。

## Required After Coding

1. 更新测试或新增测试，覆盖可观察行为变化。
2. 运行相关测试；默认优先运行：

```bash
venv/bin/python -m pytest
```

3. 如果改了依赖，同步更新：
   - `requirement.txt`
   - `pyproject.toml`
4. 如果改了 API 输入/输出、数据库字段、JSON 结构，同步更新 `docs/SCHEMAS.md`。
5. 如果改了目录分层、模块职责、调用边界，同步更新 `docs/ARCHITECTURE.md`。
6. 如果推进了 MVP 阶段或改变产品/技术下一步，同步更新 `docs/ROADMAP.md`。
7. 如果做了技术选型或放弃某个方案，同步更新 `docs/DECISIONS.md`。
8. 如果这次改动会影响下一个 agent 接手，同步更新 `docs/AI_HANDOFF.md`。
9. 最终回复中说明：
   - 改了哪些文件。
   - 跑了哪些测试。
   - 是否改了依赖。
   - 是否改了文档。
   - 有哪些已知限制。

## Current Commands

安装依赖：

```bash
venv/bin/pip install -r requirement.txt
```

运行服务：

```bash
venv/bin/uvicorn app.main:app --reload
```

初始化数据库：

```bash
export DATABASE_URL='mysql+pymysql://user:password@localhost:3306/pose_app?charset=utf8mb4'
venv/bin/python -m app.main init-db
```

运行测试：

```bash
venv/bin/python -m pytest
```

调用主接口：

```bash
curl -X POST "http://127.0.0.1:8000/analyze_image_path" \
  -H "Content-Type: application/json" \
  -d '{"image_url":"images/example.jpg","should_store":true}'
```
