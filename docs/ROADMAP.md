# Roadmap

本文档记录产品和技术迭代方向。它不是详细任务清单，而是帮助维护者判断“当前阶段做什么、下一步做什么、长期不急着做什么”。

## Current: MVP 实现阶段

当前 MVP 聚焦一条主链路：

```text
图片路径 -> 姿势分析 -> 背景分析 -> 可选入库
```

已经实现：

- `POST /analyze_image_path`
- MediaPipe Pose 姿势骨架分析
- CLIP zero-shot 背景分类
- MySQL 三表存储：
  - `app_images`
  - `poses`
  - `pose_keypoints`
- `should_store` 控制是否入库
- 入库失败时仍返回分析结果
- JSON 姿势库读取
- JSON 姿势库参考图片 ingest 脚本
- 基础测试覆盖
- 保留 `scripts/` 作为手动维护/调试工具
- 保留 `tests/` 作为回归测试

当前明确不做：

- 姿势推荐 API
- rule-based / embedding recommender
- LLM 摄影建议
- 实时摄像头
- pose card 渲染
- 背景分析落库

## Next

建议下一步在和维护者确认后推进：

- 明确 `image_url` 是否只代表本地路径，还是需要支持远程 URL / 对象存储 URL。
- 明确是否要增加 HTTP 上传入口。如果增加，应复用 `image_analysis_service.analyze_image_path` 或抽象成共享 image source loader，避免重新出现两条分析链路。
- 明确背景分析是否需要落库。如果需要，需要设计第四张表或扩展 `app_images`，并更新 `docs/SCHEMAS.md`。
- 明确姿势库 `poses.json` 是否继续作为源数据，还是迁移到数据库。
- 明确 `bbox`、`confidence`、`raw_result` 是否满足后续产品检索需求。
- 为 MySQL 增加真实集成测试或本地 docker-compose 测试环境。
- 评估是否引入 Alembic 做迁移管理。

## Later

长期方向，暂不在当前 MVP 中实现：

- 姿势推荐 API。
- embedding match。
- 基于姿势和背景的拍照建议。
- 多人姿势检测。
- 图片上传、压缩、对象存储。
- 后台任务队列，异步分析大图。
- 管理端姿势库维护界面。
- pose card 可视化。
