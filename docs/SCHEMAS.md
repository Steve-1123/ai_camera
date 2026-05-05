# Schemas

本文档记录当前 MVP 的两个核心 schema：

1. 姿势分析 + 背景分析的 HTTP 输出。
2. MySQL 三张表的存储字段。

## 1. 姿势分析 + 背景分析输出 Schema

接口：

```text
POST /analyze_image_path
```

入参。当前唯一 HTTP 主入口使用图片路径，不接收 multipart 上传：

```json
{
  "image_url": "images/example.jpg",
  "should_store": true,
  "source": "upload",
  "detector": "local_pose_estimator"
}
```

字段说明：

- `image_url`: 图片存储路径。当前实现按本地文件路径读取。
- `should_store`: 是否把图片、姿势、关键点写入 MySQL。
- `source`: 图片来源，默认 `upload`。
- `detector`: 姿势检测器名称，默认 `local_pose_estimator`。

输出：

```json
{
  "image_info": {
    "filename": "example.jpg",
    "width": 1080,
    "height": 1440
  },
  "pose": {
    "has_person": true,
    "landmark_count": 33,
    "landmarks": [
      {
        "name": "left_shoulder",
        "x": 0.4,
        "y": 0.3,
        "z": -0.1,
        "visibility": 0.9
      }
    ]
  },
  "scene": {
    "primary_category": "street",
    "candidates": [
      {
        "label": "street",
        "score": 0.82
      }
    ],
    "model_name": "clip:openai/clip-vit-base-patch32"
  },
  "storage": {
    "requested": true,
    "image_id": 1,
    "analysis_status": "analyzed",
    "pose_count": 1,
    "keypoint_count": 33,
    "error": null
  }
}
```

字段说明：

- `image_info`: 图片基础信息。
  - `filename`: 图片文件名。
  - `width`: 图片宽度。
  - `height`: 图片高度。
- `pose`: 姿势分析结果。
  - `has_person`: 是否检测到人物。
  - `landmark_count`: 关键点数量。
  - `landmarks`: MediaPipe Pose normalized landmarks。
  - `x` / `y`: 归一化坐标，范围 `[0.0, 1.0]`。
  - `z`: MediaPipe 相对深度。
  - `visibility`: 关键点可见性，范围 `[0.0, 1.0]`。
- `scene`: 背景/场景分析结果。
  - `primary_category`: 最可能的场景类别。
  - `candidates`: top 场景候选。
  - `model_name`: 实际使用的 CLIP 模型，例如 `clip:openai/clip-vit-base-patch32`。
- `storage`: 入库结果。
  - `requested`: 本次是否请求入库。
  - `image_id`: 入库后的 `app_images.id`。
  - `analysis_status`: 入库状态，通常为 `analyzed` 或 `failed`。
  - `pose_count`: 写入 `poses` 的数量。
  - `keypoint_count`: 写入 `pose_keypoints` 的数量。
  - `error`: 入库失败原因。注意：入库失败时仍会尽量返回分析结果。

## 2. 数据库三张表存储字段

当前只确认存储三张表：

- 图片表：`app_images`
- 姿势表：`poses`
- 姿势骨架关键点表：`pose_keypoints`

背景/场景分析本次只返回，不落库。

### app_images

图片表，一张待分析或已分析图片一行。

| 字段 | 类型 | 说明 |
|---|---|---|
| `id` | BIGINT | 自增主键 |
| `source` | VARCHAR(50) | 图片来源，默认 `upload` |
| `image_url` | TEXT | 图片路径，不能为空 |
| `compressed_image_url` | TEXT | 压缩图路径，可为空 |
| `width` | INT | 图片宽度，可为空 |
| `height` | INT | 图片高度，可为空 |
| `mime_type` | VARCHAR(100) | MIME 类型，可为空 |
| `file_size_bytes` | BIGINT | 文件大小，可为空 |
| `analysis_status` | VARCHAR(30) | 分析状态：`pending` / `analyzed` / `failed` |
| `analysis_error` | TEXT | 分析或入库错误，可为空 |
| `created_at` | DATETIME | 创建时间 |
| `updated_at` | DATETIME | 更新时间 |

### poses

姿势表，一张图片可以有 0 个或多个姿势。当前 MediaPipe Pose 只返回单人姿势，所以通常是 0 或 1 行。

| 字段 | 类型 | 说明 |
|---|---|---|
| `id` | BIGINT | 自增主键 |
| `image_id` | BIGINT | 外键，关联 `app_images.id`，级联删除 |
| `detector` | VARCHAR(50) | 检测器名称，默认 `local_pose_estimator` |
| `pose_name` | VARCHAR(100) | 姿势名称，可为空 |
| `pose_description` | TEXT | 姿势描述，可为空 |
| `confidence` | FLOAT | 姿势置信度，当前使用关键点 visibility 均值 |
| `bbox` | JSON | 姿势框，格式 `{x, y, w, h}`，坐标为归一化坐标 |
| `raw_result` | JSON | 原始姿势检测结果 |
| `created_at` | DATETIME | 创建时间 |

### pose_keypoints

姿势骨架关键点表，一个 pose 对应多个关键点。

| 字段 | 类型 | 说明 |
|---|---|---|
| `id` | BIGINT | 自增主键 |
| `pose_id` | BIGINT | 外键，关联 `poses.id`，级联删除 |
| `keypoint_name` | VARCHAR(100) | 关键点名称，例如 `left_shoulder` |
| `x` | FLOAT | 归一化 x 坐标，可为空 |
| `y` | FLOAT | 归一化 y 坐标，可为空 |
| `confidence` | FLOAT | 当前使用 MediaPipe landmark visibility |
| `created_at` | DATETIME | 创建时间 |
