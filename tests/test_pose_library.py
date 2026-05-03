from app.pose_library.loader import load_pose_library


def test_load_pose_library_returns_valid_poses() -> None:
    poses = load_pose_library()

    assert len(poses) >= 3
    assert all(pose.id for pose in poses)
    assert all(pose.scene_tags for pose in poses)
    assert all(pose.style_tags for pose in poses)
