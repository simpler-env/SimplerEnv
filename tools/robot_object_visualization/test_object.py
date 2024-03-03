import argparse
from pathlib import Path
import time

import numpy as np
import sapien.core as sapien
from sapien.utils.viewer import Viewer


def build_actor(
    model_dir: str,
    scene: sapien.Scene,
    scale: float = 1.0,
    physical_material: sapien.PhysicalMaterial = None,
    density=1000,
):
    builder = scene.create_actor_builder()
    model_dir = Path(model_dir)

    collision_file = str(model_dir / "collision.obj")
    builder.add_multiple_collisions_from_file(
        filename=collision_file,
        scale=[scale] * 3,
        material=physical_material,
        density=density,
    )

    visual_file = str(model_dir / "textured.dae")
    if not Path(visual_file).exists():
        visual_file = str(model_dir / "textured.glb")
    builder.add_visual_from_file(filename=visual_file, scale=[scale] * 3)

    actor = builder.build()
    return actor


def demo(model_dir):
    engine = sapien.Engine()
    renderer = sapien.SapienRenderer()
    engine.set_renderer(renderer)

    scene_config = sapien.SceneConfig()
    scene = engine.create_scene(scene_config)
    scene.set_timestep(1 / 500.0)
    scene.add_ground(0)

    scene.set_ambient_light([0.5, 0.5, 0.5])
    scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5])

    viewer = Viewer(renderer)
    viewer.set_scene(scene)
    viewer.set_camera_xyz(x=-2, y=0, z=1)
    viewer.set_camera_rpy(r=0, p=-0.3, y=0)

    obj = build_actor(
        model_dir,
        scene,
        scale=1.0,
        density=1000,
        physical_material=scene.create_physical_material(static_friction=2.0, dynamic_friction=2.0, restitution=0.0),
    )
    obj.set_pose(sapien.Pose([0, 0, 1], [1, 0, 0, 0]))

    camera = scene.add_camera(
        name="camera",
        width=int(848),
        height=int(480),
        fovy=np.deg2rad(78.0),  # D435 fovy
        near=0.1,
        far=10.0,
    )
    camera.set_focal_lengths(605.12, 604.91)
    camera.set_principal_point(424.59, 236.67)
    camera.set_parent(parent=obj, keep_pose=False)

    while not viewer.closed:
        for _ in range(4):  # render every 4 steps
            scene.step()
        scene.update_render()
        viewer.render()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir",
        type=str,
        default="ManiSkill2_real2sim/data/custom/models/opened_coke_can",
    )
    args = parser.parse_args()

    demo(args.model_dir)


if __name__ == "__main__":
    main()
